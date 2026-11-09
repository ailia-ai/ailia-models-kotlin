package jp.axinc.ailia_kotlin

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * マイク入力を16kHzモノラルのWAVファイルに録音する。
 *
 * ailia LLMのマルチモーダル入力はファイルパスで音声を受け取るため、
 * 録音結果はそのまま渡せるWAV(PCM 16bit)として保存する。
 */
class WavRecorder(
    /**
     * 録音の上限(秒)。上限に達すると録音を終了して[RecordingListener.onCompleted]を呼ぶ。
     * nullの場合は[stopRecording]が呼ばれるまで無制限に録音する。
     */
    private val maxRecordingSeconds: Int? = MAX_RECORDING_SECONDS,
    /**
     * falseの場合、録音したPCMを保持せずWAVも書き出さない(波形は[RecordingListener.onWaveform]で受け取る)。
     * [RecordingListener.onCompleted]には空の音声が渡され、ファイルは作成されない。長時間録音のメモリ対策。
     */
    private val retainAudio: Boolean = true,
) {
    interface RecordingListener {
        /** 録音中の波形。UIの波形表示を更新するために定期的に呼ばれる。 */
        fun onWaveform(chunk: FloatArray, sampleRate: Int)

        /** 録音完了。[file]はWAVとして書き出し済み。 */
        fun onCompleted(file: File, audio: FloatArray, sampleRate: Int)

        fun onError(error: String)
    }

    private val recording = AtomicBoolean(false)
    private val cancelled = AtomicBoolean(false)
    private var audioRecord: AudioRecord? = null
    private var recordingExecutor: ExecutorService? = null

    val isRecording: Boolean get() = recording.get()

    companion object {
        private const val TAG = "WavRecorder"
        const val SAMPLE_RATE = 16_000
        const val MAX_RECORDING_SECONDS = 30
    }

    @SuppressLint("MissingPermission")
    fun startRecording(outputFile: File, listener: RecordingListener): Boolean {
        if (!recording.compareAndSet(false, true)) return false
        cancelled.set(false)

        val minimumBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val bufferBytes = maxOf(minimumBuffer, SAMPLE_RATE * Short.SIZE_BYTES / 2)

        return try {
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferBytes,
            )
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "Failed to initialize AudioRecord" }
            audioRecord = recorder
            recordingExecutor = Executors.newSingleThreadExecutor { runnable ->
                Thread({
                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO)
                    runnable.run()
                }, "ailia-alm-recording")
            }
            recorder.startRecording()
            recordingExecutor?.execute { recordLoop(recorder, outputFile, listener) }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            recording.set(false)
            releaseRecorder()
            listener.onError(e.message ?: "Failed to start recording")
            false
        }
    }

    private fun recordLoop(recorder: AudioRecord, outputFile: File, listener: RecordingListener) {
        val chunkSize = SAMPLE_RATE / 10
        val buffer = ShortArray(chunkSize)
        val chunks = mutableListOf<ShortArray>()
        var sampleCount = 0L
        val maximumSamples = maxRecordingSeconds?.let { SAMPLE_RATE.toLong() * it } ?: Long.MAX_VALUE
        try {
            while (recording.get() && sampleCount < maximumSamples) {
                val count = recorder.read(buffer, 0, buffer.size)
                if (count < 0) error("AudioRecord read failed: $count")
                if (count == 0) continue
                if (retainAudio) chunks += buffer.copyOf(count)
                sampleCount += count
                listener.onWaveform(FloatArray(count) { buffer[it] / 32768f }, SAMPLE_RATE)
            }
        } catch (e: Exception) {
            // AudioRecord.stop()は待機中のreadをエラーコードで中断させることがあるため、
            // 継続中に発生した失敗だけを通知する。
            if (!cancelled.get() && recording.get()) {
                listener.onError(e.message ?: "Recording failed")
            }
        } finally {
            recording.set(false)
            releaseRecorder()
            recordingExecutor?.shutdown()
            recordingExecutor = null
            if (!cancelled.get() && sampleCount > 0 && !retainAudio) {
                listener.onCompleted(outputFile, FloatArray(0), SAMPLE_RATE)
            } else if (!cancelled.get() && sampleCount > 0) {
                try {
                    val pcm = ShortArray(sampleCount.toInt())
                    var offset = 0
                    chunks.forEach { chunk ->
                        System.arraycopy(chunk, 0, pcm, offset, chunk.size)
                        offset += chunk.size
                    }
                    writeWav(outputFile, pcm)
                    listener.onCompleted(outputFile, FloatArray(pcm.size) { pcm[it] / 32768f }, SAMPLE_RATE)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to save recording", e)
                    listener.onError(e.message ?: "Failed to save recording")
                }
            }
        }
    }

    fun stopRecording() {
        recording.set(false)
        try {
            audioRecord?.stop()
        } catch (_: Exception) {
        }
    }

    fun cancelRecording() {
        cancelled.set(true)
        stopRecording()
    }

    private fun releaseRecorder() {
        try {
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release AudioRecord: ${e.message}")
        }
        audioRecord = null
    }

    /** PCM 16bitモノラルのWAVファイルを書き出す。 */
    private fun writeWav(file: File, pcm: ShortArray) {
        val dataBytes = pcm.size * Short.SIZE_BYTES
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(36 + dataBytes)
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16) // PCM fmt chunk size
            putShort(1) // PCM
            putShort(1) // mono
            putInt(SAMPLE_RATE)
            putInt(SAMPLE_RATE * Short.SIZE_BYTES) // byte rate
            putShort(Short.SIZE_BYTES.toShort()) // block align
            putShort(16) // bits per sample
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(dataBytes)
        }
        val body = ByteBuffer.allocate(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        pcm.forEach { body.putShort(it) }
        FileOutputStream(file).use { output ->
            output.write(header.array())
            output.write(body.array())
            output.fd.sync()
        }
    }
}
