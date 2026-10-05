package jp.axinc.ailia_kotlin

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

data class SpeakerEmbeddingResult(
    val embedding: FloatArray,
    val speechDurationMs: Long,
    val processingTimeMs: Long,
)

data class SpeakerVerificationResult(
    val metrics: SpeakerVerificationMetrics,
    val speechDurationMs: Long,
    val processingTimeMs: Long,
)

/**
 * Standalone WeSpeaker verification sample.
 *
 * Both WAV and microphone audio are converted to 16 kHz mono and separated with
 * Silero VAD v6 before WeSpeaker feature extraction. The recorder returns audio
 * only; enrollment policy and UI remain outside this copyable model sample.
 */
class AiliaWeSpeakerSample(private val modelDirectory: File) {
    interface RecordingListener {
        fun onWaveform(samples: FloatArray, sampleRate: Int)
        fun onCompleted(audio: FloatArray, sampleRate: Int)
        fun onError(error: String)
    }

    companion object {
        private const val TAG = "AILIA_WeSpeaker"
        private const val WESPEAKER_MODEL_URL =
            "https://storage.googleapis.com/ailia-models/wespeaker/voxceleb_resnet34.onnx"
        private const val WESPEAKER_MODEL_FILE = "voxceleb_resnet34.onnx"
        private const val WESPEAKER_PROTO_URL =
            "https://storage.googleapis.com/ailia-models/wespeaker/voxceleb_resnet34.onnx.prototxt"
        private const val WESPEAKER_PROTO_FILE = "voxceleb_resnet34.onnx.prototxt"
        private const val VAD_MODEL_URL =
            "https://storage.googleapis.com/ailia-models/silero-vad/silero_vad_v6.onnx"
        private const val VAD_MODEL_FILE = "silero_vad_v6.onnx"
        private const val VAD_PROTO_URL =
            "https://storage.googleapis.com/ailia-models/silero-vad/silero_vad_v6.onnx.prototxt"
        private const val VAD_PROTO_FILE = "silero_vad_v6.onnx.prototxt"
        private const val VAD_WINDOW_SIZE = 512
        private const val VAD_CONTEXT_SIZE = 64
        private const val VAD_STATE_SIZE = 2 * 1 * 128
        private const val EMBEDDING_SIZE = 256
        private const val MAX_RECORDING_SECONDS = 30

        init {
            System.loadLibrary("ailia")
        }
    }

    private var speakerModel: AiliaModel? = null
    private var vadModel: AiliaModel? = null
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var ortSpeakerSession: OrtSession? = null
    private var ortVadSession: OrtSession? = null
    @Volatile private var initialized = false

    private val recording = AtomicBoolean(false)
    private val cancelRecording = AtomicBoolean(false)
    private var audioRecord: AudioRecord? = null
    private var recordingExecutor: ExecutorService? = null
    private var audioTrack: AudioTrack? = null

    val isRecording: Boolean
        get() = recording.get()

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            val files = listOf(
                ModelFileSpec(WESPEAKER_PROTO_URL, WESPEAKER_PROTO_FILE),
                ModelFileSpec(WESPEAKER_MODEL_URL, WESPEAKER_MODEL_FILE),
                ModelFileSpec(VAD_PROTO_URL, VAD_PROTO_FILE),
                ModelFileSpec(VAD_MODEL_URL, VAD_MODEL_FILE),
            )
            files.forEach { spec ->
                check(ModelDownloader.downloadFile(modelDirectory, spec, listener) != null)
            }
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initialize(envId: Int): Boolean {
        releaseModels()
        return try {
            speakerModel = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                File(modelDirectory, WESPEAKER_PROTO_FILE).absolutePath,
                File(modelDirectory, WESPEAKER_MODEL_FILE).absolutePath,
            )
            vadModel = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                File(modelDirectory, VAD_PROTO_FILE).absolutePath,
                File(modelDirectory, VAD_MODEL_FILE).absolutePath,
            )
            initialized = true
            true
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            releaseModels()
            false
        }
    }

    fun initializeOnnxRuntime(): Boolean {
        releaseModels()
        return try {
            val options = OrtSession.SessionOptions()
            try {
                ortSpeakerSession = ortEnvironment.createSession(
                    File(modelDirectory, WESPEAKER_MODEL_FILE).absolutePath,
                    options,
                )
                ortVadSession = ortEnvironment.createSession(
                    File(modelDirectory, VAD_MODEL_FILE).absolutePath,
                    options,
                )
            } finally {
                options.close()
            }
            initialized = true
            Log.i(TAG, "WeSpeaker and Silero VAD initialized with ONNX Runtime CPU")
            true
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime initialization failed", e)
            releaseModels()
            false
        }
    }

    fun extractEmbedding(
        interleavedAudio: FloatArray,
        channels: Int,
        sampleRate: Int,
    ): SpeakerEmbeddingResult {
        check(initialized) { "WeSpeaker is not initialized" }
        val startTime = System.nanoTime()
        val mono = SpeakerVerificationAudio.toMono16k(interleavedAudio, channels, sampleRate)
        require(mono.isNotEmpty()) { "Audio is empty" }
        val segments = detectSpeech(mono)
        require(segments.isNotEmpty()) { "Silero VAD did not detect speech" }
        val speech = SpeakerVerificationAudio.concatenateSegments(mono, segments)
        require(speech.size >= SpeakerVerificationAudio.SAMPLE_RATE / 2) {
            "At least 0.5 seconds of speech is required"
        }

        val features = SpeakerVerificationAudio.computeFbank(speech)
        val frameCount = features.size / 80
        val embedding = ortSpeakerSession?.let { session ->
            val tensor = createFloatTensor(
                ortEnvironment,
                features,
                longArrayOf(1, frameCount.toLong(), 80),
            )
            try {
                runFloatTensor(session, mapOf("feats" to tensor), "embs")
            } finally {
                tensor.close()
            }
        } ?: run {
            val model = checkNotNull(speakerModel)
            val inputIndex = Ailia.FindBlobIndexByName(model.handle, "feats")
            model.setInputBlobShapeND(intArrayOf(1, frameCount, 80), inputIndex)
            model.setInputBlobData(features, features.size * Float.SIZE_BYTES, inputIndex)
            model.update()
            FloatArray(EMBEDDING_SIZE).also {
                model.getBlobData(
                    it,
                    it.size * Float.SIZE_BYTES,
                    Ailia.FindBlobIndexByName(model.handle, "embs"),
                )
            }
        }
        return SpeakerEmbeddingResult(
            embedding = embedding,
            speechDurationMs = speech.size * 1000L / SpeakerVerificationAudio.SAMPLE_RATE,
            processingTimeMs = (System.nanoTime() - startTime) / 1_000_000,
        )
    }

    fun verify(
        referenceEmbedding: FloatArray,
        interleavedAudio: FloatArray,
        channels: Int,
        sampleRate: Int,
    ): SpeakerVerificationResult {
        val result = extractEmbedding(interleavedAudio, channels, sampleRate)
        return SpeakerVerificationResult(
            metrics = SpeakerVerificationAudio.compare(referenceEmbedding, result.embedding),
            speechDurationMs = result.speechDurationMs,
            processingTimeMs = result.processingTimeMs,
        )
    }

    private fun detectSpeech(audio: FloatArray): List<VadSegment> {
        ortVadSession?.let { session ->
            val probabilities = runOrtSileroVad(
                ortEnvironment,
                session,
                audio,
                VAD_WINDOW_SIZE,
                VAD_CONTEXT_SIZE,
                VAD_STATE_SIZE,
            )
            return SpeakerVerificationAudio.speechSegments(probabilities, audio.size)
        }
        val model = checkNotNull(vadModel)
        val inputIndex = Ailia.FindBlobIndexByName(model.handle, "input")
        val stateIndex = Ailia.FindBlobIndexByName(model.handle, "state")
        val sampleRateIndex = Ailia.FindBlobIndexByName(model.handle, "sr")
        val probabilityIndex = Ailia.FindBlobIndexByName(model.handle, "output")
        val nextStateIndex = Ailia.FindBlobIndexByName(model.handle, "stateN")
        var state = FloatArray(VAD_STATE_SIZE)
        var context = FloatArray(VAD_CONTEXT_SIZE)
        val probabilities = FloatArray((audio.size + VAD_WINDOW_SIZE - 1) / VAD_WINDOW_SIZE)

        model.setInputBlobShapeND(intArrayOf(1, VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE), inputIndex)
        model.setInputBlobShapeND(intArrayOf(2, 1, 128), stateIndex)
        // Silero v6 requires a scalar int64 sample-rate input, while ailia's Java data
        // setter accepts only float arrays. The graph already contains the same int64
        // value (16000) for its sample-rate branch, so copy that typed constant directly.
        // This avoids changing ailia-sdk-jni or losing subnormal raw bits on ARM.
        val sampleRateConstantIndex =
            Ailia.FindBlobIndexByName(model.handle, "Constant_0_output")
        model.copyBlob(sampleRateIndex, sampleRateConstantIndex, model)

        probabilities.indices.forEach { windowIndex ->
            val chunk = FloatArray(VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE)
            System.arraycopy(context, 0, chunk, 0, VAD_CONTEXT_SIZE)
            val audioOffset = windowIndex * VAD_WINDOW_SIZE
            val available = minOf(VAD_WINDOW_SIZE, audio.size - audioOffset)
            System.arraycopy(audio, audioOffset, chunk, VAD_CONTEXT_SIZE, available)
            model.setInputBlobData(chunk, chunk.size * Float.SIZE_BYTES, inputIndex)
            model.setInputBlobData(state, state.size * Float.SIZE_BYTES, stateIndex)
            model.update()

            val probability = FloatArray(1)
            model.getBlobData(probability, Float.SIZE_BYTES, probabilityIndex)
            probabilities[windowIndex] = probability[0]
            val nextState = FloatArray(VAD_STATE_SIZE)
            model.getBlobData(nextState, nextState.size * Float.SIZE_BYTES, nextStateIndex)
            state = nextState
            context = chunk.copyOfRange(chunk.size - VAD_CONTEXT_SIZE, chunk.size)
        }
        return SpeakerVerificationAudio.speechSegments(probabilities, audio.size)
    }

    @SuppressLint("MissingPermission")
    fun startRecording(listener: RecordingListener): Boolean {
        if (!initialized) {
            listener.onError("WeSpeaker is not initialized")
            return false
        }
        if (!recording.compareAndSet(false, true)) return false
        cancelRecording.set(false)

        val sampleRate = SpeakerVerificationAudio.SAMPLE_RATE
        val useFloat = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
        val encoding = if (useFloat) AudioFormat.ENCODING_PCM_FLOAT else AudioFormat.ENCODING_PCM_16BIT
        val bytesPerSample = if (useFloat) Float.SIZE_BYTES else Short.SIZE_BYTES
        val minimumBuffer = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, encoding)
        val bufferBytes = maxOf(minimumBuffer, sampleRate * bytesPerSample / 2)

        return try {
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                encoding,
                bufferBytes,
            )
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "Failed to initialize AudioRecord" }
            audioRecord = recorder
            recordingExecutor = Executors.newSingleThreadExecutor { runnable ->
                Thread({
                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO)
                    runnable.run()
                }, "ailia-wespeaker-recording")
            }
            recorder.startRecording()
            recordingExecutor?.execute { recordLoop(recorder, useFloat, listener) }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            recording.set(false)
            releaseRecorder()
            listener.onError(e.message ?: "Failed to start recording")
            false
        }
    }

    private fun recordLoop(recorder: AudioRecord, useFloat: Boolean, listener: RecordingListener) {
        val chunkSize = SpeakerVerificationAudio.SAMPLE_RATE / 10
        val floatBuffer = FloatArray(chunkSize)
        val shortBuffer = if (useFloat) null else ShortArray(chunkSize)
        val chunks = mutableListOf<FloatArray>()
        var sampleCount = 0
        val maximumSamples = SpeakerVerificationAudio.SAMPLE_RATE * MAX_RECORDING_SECONDS
        try {
            while (recording.get() && sampleCount < maximumSamples) {
                val count = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && useFloat) {
                    recorder.read(floatBuffer, 0, floatBuffer.size, AudioRecord.READ_BLOCKING)
                } else {
                    @Suppress("DEPRECATION")
                    val read = recorder.read(shortBuffer!!, 0, shortBuffer.size)
                    if (read > 0) for (i in 0 until read) floatBuffer[i] = shortBuffer[i] / 32768f
                    read
                }
                if (count < 0) error("AudioRecord read failed: $count")
                if (count == 0) continue
                val chunk = floatBuffer.copyOf(count)
                chunks += chunk
                sampleCount += count
                listener.onWaveform(chunk, SpeakerVerificationAudio.SAMPLE_RATE)
            }
        } catch (e: Exception) {
            // AudioRecord.stop() commonly unblocks a pending read with an error code.
            // Report only failures that happened while the session was meant to continue.
            if (!cancelRecording.get() && recording.get()) {
                listener.onError(e.message ?: "Recording failed")
            }
        } finally {
            recording.set(false)
            releaseRecorder()
            recordingExecutor?.shutdown()
            recordingExecutor = null
            if (!cancelRecording.get() && sampleCount > 0) {
                val audio = FloatArray(sampleCount)
                var offset = 0
                chunks.forEach { chunk ->
                    System.arraycopy(chunk, 0, audio, offset, chunk.size)
                    offset += chunk.size
                }
                listener.onCompleted(audio, SpeakerVerificationAudio.SAMPLE_RATE)
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
        cancelRecording.set(true)
        stopRecording()
    }

    fun playAudio(
        audio: FloatArray,
        sampleRate: Int = SpeakerVerificationAudio.SAMPLE_RATE,
        onComplete: (() -> Unit)? = null,
    ) {
        require(audio.isNotEmpty()) { "Audio is empty" }
        stopPlayback()
        try {
            val pcm16 = ShortArray(audio.size) { index ->
                (audio[index].coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt().toShort()
            }
            val minimumBuffer = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            val bufferBytes = maxOf(minimumBuffer, pcm16.size * Short.SIZE_BYTES)
            val track = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(bufferBytes)
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .build()
            } else {
                @Suppress("DEPRECATION")
                AudioTrack(
                    AudioManager.STREAM_MUSIC,
                    sampleRate,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferBytes,
                    AudioTrack.MODE_STATIC,
                )
            }
            audioTrack = track
            val written = track.write(pcm16, 0, pcm16.size)
            check(written == pcm16.size) { "AudioTrack write failed: $written/${pcm16.size}" }
            track.setNotificationMarkerPosition(pcm16.size)
            track.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(completedTrack: AudioTrack?) {
                    if (audioTrack === completedTrack) {
                        stopPlayback()
                        onComplete?.invoke()
                    }
                }

                override fun onPeriodicNotification(completedTrack: AudioTrack?) = Unit
            })
            track.play()
        } catch (e: Exception) {
            Log.e(TAG, "Audio playback failed", e)
            stopPlayback()
            throw e
        }
    }

    @Synchronized
    fun stopPlayback() {
        val track = audioTrack ?: return
        audioTrack = null
        try {
            if (track.playState == AudioTrack.PLAYSTATE_PLAYING) track.stop()
        } catch (_: Exception) {
        }
        try {
            track.release()
        } catch (_: Exception) {
        }
    }

    fun release() {
        cancelRecording()
        stopPlayback()
        releaseModels()
    }

    private fun releaseModels() {
        try {
            speakerModel?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release WeSpeaker", e)
        }
        try {
            vadModel?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release Silero VAD", e)
        }
        try {
            ortSpeakerSession?.close()
            ortVadSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release ONNX Runtime audio sessions", e)
        }
        speakerModel = null
        vadModel = null
        ortSpeakerSession = null
        ortVadSession = null
        initialized = false
    }

    @Synchronized
    private fun releaseRecorder() {
        try {
            audioRecord?.release()
        } catch (_: Exception) {
        }
        audioRecord = null
    }
}
