package jp.axinc.ailia_kotlin

import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

import axip.ailia_speech.AiliaSpeech
import axip.ailia_speech.AiliaSpeechText

class AiliaSpeechSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    interface DownloadListener {
        fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onComplete()
        fun onError(error: String)
    }

    private var speech: AiliaSpeech? = null
    private var isInitialized = false
    var modelDir: String = ""

    private fun downloadFile(urlStr: String, fileName: String, listener: DownloadListener? = null): String {
        val dir = modelDir
        if (dir.isEmpty()) throw IllegalStateException("modelDir not set")
        val path = "$dir/$fileName"
        val file = File(path)
        if (file.exists()) {
            if (file.canRead()) {
                Log.i(TAG, "Model file already exists and readable: $path (${file.length()} bytes)")
                return path
            } else {
                Log.w(TAG, "Model file exists but not readable, re-downloading: $path")
                file.delete()
            }
        }
        File(path).parentFile?.mkdirs()
        val tmpFile = File("$path.tmp")
        val url = URL(urlStr)
        val connection = url.openConnection() as HttpURLConnection
        connection.connectTimeout = 30000
        connection.readTimeout = 60000
        connection.connect()
        val totalBytes = connection.contentLengthLong
        connection.inputStream.use { input ->
            FileOutputStream(tmpFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesDownloaded: Long = 0
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    bytesDownloaded += bytesRead
                    listener?.onProgress(fileName, bytesDownloaded, totalBytes)
                }
            }
        }
        tmpFile.renameTo(File(path))
        return path
    }

    fun downloadModel(listener: DownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting speech model download/check...")
            downloadFile(
                "https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.opt3.onnx",
                "encoder_tiny.onnx",
                listener
            )
            downloadFile(
                "https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt3.onnx",
                "decoder_tiny.onnx",
                listener
            )
            listener?.onComplete()
            Log.i(TAG, "Speech model download/check complete")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Speech model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initializeSpeech(envId: Int = -1): Boolean {
        if (isInitialized) {
            releaseSpeech()
        }

        return try {
            val dir = modelDir
            val encoderPath = "$dir/encoder_tiny.onnx"
            val decoderPath = "$dir/decoder_tiny.onnx"

            Log.i(TAG, "Initializing speech with envId=$envId")
            Log.i(TAG, "Encoder: $encoderPath")
            Log.i(TAG, "Decoder: $decoderPath")

            speech = AiliaSpeech(
                envId = envId,
                task = AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE
            )
            speech?.openModel(encoderPath, decoderPath, AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_TINY)
            isInitialized = true
            Log.i(TAG, "Speech initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize speech: ${e.javaClass.name}: ${e.message}")
            releaseSpeech()
            false
        }
    }

    fun process(audio: FloatArray, channels: Int, sampleRate: Int): String {
        Log.i(TAG, "Speech process: audio.size=${audio.size}, channels=$channels, sampleRate=$sampleRate, samples=${audio.size / channels}")
        val pushResult = speech?.pushInputData(audio, channels, audio.size / channels, sampleRate)
        Log.i(TAG, "Speech pushInputData result=$pushResult")
        val finalizeResult = speech?.finalizeInputData()
        Log.i(TAG, "Speech finalizeInputData result=$finalizeResult")
        val transcribeResult = speech?.transcribe()
        Log.i(TAG, "Speech transcribe result=$transcribeResult")
        if (transcribeResult != null && transcribeResult != 0) {
            val errorDetail = speech?.getErrorDetail()
            Log.e(TAG, "Speech transcribe error detail: $errorDetail")
        }
        val count: Int? = speech?.getTextCount()
        Log.i(TAG, "Speech getTextCount=$count")
        if (count == null) {
            return ""
        }
        var ret = ""
        for (i in 0 until count) {
            val text: AiliaSpeechText? = speech?.getText(i)
            if (text == null) {
                continue
            }
            Log.i(TAG, "Speech text[$i]: '${text.text}' confidence=${text.confidence}")
            ret = ret + text.text + "\n"
        }
        speech?.resetTranscribeState()
        Log.i(TAG, "Speech process result: '$ret'")
        return ret
    }

    fun releaseSpeech() {
        try {
            speech?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing speech: ${e.javaClass.name}: ${e.message}")
        } finally {
            speech = null
            isInitialized = false
            Log.i(TAG, "Speech released")
        }
    }
}
