package jp.axinc.ailia_kotlin

import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

import axip.ailia_speech.AiliaSpeech
import axip.ailia_speech.AiliaSpeechText

/**
 * Enum defining available speech recognition models.
 */
enum class SpeechModelType(
    val displayName: String,
    val encoderUrl: String,
    val encoderFileName: String,
    val decoderUrl: String,
    val decoderFileName: String,
    val modelTypeId: Int,
    val needsDecoder: Boolean = true
) {
    WHISPER_TINY(
        "Whisper Tiny",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.opt3.onnx",
        "encoder_tiny.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt3.onnx",
        "decoder_tiny.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_TINY
    ),
    WHISPER_BASE(
        "Whisper Base",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_base.opt3.onnx",
        "encoder_base.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_base_fix_kv_cache.opt3.onnx",
        "decoder_base.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_BASE
    ),
    WHISPER_SMALL(
        "Whisper Small",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_small.opt3.onnx",
        "encoder_small.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_small_fix_kv_cache.opt3.onnx",
        "decoder_small.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_SMALL
    ),
    SENSEVOICE_SMALL(
        "SenseVoice Small",
        "https://storage.googleapis.com/ailia-models/sensevoice/sensevoice_small.onnx",
        "sensevoice_small.onnx",
        "",
        "",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL,
        needsDecoder = false
    )
}

class AiliaSpeechSample {
    interface DownloadListener {
        fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onComplete()
        fun onError(error: String)
    }

    companion object {
        private const val TAG = "AILIA_Main"
        private const val VAD_URL = "https://storage.googleapis.com/ailia-models/silero-vad/silero_vad_v6_2.onnx"
        private const val VAD_FILE = "silero_vad_v6_2.onnx"
        private const val DIARIZATION_SEGMENTATION_URL = "https://storage.googleapis.com/ailia-models/pyannote-audio/segmentation.onnx"
        private const val DIARIZATION_EMBEDDING_URL = "https://storage.googleapis.com/ailia-models/pyannote-audio/speaker-embedding.onnx"
        private const val DIARIZATION_SEGMENTATION_FILE = "segmentation.onnx"
        private const val DIARIZATION_EMBEDDING_FILE = "speaker-embedding.onnx"
    }

    private var speech: AiliaSpeech? = null
    private var isInitialized = false
    var modelDir: String = ""
    var currentModelType: SpeechModelType = SpeechModelType.WHISPER_TINY
    var diarizationEnabled: Boolean = false

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

    /**
     * Downloads model files for the specified (or current) speech model type.
     * Always downloads Silero VAD model for all modes.
     * If diarizationEnabled is true, also downloads pyannote-audio segmentation and embedding models.
     */
    fun downloadModel(modelType: SpeechModelType = currentModelType, listener: DownloadListener? = null): Boolean {
        currentModelType = modelType
        return try {
            Log.i(TAG, "Starting speech model download/check for ${modelType.displayName}...")
            downloadFile(
                modelType.encoderUrl,
                modelType.encoderFileName,
                listener
            )
            if (modelType.needsDecoder) {
                downloadFile(
                    modelType.decoderUrl,
                    modelType.decoderFileName,
                    listener
                )
            }
            // Always download VAD model
            Log.i(TAG, "Downloading VAD model...")
            downloadFile(VAD_URL, VAD_FILE, listener)
            if (diarizationEnabled) {
                Log.i(TAG, "Downloading diarization models...")
                downloadFile(DIARIZATION_SEGMENTATION_URL, DIARIZATION_SEGMENTATION_FILE, listener)
                downloadFile(DIARIZATION_EMBEDDING_URL, DIARIZATION_EMBEDDING_FILE, listener)
            }
            listener?.onComplete()
            Log.i(TAG, "Speech model download/check complete for ${modelType.displayName}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Speech model download failed for ${modelType.displayName}", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    /**
     * Initializes the speech engine with the current model type.
     *
     * @param envId Environment ID (-1 for auto)
     * @param liveMode If true, initializes with AILIA_SPEECH_FLAG_LIVE for streaming mic input
     */
    fun initializeSpeech(envId: Int = -1, liveMode: Boolean = false): Boolean {
        if (isInitialized) {
            releaseSpeech()
        }

        return try {
            val dir = modelDir
            val encoderPath = "$dir/${currentModelType.encoderFileName}"
            val decoderPath = if (currentModelType.needsDecoder) {
                "$dir/${currentModelType.decoderFileName}"
            } else {
                ""
            }

            // When diarization is enabled, disable LIVE flag (diarization requires non-live mode)
            // Streaming still works without the LIVE flag
            val useLiveFlag = liveMode && !diarizationEnabled
            val flags = if (useLiveFlag) AiliaSpeech.AILIA_SPEECH_FLAG_LIVE else AiliaSpeech.AILIA_SPEECH_FLAG_NONE

            Log.i(TAG, "Initializing speech with envId=$envId, model=${currentModelType.displayName}, liveMode=$liveMode, diarization=$diarizationEnabled, useLiveFlag=$useLiveFlag")
            Log.i(TAG, "Encoder: $encoderPath")
            Log.i(TAG, "Decoder: $decoderPath")

            speech = AiliaSpeech(
                envId = envId,
                task = AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE,
                flags = flags
            )
            speech?.openModel(encoderPath, decoderPath, currentModelType.modelTypeId)

            // Always open VAD (Silero VAD)
            val vadPath = "$dir/$VAD_FILE"
            Log.i(TAG, "Opening VAD: $vadPath")
            val vadResult = speech?.openVad(vadPath, AiliaSpeech.AILIA_SPEECH_VAD_TYPE_SILERO)
            Log.i(TAG, "VAD openVad result=$vadResult")

            // Open diarization if enabled
            if (diarizationEnabled) {
                val segmentationPath = "$dir/$DIARIZATION_SEGMENTATION_FILE"
                val embeddingPath = "$dir/$DIARIZATION_EMBEDDING_FILE"
                Log.i(TAG, "Opening diarization: segmentation=$segmentationPath, embedding=$embeddingPath")
                val diarResult = speech?.openDiarization(
                    segmentationPath, embeddingPath,
                    AiliaSpeech.AILIA_SPEECH_DIARIZATION_TYPE_PYANNOTE_AUDIO
                )
                Log.i(TAG, "Diarization openDiarization result=$diarResult")
            }

            isInitialized = true
            Log.i(TAG, "Speech initialized successfully with envId=$envId, model=${currentModelType.displayName}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize speech: ${e.javaClass.name}: ${e.message}")
            releaseSpeech()
            false
        }
    }

    /**
     * Processes audio from a WAV file (non-live mode).
     * Calls pushInputData, finalizeInputData, transcribe, and returns text.
     */
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
        return collectTextResults()
    }

    /**
     * Pushes live audio data for streaming recognition (live mode).
     * Does NOT call finalizeInputData - use finalizeLiveAudio() when recording stops.
     */
    fun pushLiveAudio(audio: FloatArray, channels: Int, sampleRate: Int): String {
        val pushResult = speech?.pushInputData(audio, channels, audio.size / channels, sampleRate)
        Log.d(TAG, "Speech pushLiveAudio: pushInputData result=$pushResult, samples=${audio.size / channels}")
        val transcribeResult = speech?.transcribe()
        Log.d(TAG, "Speech pushLiveAudio: transcribe result=$transcribeResult")
        if (transcribeResult != null && transcribeResult != 0) {
            val errorDetail = speech?.getErrorDetail()
            Log.e(TAG, "Speech pushLiveAudio transcribe error: $errorDetail")
        }
        return collectTextResults()
    }

    /**
     * Finalizes live audio input and returns final transcription.
     * Call this when mic recording stops.
     */
    fun finalizeLiveAudio(): String {
        val finalizeResult = speech?.finalizeInputData()
        Log.i(TAG, "Speech finalizeLiveAudio: finalizeInputData result=$finalizeResult")
        val transcribeResult = speech?.transcribe()
        Log.i(TAG, "Speech finalizeLiveAudio: transcribe result=$transcribeResult")
        if (transcribeResult != null && transcribeResult != 0) {
            val errorDetail = speech?.getErrorDetail()
            Log.e(TAG, "Speech finalizeLiveAudio transcribe error: $errorDetail")
        }
        return collectTextResults()
    }

    /**
     * Collects text results from the speech engine.
     * When diarization is enabled, prefixes each line with speaker ID.
     */
    private fun collectTextResults(): String {
        val count: Int? = speech?.getTextCount()
        Log.i(TAG, "Speech getTextCount=$count")
        if (count == null || count == 0) {
            return ""
        }
        val sb = StringBuilder()
        for (i in 0 until count) {
            val text: AiliaSpeechText? = speech?.getText(i)
            if (text == null) {
                continue
            }
            if (diarizationEnabled && text.speakerId.toLong() and 0xFFFFFFFFL != AiliaSpeech.AILIA_SPEECH_SPEAKER_ID_UNKNOWN.toLong() and 0xFFFFFFFFL) {
                Log.i(TAG, "Speech text[$i]: speaker=#${text.speakerId} '${text.text}' confidence=${text.confidence}")
                sb.append("[Speaker ${text.speakerId}] ${text.text}\n")
            } else {
                Log.i(TAG, "Speech text[$i]: '${text.text}' confidence=${text.confidence}")
                sb.append(text.text).append("\n")
            }
        }
        speech?.resetTranscribeState()
        val result = sb.toString()
        Log.i(TAG, "Speech result: '$result'")
        return result
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
