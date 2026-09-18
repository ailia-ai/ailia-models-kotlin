package jp.axinc.ailia_kotlin

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

import axip.ailia.AiliaEnvironment
import axip.ailia.AiliaModel
import axip.ailia_speech.AiliaSpeech
import axip.ailia_speech.AiliaSpeechText
import axip.ailia_speech.IntermediateCallback

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
    val needsDecoder: Boolean = true,
    /** onnx本体以外に必要な追加ファイル(外部weightなど)。ファイル名はonnx内の参照名と一致させること。 */
    val extraFiles: List<ModelFileSpec> = emptyList(),
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
    WHISPER_MEDIUM(
        "Whisper Medium",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_medium.opt3.onnx",
        "encoder_medium.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_medium_fix_kv_cache.opt3.onnx",
        "decoder_medium.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_MEDIUM
    ),
    WHISPER_LARGE_V3_TURBO(
        "Whisper Large V3 Turbo",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_turbo.opt.onnx",
        "encoder_turbo.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_turbo_fix_kv_cache.opt.onnx",
        "decoder_turbo.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_LARGE_V3,
        // encoderのweightは外部ファイル参照のため別途ダウンロードする(約2.5GB)
        extraFiles = listOf(
            ModelFileSpec(
                "https://storage.googleapis.com/ailia-models/whisper/encoder_turbo_weights.opt.pb",
                "encoder_turbo_weights.opt.pb"
            )
        ),
    ),
    SENSEVOICE_SMALL(
        "SenseVoice Small",
        "https://storage.googleapis.com/ailia-models/sensevoice/sensevoice_small.onnx",
        "sensevoice_small.onnx",
        "https://storage.googleapis.com/ailia-models/sensevoice/sensevoice_small.model",
        "sensevoice_small.model",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL
    )
}

class AiliaSpeechSample(private val modelDirectory: File) {
    /** マイク録音中にUIへ返すデータだけを定義する。各コールバックはバックグラウンドスレッドで呼ばれる。 */
    interface MicRecordingListener {
        fun onWaveform(samples: FloatArray, sampleRate: Int)
        fun onIntermediateResult(text: String)
        fun onResult(lines: List<String>, isFinal: Boolean, processingTimeMs: Long)
        fun onError(error: String)
    }

    companion object {
        private const val TAG = "AILIA_Main"
        val DEFAULT_MODEL_TYPE: SpeechModelType = SpeechModelType.SENSEVOICE_SMALL
        private const val VAD_URL = "https://storage.googleapis.com/ailia-models/silero-vad/silero_vad_v6_2.onnx"
        private const val VAD_FILE = "silero_vad_v6_2.onnx"
        private const val DIARIZATION_SEGMENTATION_URL = "https://storage.googleapis.com/ailia-models/pyannote-audio/segmentation.onnx"
        private const val DIARIZATION_EMBEDDING_URL = "https://storage.googleapis.com/ailia-models/pyannote-audio/speaker-embedding.onnx"
        private const val DIARIZATION_SEGMENTATION_FILE = "segmentation.onnx"
        private const val DIARIZATION_EMBEDDING_FILE = "speaker-embedding.onnx"
        /** QNN実行時にSenseVoiceの入力シェイプを固定する長さ(秒) */
        private const val QNN_STATIC_INPUT_LENGTH_SEC = 11
    }

    private var speech: AiliaSpeech? = null
    private val speechLock = Any()
    @Volatile
    private var isInitialized = false
    private var audioRecord: AudioRecord? = null
    private var micReadExecutor: ExecutorService? = null
    private var micRecognitionExecutor: ExecutorService? = null
    private val micRecording = AtomicBoolean(false)
    private val micSessionActive = AtomicBoolean(false)
    private val finalizeMicInput = AtomicBoolean(true)
    private val micProcessingTimeNanos = AtomicLong(0)
    private var liveModeEnabled = false
    private var speechIntermediateCallback: IntermediateCallback? = null
    var currentModelType: SpeechModelType = DEFAULT_MODEL_TYPE
    var diarizationEnabled: Boolean = false

    val isMicRecording: Boolean
        get() = micRecording.get()

    /** 認識言語("ja"/"en"など)。"auto"の場合は自動判定(setLanguageを呼ばない) */
    var language: String = "ja"

    /** transcribeを1回実行するたびに所要時間(ms)を通知する。バックグラウンドスレッドで呼ばれる。 */
    @Volatile
    var transcribeTimeListener: ((Long) -> Unit)? = null

    /**
     * Downloads model files for the specified (or current) speech model type.
     * Always downloads Silero VAD model for all modes.
     * If diarizationEnabled is true, also downloads pyannote-audio segmentation and embedding models.
     */
    fun downloadModel(modelType: SpeechModelType = currentModelType, listener: ModelDownloadListener? = null): Boolean {
        currentModelType = modelType
        return try {
            Log.i(TAG, "Starting speech model download/check for ${modelType.displayName}...")
            check(ModelDownloader.downloadFile(
                modelDirectory,
                ModelFileSpec(modelType.encoderUrl, modelType.encoderFileName),
                listener,
            ) != null)
            if (modelType.needsDecoder) {
                check(ModelDownloader.downloadFile(
                    modelDirectory,
                    ModelFileSpec(modelType.decoderUrl, modelType.decoderFileName),
                    listener,
                ) != null)
            }
            for (extraFile in modelType.extraFiles) {
                check(ModelDownloader.downloadFile(modelDirectory, extraFile, listener) != null)
            }
            // Always download VAD model
            Log.i(TAG, "Downloading VAD model...")
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(VAD_URL, VAD_FILE), listener) != null)
            if (diarizationEnabled) {
                Log.i(TAG, "Downloading diarization models...")
                check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(DIARIZATION_SEGMENTATION_URL, DIARIZATION_SEGMENTATION_FILE), listener) != null)
                check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(DIARIZATION_EMBEDDING_URL, DIARIZATION_EMBEDDING_FILE), listener) != null)
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
            val encoderPath = File(modelDirectory, currentModelType.encoderFileName).absolutePath
            val decoderPath = if (currentModelType.needsDecoder) {
                File(modelDirectory, currentModelType.decoderFileName).absolutePath
            } else {
                ""
            }

            // When diarization is enabled, disable LIVE flag (diarization requires non-live mode)
            // Streaming still works without the LIVE flag
            val useLiveFlag = liveMode && !diarizationEnabled
            val flags = if (useLiveFlag) AiliaSpeech.AILIA_SPEECH_FLAG_LIVE else AiliaSpeech.AILIA_SPEECH_FLAG_NONE
            liveModeEnabled = useLiveFlag

            Log.i(TAG, "Initializing speech with envId=$envId, model=${currentModelType.displayName}, liveMode=$liveMode, diarization=$diarizationEnabled, useLiveFlag=$useLiveFlag")
            Log.i(TAG, "Encoder: $encoderPath")
            Log.i(TAG, "Decoder: $decoderPath")

            val qnnSelected = isQnnEnvironment(envId)
            val isSenseVoice =
                currentModelType.modelTypeId == AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL
            // QNN選択時、WhisperはdecoderがQNNで動作しないためencoderのみQNNで実行し、
            // decoderとVADはCPU側で実行する。SenseVoiceは全体をQNNで実行する。
            val engineEnvId = if (qnnSelected && !isSenseVoice) cpuEnvironmentId() else envId

            val engine = AiliaSpeech(
                envId = engineEnvId,
                task = AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE,
                flags = flags
            )
            speech = engine
            if (qnnSelected) {
                if (isSenseVoice) {
                    // QNNは固定シェイプが必要なため入力長を10秒に固定する
                    Log.i(TAG, "QNN selected: setStaticInputLength(${QNN_STATIC_INPUT_LENGTH_SEC}) for SenseVoice")
                    requireSuccess(
                        "setStaticInputLength",
                        engine.setStaticInputLength(QNN_STATIC_INPUT_LENGTH_SEC)
                    )
                } else {
                    Log.i(TAG, "QNN selected: encoder on envId=$envId, others on envId=$engineEnvId")
                    requireSuccess(
                        "setEnvId(encoder)",
                        engine.setEnvId(AiliaSpeech.AILIA_SPEECH_MODEL_TARGET_ENCODER, envId)
                    )
                }
            }
            requireSuccess("openModel", engine.openModel(encoderPath, decoderPath, currentModelType.modelTypeId))

            if (qnnSelected) {
                // QNNは初回推論時にグラフを構築するため、事前にウォームアップして初回transcribeの遅延を抑える
                val warmupStartNanos = System.nanoTime()
                requireSuccess("warmup", engine.warmup())
                Log.i(TAG, "QNN warmup finished in ${(System.nanoTime() - warmupStartNanos) / 1_000_000} ms")
            }

            // 言語設定("auto"は自動判定のためsetLanguageを呼ばない。Flutter版と同じ挙動)
            if (language != "auto") {
                val langResult = engine.setLanguage(language)
                Log.i(TAG, "setLanguage($language) result=$langResult")
                requireSuccess("setLanguage", langResult)
            }

            // Always open VAD (Silero VAD)
            val vadPath = File(modelDirectory, VAD_FILE).absolutePath
            Log.i(TAG, "Opening VAD: $vadPath")
            val vadResult = engine.openVad(vadPath, AiliaSpeech.AILIA_SPEECH_VAD_TYPE_SILERO)
            Log.i(TAG, "VAD openVad result=$vadResult")
            requireSuccess("openVad", vadResult)

            // VADを有効化するにはsetSilentThresholdの設定が必要
            // (ailia-models-flutterと同じ閾値: threshold=0.5, speechSec=1.0, noSpeechSec=1.0)
            val thresholdResult = engine.setSilentThreshold(0.5f, 1.0f, 1.0f)
            Log.i(TAG, "VAD setSilentThreshold result=$thresholdResult")
            requireSuccess("setSilentThreshold", thresholdResult)

            // Open diarization if enabled
            if (diarizationEnabled) {
                val segmentationPath = File(modelDirectory, DIARIZATION_SEGMENTATION_FILE).absolutePath
                val embeddingPath = File(modelDirectory, DIARIZATION_EMBEDDING_FILE).absolutePath
                Log.i(TAG, "Opening diarization: segmentation=$segmentationPath, embedding=$embeddingPath")
                val diarResult = engine.openDiarization(
                    segmentationPath, embeddingPath,
                    AiliaSpeech.AILIA_SPEECH_DIARIZATION_TYPE_PYANNOTE_AUDIO
                )
                Log.i(TAG, "Diarization openDiarization result=$diarResult")
                requireSuccess("openDiarization", diarResult)
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

    /** 指定envIdがQNN環境かどうかを環境名から判定する。 */
    private fun isQnnEnvironment(envId: Int): Boolean {
        if (envId < 0) return false
        return try {
            AiliaModel.getEnvironments().firstOrNull { it.id == envId }
                ?.name?.contains("QNN", ignoreCase = true) == true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to query ailia environments: ${e.message}")
            false
        }
    }

    /** BLASを優先し、なければ通常CPUのenvIdを返す。取得できない場合は自動選択(-1)。 */
    private fun cpuEnvironmentId(): Int {
        return try {
            val environments = AiliaModel.getEnvironments()
            val cpuEnv = environments.firstOrNull { it.type == AiliaEnvironment.TYPE_BLAS }
                ?: environments.firstOrNull { it.type == AiliaEnvironment.TYPE_CPU }
            cpuEnv?.id ?: -1
        } catch (e: Exception) {
            Log.e(TAG, "Failed to query ailia environments: ${e.message}")
            -1
        }
    }

    /**
     * Processes audio from a WAV file (non-live mode).
     * Calls pushInputData, finalizeInputData, transcribe, and returns transcript lines.
     */
    fun process(audio: FloatArray, channels: Int, sampleRate: Int): List<String> {
        return synchronized(speechLock) {
            val engine = speech ?: error("Speech model not initialized")
            Log.i(TAG, "Speech process: audio.size=${audio.size}, channels=$channels, sampleRate=$sampleRate, samples=${audio.size / channels}")
            requireSuccess("pushInputData", engine.pushInputData(audio, channels, audio.size / channels, sampleRate))
            requireSuccess("finalizeInputData", engine.finalizeInputData())
            val lines = mutableListOf<String>()
            while (engine.getComplete() == 0) {
                transcribeOnce(engine)
                lines.addAll(collectTextLines(engine))
            }
            requireSuccess("resetTranscribeState", engine.resetTranscribeState())
            lines
        }
    }

    /** transcribeを1回実行し、その所要時間を[transcribeTimeListener]へ通知する。 */
    private fun transcribeOnce(engine: AiliaSpeech) {
        val startNanos = System.nanoTime()
        requireSuccess("transcribe", engine.transcribe())
        transcribeTimeListener?.invoke((System.nanoTime() - startNanos) / 1_000_000)
    }

    /**
     * Pushes live audio data for streaming recognition (live mode).
     * Does NOT call finalizeInputData - use finalizeLiveAudio() when recording stops.
     * Returns transcript lines confirmed by this call.
     */
    fun pushLiveAudio(audio: FloatArray, channels: Int, sampleRate: Int): List<String> {
        return synchronized(speechLock) {
            val engine = speech ?: return@synchronized emptyList()
            requireSuccess("pushInputData", engine.pushInputData(audio, channels, audio.size / channels, sampleRate))
            val lines = mutableListOf<String>()
            while (engine.getBuffered() != 0) {
                transcribeOnce(engine)
                lines.addAll(collectTextLines(engine))
            }
            lines
        }
    }

    /**
     * Finalizes live audio input and returns the remaining transcript lines.
     * Call this when mic recording stops.
     */
    fun finalizeLiveAudio(): List<String> {
        return synchronized(speechLock) {
            val engine = speech ?: return@synchronized emptyList()
            requireSuccess("finalizeInputData", engine.finalizeInputData())
            val lines = mutableListOf<String>()
            while (engine.getComplete() == 0) {
                transcribeOnce(engine)
                lines.addAll(collectTextLines(engine))
            }
            requireSuccess("resetTranscribeState", engine.resetTranscribeState())
            lines
        }
    }

    /**
     * 16kHz/monoでマイク録音を開始する。API 23以降はPCM_FLOAT、旧端末はPCM16を使用する。
     * Flutter版と同様に短いチャンクを逐次投入し、SDKが処理可能になった時だけ認識する。
     */
    @SuppressLint("MissingPermission")
    fun startMicRecording(listener: MicRecordingListener): Boolean {
        if (!isInitialized) {
            listener.onError("Speech model not ready")
            return false
        }
        if (!micSessionActive.compareAndSet(false, true)) {
            listener.onError("Previous microphone session is still finalizing")
            return false
        }

        if (liveModeEnabled) {
            val callback = object : IntermediateCallback {
                override fun onIntermediateResult(text: String): Int {
                    listener.onIntermediateResult(text)
                    return 0
                }
            }
            val callbackResult = synchronized(speechLock) { speech?.setIntermediateCallback(callback) }
            Log.i(TAG, "Speech setIntermediateCallback result=$callbackResult")
            if (callbackResult != 0) {
                speechIntermediateCallback = null
                micSessionActive.set(false)
                listener.onError("Failed to register intermediate result callback: $callbackResult")
                return false
            }
            // Keep a strong reference while the native speech instance owns the callback.
            speechIntermediateCallback = callback
        } else {
            speechIntermediateCallback = null
        }

        val sampleRate = 16000
        val readChunkSize = sampleRate / 10
        val recognitionChunkSize = readChunkSize
        val useFloatInput = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
        val encoding = if (useFloatInput) AudioFormat.ENCODING_PCM_FLOAT else AudioFormat.ENCODING_PCM_16BIT
        val bytesPerSample = if (useFloatInput) Float.SIZE_BYTES else Short.SIZE_BYTES
        val audioRecordBufferBytes = sampleRate * bytesPerSample * 2

        return try {
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                encoding,
                audioRecordBufferBytes
            )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                micSessionActive.set(false)
                listener.onError("Failed to initialize AudioRecord")
                return false
            }

            synchronized(this) {
                audioRecord = recorder
            }
            finalizeMicInput.set(true)
            micProcessingTimeNanos.set(0)
            micRecognitionExecutor = newMicExecutor("ailia-speech-recognition", android.os.Process.THREAD_PRIORITY_BACKGROUND)
            micReadExecutor = newMicExecutor("ailia-speech-recording", android.os.Process.THREAD_PRIORITY_AUDIO)
            recorder.startRecording()
            micRecording.set(true)
            micReadExecutor?.execute {
                runMicRecordingLoop(
                    recorder,
                    listener,
                    sampleRate,
                    readChunkSize,
                    recognitionChunkSize,
                    useFloatInput,
                )
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start microphone recording", e)
            releaseAudioRecord()
            shutdownMicExecutors()
            micSessionActive.set(false)
            listener.onError(e.message ?: "Failed to start microphone recording")
            false
        }
    }

    private fun newMicExecutor(name: String, priority: Int): ExecutorService {
        return Executors.newSingleThreadExecutor { runnable ->
            Thread({
                android.os.Process.setThreadPriority(priority)
                runnable.run()
            }, name)
        }
    }

    private fun runMicRecordingLoop(
        recorder: AudioRecord,
        listener: MicRecordingListener,
        sampleRate: Int,
        readChunkSize: Int,
        recognitionChunkSize: Int,
        useFloatInput: Boolean,
    ) {
        val readBuffer = FloatArray(readChunkSize)
        val shortReadBuffer = if (useFloatInput) null else ShortArray(readChunkSize)
        val recognitionBuffer = FloatArray(recognitionChunkSize)
        var recognitionFill = 0

        try {
            while (micRecording.get()) {
                val readResult = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && useFloatInput) {
                    recorder.read(readBuffer, 0, readBuffer.size, AudioRecord.READ_BLOCKING)
                } else {
                    @Suppress("DEPRECATION")
                    val count = recorder.read(shortReadBuffer!!, 0, shortReadBuffer.size)
                    if (count > 0) {
                        for (i in 0 until count) readBuffer[i] = shortReadBuffer[i] / 32768.0f
                    }
                    count
                }
                if (readResult < 0) {
                    micRecording.set(false)
                    listener.onError("AudioRecord read failed: $readResult")
                    break
                }
                if (readResult == 0) {
                    continue
                }

                listener.onWaveform(readBuffer.copyOf(readResult), sampleRate)

                var sourceOffset = 0
                while (sourceOffset < readResult) {
                    val copySize = minOf(
                        readResult - sourceOffset,
                        recognitionChunkSize - recognitionFill
                    )
                    System.arraycopy(
                        readBuffer,
                        sourceOffset,
                        recognitionBuffer,
                        recognitionFill,
                        copySize
                    )
                    sourceOffset += copySize
                    recognitionFill += copySize

                    if (recognitionFill == recognitionChunkSize) {
                        submitMicChunk(recognitionBuffer.copyOf(), listener, sampleRate)
                        recognitionFill = 0
                    }
                }
            }
        } catch (e: Exception) {
            if (micRecording.get()) {
                Log.e(TAG, "Microphone recording loop failed", e)
                listener.onError(e.message ?: "Microphone recording failed")
            }
        } finally {
            micRecording.set(false)
            releaseAudioRecord(recorder)
            micReadExecutor?.shutdown()
            micReadExecutor = null

            if (finalizeMicInput.get()) {
                submitMicFinalization(
                    recognitionBuffer.copyOf(recognitionFill),
                    listener,
                    sampleRate
                )
            } else {
                micRecognitionExecutor?.shutdownNow()
                micRecognitionExecutor = null
                micSessionActive.set(false)
            }
        }
    }

    private fun submitMicChunk(
        chunk: FloatArray,
        listener: MicRecordingListener,
        sampleRate: Int
    ) {
        micRecognitionExecutor?.execute {
            if (!isInitialized) return@execute
            try {
                val startTime = System.nanoTime()
                val lines = pushLiveAudio(chunk, 1, sampleRate)
                val processingTimeMs = micProcessingTimeNanos.addAndGet(
                    System.nanoTime() - startTime
                ) / 1_000_000
                if (lines.isNotEmpty()) {
                    listener.onResult(lines, false, processingTimeMs)
                }
            } catch (e: Exception) {
                Log.e(TAG, "pushLiveAudio failed", e)
                listener.onError(e.message ?: "Live speech recognition failed")
            }
        }
    }

    private fun submitMicFinalization(
        tail: FloatArray,
        listener: MicRecordingListener,
        sampleRate: Int
    ) {
        val executor = micRecognitionExecutor
        if (executor == null) {
            micSessionActive.set(false)
            return
        }
        executor.execute {
            try {
                if (isInitialized && tail.isNotEmpty()) {
                    val startTime = System.nanoTime()
                    val lines = pushLiveAudio(tail, 1, sampleRate)
                    val processingTimeMs = micProcessingTimeNanos.addAndGet(
                        System.nanoTime() - startTime
                    ) / 1_000_000
                    if (lines.isNotEmpty()) {
                        listener.onResult(lines, false, processingTimeMs)
                    }
                }
                val finalLines = if (isInitialized) {
                    val startTime = System.nanoTime()
                    val lines = finalizeLiveAudio()
                    micProcessingTimeNanos.addAndGet(System.nanoTime() - startTime)
                    lines
                } else {
                    emptyList()
                }
                listener.onResult(
                    finalLines,
                    true,
                    micProcessingTimeNanos.get() / 1_000_000,
                )
            } catch (e: Exception) {
                Log.e(TAG, "finalizeLiveAudio failed", e)
                listener.onError(e.message ?: "Failed to finalize live speech recognition")
            } finally {
                micSessionActive.set(false)
                synchronized(this) {
                    if (micRecognitionExecutor === executor) {
                        micRecognitionExecutor = null
                    }
                }
            }
        }
        executor.shutdown()
    }

    /** 録音を停止する。通常停止時は1秒未満の末尾も投入して認識を確定する。 */
    fun stopMicRecording(finalize: Boolean = true) {
        if (!micSessionActive.get()) return
        finalizeMicInput.set(finalize)
        micRecording.set(false)
        releaseAudioRecord()
    }

    private fun releaseAudioRecord(expected: AudioRecord? = null) {
        val recorder = synchronized(this) {
            if (expected != null && audioRecord !== expected) {
                null
            } else {
                val current = audioRecord
                audioRecord = null
                current
            }
        } ?: return
        try {
            if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                recorder.stop()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop AudioRecord: ${e.message}")
        }
        try {
            recorder.release()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release AudioRecord: ${e.message}")
        }
    }

    private fun shutdownMicExecutors() {
        micReadExecutor?.shutdownNow()
        micReadExecutor = null
        micRecognitionExecutor?.shutdownNow()
        micRecognitionExecutor = null
    }

    private fun formatTimeStamp(sec: Float): String {
        val total = sec.toInt()
        return String.format(Locale.ROOT, "%02d:%02d", total / 60, total % 60)
    }

    /**
     * Collects transcript lines from the speech engine in the
     * meeting-minutes format used by ailia-models-flutter:
     * "[mm:ss - mm:ss] text". When diarization is enabled, each line
     * is prefixed with the speaker ID.
     */
    private fun collectTextLines(engine: AiliaSpeech): List<String> {
        val count = engine.getTextCount()
        Log.i(TAG, "Speech getTextCount=$count")
        if (count == 0) {
            return emptyList()
        }
        val lines = mutableListOf<String>()
        for (i in 0 until count) {
            val text: AiliaSpeechText? = engine.getText(i)
            if (text == null) {
                continue
            }
            val stamp = "[${formatTimeStamp(text.timeStampBegin)} - ${formatTimeStamp(text.timeStampEnd)}]"
            if (diarizationEnabled && text.speakerId.toLong() and 0xFFFFFFFFL != AiliaSpeech.AILIA_SPEECH_SPEAKER_ID_UNKNOWN.toLong() and 0xFFFFFFFFL) {
                Log.i(TAG, "Speech text[$i]: speaker=#${text.speakerId} '${text.text}' confidence=${text.confidence}")
                lines.add("$stamp [Speaker ${text.speakerId}] ${text.text}")
            } else {
                Log.i(TAG, "Speech text[$i]: '${text.text}' confidence=${text.confidence}")
                lines.add("$stamp ${text.text}")
            }
        }
        Log.i(TAG, "Speech result lines: $lines")
        return lines
    }

    fun releaseSpeech() {
        isInitialized = false
        stopMicRecording(finalize = false)
        shutdownMicExecutors()
        micSessionActive.set(false)
        synchronized(speechLock) {
            try {
                speech?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing speech: ${e.javaClass.name}: ${e.message}")
            } finally {
                speech = null
                speechIntermediateCallback = null
                liveModeEnabled = false
                isInitialized = false
                Log.i(TAG, "Speech released")
            }
        }
    }

    private fun requireSuccess(operation: String, status: Int) {
        if (status != 0) {
            val detail = speech?.getErrorDetail().orEmpty()
            throw IllegalStateException("$operation failed: status=$status $detail".trim())
        }
    }
}
