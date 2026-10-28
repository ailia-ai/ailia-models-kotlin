package jp.axinc.ailia_kotlin

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import java.io.File
import java.util.Locale
import java.util.concurrent.CancellationException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

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
    }

    private var speech: AiliaSpeech? = null
    private val speechLock = Any()
    @Volatile
    private var isInitialized = false
    private var audioRecord: AudioRecord? = null
    @Volatile private var micReadExecutor: ExecutorService? = null
    @Volatile private var micRecognitionExecutor: ExecutorService? = null
    private val micRecording = AtomicBoolean(false)
    private val micSessionActive = AtomicBoolean(false)
    private val finalizeMicInput = AtomicBoolean(true)
    private val micProcessingTimeNanos = AtomicLong(0)
    private var liveModeEnabled = false
    private var speechIntermediateCallback: IntermediateCallback? = null
    var currentModelType: SpeechModelType = DEFAULT_MODEL_TYPE
    var diarizationEnabled: Boolean = false

    /** One drain task per session; audio captured during recognition becomes the next batch. */
    private class MicAudioBuffer {
        private val chunks = ArrayDeque<FloatArray>()
        private var draining = false

        @Synchronized
        fun add(chunk: FloatArray): Boolean {
            chunks.addLast(chunk)
            if (draining) return false
            draining = true
            return true
        }

        @Synchronized
        fun take(): FloatArray? {
            if (chunks.isEmpty()) {
                draining = false
                return null
            }
            if (chunks.size == 1) return chunks.removeFirst()
            val batch = FloatArray(chunks.sumOf { it.size })
            var offset = 0
            while (chunks.isNotEmpty()) {
                val chunk = chunks.removeFirst()
                chunk.copyInto(batch, offset)
                offset += chunk.size
            }
            return batch
        }
    }

    /**
     * trueの場合、SoC固有の変換済みQNNモデル(.qnn)をNPUで実行する。
     * .qnnはencoder/decoderの両方を含むため、openModelに渡すだけでQNNで動作する。
     * VADと話者分離はinitializeSpeechに渡したenvIdで実行する。
     */
    var useQnn: Boolean = false

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
            val qnnFileName = qnnFileNameOrNull()
            if (qnnFileName != null) {
                check(ModelDownloader.downloadFile(
                    modelDirectory,
                    ModelDownloader.qnnSpeechModelSpec(qnnFileName),
                    listener,
                ) != null)
            } else {
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
            val qnnFileName = qnnFileNameOrNull()
            if (useQnn && qnnFileName == null) {
                throw IllegalStateException("QNN model is not available for ${currentModelType.displayName} on ${QnnSupport.socName}")
            }
            // .qnnはencoder/decoderの両方を含むため、両方に同じファイルを渡す
            val encoderPath = File(modelDirectory, qnnFileName ?: currentModelType.encoderFileName).absolutePath
            val decoderPath = when {
                qnnFileName != null -> encoderPath
                currentModelType.needsDecoder -> File(modelDirectory, currentModelType.decoderFileName).absolutePath
                else -> ""
            }

            val flags = if (liveMode) AiliaSpeech.AILIA_SPEECH_FLAG_LIVE else AiliaSpeech.AILIA_SPEECH_FLAG_NONE
            liveModeEnabled = liveMode

            Log.i(TAG, "Initializing speech with envId=$envId, model=${currentModelType.displayName}, qnn=${qnnFileName != null}, liveMode=$liveMode, diarization=$diarizationEnabled")
            Log.i(TAG, "Encoder: $encoderPath")
            Log.i(TAG, "Decoder: $decoderPath")

            val engine = AiliaSpeech(
                envId = envId,
                task = AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE,
                flags = flags
            )
            speech = engine
            requireSuccess("openModel", engine.openModel(encoderPath, decoderPath, currentModelType.modelTypeId))

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

    /** QNNを選択している場合のQNNモデルファイル名。CPU時や未対応の組み合わせではnull。 */
    private fun qnnFileNameOrNull(): String? =
        if (useQnn) QnnSupport.speechQnnFileName(currentModelType) else null

    /**
     * Processes audio from a WAV file (non-live mode).
     * Calls pushInputData, finalizeInputData, transcribe, and returns transcript lines.
     */
    fun process(audio: FloatArray, channels: Int, sampleRate: Int,
                onText: (List<String>) -> Unit = {}): List<String> {
        return synchronized(speechLock) {
            val engine = speech ?: error("Speech model not initialized")
            Log.i(TAG, "Speech process: audio.size=${audio.size}, channels=$channels, sampleRate=$sampleRate, samples=${audio.size / channels}")
            requireSuccess("pushInputData", engine.pushInputData(audio, channels, audio.size / channels, sampleRate))
            requireSuccess("finalizeInputData", engine.finalizeInputData())
            val lines = mutableListOf<String>()
            while (engine.getComplete() == 0) {
                transcribeOnce(engine)
                val confirmed = collectTextLines(engine)
                lines.addAll(confirmed)
                if (confirmed.isNotEmpty()) onText(confirmed)
            }
            requireSuccess("resetTranscribeState", engine.resetTranscribeState())
            lines
        }
    }

    /** transcribeを1回実行し、その所要時間を[transcribeTimeListener]へ通知する。 */
    private fun transcribeOnce(engine: AiliaSpeech, checkActive: () -> Unit = {}) {
        checkActive()
        val startNanos = System.nanoTime()
        val status = engine.transcribe()
        checkActive()
        requireSuccess("transcribe", status)
        transcribeTimeListener?.invoke((System.nanoTime() - startNanos) / 1_000_000)
    }

    /** Non-LIVE input: Buffered is raised by Silero endpointing or the SDK's 30-second limit. */
    internal fun createBufferedTurn(onIntermediate: ((String) -> Unit)? = null,
                                   checkActive: () -> Unit = {}): VoiceAgentSpeechTurn {
        val engine = checkNotNull(speech) { "Speech model not initialized" }
        check(!liveModeEnabled) { "Buffered endpoint detection requires non-LIVE mode" }
        // Short Japanese replies should not require the general sample's one-second speech start.
        requireSuccess("setSilentThreshold", engine.setSilentThreshold(0.5f, 0.25f, 1.0f))
        Log.i(TAG, "Speech endpoint: Silero VAD, threshold=0.5, speechSec=0.25, noSpeechSec=1.0, non-LIVE")
        val confirmed = mutableListOf<String>()
        if (onIntermediate != null) {
            val callback = object : IntermediateCallback {
                override fun onIntermediateResult(text: String): Int = try {
                    checkActive()
                    onIntermediate(voiceAgentTranscript(confirmed + text))
                    0
                } catch (_: CancellationException) {
                    // Abort decoding without throwing through the native callback boundary.
                    1
                }
            }
            requireSuccess("setIntermediateCallback", engine.setIntermediateCallback(callback))
            speechIntermediateCallback = callback
        }
        fun flag(operation: String, value: Int): Boolean {
            check(value in 0..1) { "$operation failed ($value): ${engine.getErrorDetail()}" }
            return value == 1
        }
        return VoiceAgentSpeechTurn(object : VoiceAgentSpeechTurn.Engine {
            override fun pushInputData(audio: FloatArray, channels: Int, sampleRate: Int) {
                requireSuccess("pushInputData", engine.pushInputData(audio, channels, audio.size / channels, sampleRate))
            }
            override fun buffered() = flag("buffered", engine.getBuffered())
            override fun finalizeInputData() {
                requireSuccess("finalizeInputData", engine.finalizeInputData())
                Log.i(TAG, "Speech endpoint: finalizeInputData")
            }
            override fun transcribe(): List<String> {
                transcribeOnce(engine)
                val lines = collectTextLines(engine)
                confirmed += lines
                checkActive()
                onIntermediate?.invoke(voiceAgentTranscript(confirmed))
                return lines
            }
            override fun complete() = flag("complete", engine.getComplete())
            override fun resetTranscribeState() {
                requireSuccess("resetTranscribeState", engine.resetTranscribeState())
                confirmed.clear()
                Log.i(TAG, "Speech endpoint: resetTranscribeState")
            }
        })
    }

    /**
     * Pushes live audio data for streaming recognition (live mode).
     * Does NOT call finalizeInputData - use finalizeLiveAudio() when recording stops.
     * Returns transcript lines confirmed by this call.
     */
    fun pushLiveAudio(audio: FloatArray, channels: Int, sampleRate: Int,
                      onText: (List<String>) -> Unit = {}): List<String> =
        pushLiveAudio(audio, channels, sampleRate, {}, onText)

    private fun pushLiveAudio(audio: FloatArray, channels: Int, sampleRate: Int,
                              checkActive: () -> Unit, onText: (List<String>) -> Unit): List<String> {
        return synchronized(speechLock) {
            checkActive()
            val engine = speech ?: return@synchronized emptyList()
            requireSuccess("pushInputData", engine.pushInputData(audio, channels, audio.size / channels, sampleRate))
            val lines = mutableListOf<String>()
            while (engine.getBuffered() != 0) {
                transcribeOnce(engine, checkActive)
                val confirmed = collectTextLines(engine)
                lines.addAll(confirmed)
                checkActive()
                if (confirmed.isNotEmpty()) onText(confirmed)
            }
            lines
        }
    }

    /**
     * Finalizes live audio input and returns the remaining transcript lines.
     * Call this when mic recording stops.
     */
    fun finalizeLiveAudio(onText: (List<String>) -> Unit = {}): List<String> =
        finalizeLiveAudio({}, onText)

    private fun finalizeLiveAudio(checkActive: () -> Unit, onText: (List<String>) -> Unit): List<String> {
        return synchronized(speechLock) {
            checkActive()
            val engine = speech ?: return@synchronized emptyList()
            requireSuccess("finalizeInputData", engine.finalizeInputData())
            val lines = mutableListOf<String>()
            while (engine.getComplete() == 0) {
                transcribeOnce(engine, checkActive)
                val confirmed = collectTextLines(engine)
                lines.addAll(confirmed)
                checkActive()
                if (confirmed.isNotEmpty()) onText(confirmed)
            }
            requireSuccess("resetTranscribeState", engine.resetTranscribeState())
            lines
        }
    }

    /**
     * 16kHz/monoでマイク録音を開始する。API 23以降はPCM_FLOAT、旧端末はPCM16を使用する。
     * 録音は100ms単位で続け、認識中に溜まった未処理音声はまとめてSDKへ投入する。
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

        // Register in both modes so an option change can abort an in-flight native decode.
        val callback = object : IntermediateCallback {
            override fun onIntermediateResult(text: String): Int {
                if (!isInitialized || !finalizeMicInput.get()) return 1
                if (liveModeEnabled) listener.onIntermediateResult(text)
                return if (isInitialized && finalizeMicInput.get()) 0 else 1
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
        speechIntermediateCallback = callback

        val sampleRate = 16000
        val readChunkSize = sampleRate / 10
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
        useFloatInput: Boolean,
    ) {
        val readBuffer = FloatArray(readChunkSize)
        val shortReadBuffer = if (useFloatInput) null else ShortArray(readChunkSize)
        val pendingAudio = MicAudioBuffer()

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

                submitMicChunk(readBuffer.copyOf(readResult), pendingAudio, listener, sampleRate)
            }
        } catch (e: Exception) {
            if (micRecording.get()) {
                Log.e(TAG, "Microphone recording loop failed", e)
                listener.onError(e.message ?: "Microphone recording failed")
            }
        } finally {
            micRecording.set(false)
            releaseAudioRecord(recorder)
            // Cleanup is queued after recognition even when cancelled; never interrupt JNI.
            submitMicFinalization(listener)
            micReadExecutor?.shutdown()
            micReadExecutor = null
        }
    }

    private fun submitMicChunk(
        chunk: FloatArray,
        pendingAudio: MicAudioBuffer,
        listener: MicRecordingListener,
        sampleRate: Int
    ) {
        val executor = micRecognitionExecutor ?: return
        if (!pendingAudio.add(chunk)) return
        executor.execute {
            while (isInitialized && finalizeMicInput.get()) {
                val batch = pendingAudio.take() ?: break
                try {
                    val startTime = System.nanoTime()
                    pushLiveAudio(batch, 1, sampleRate, ::checkMicProcessingActive) { lines ->
                        notifyMicText(listener, lines, startTime)
                    }
                    micProcessingTimeNanos.addAndGet(System.nanoTime() - startTime)
                } catch (_: CancellationException) {
                    break
                } catch (e: Exception) {
                    Log.e(TAG, "pushLiveAudio failed", e)
                    listener.onError(e.message ?: "Live speech recognition failed")
                }
            }
        }
    }

    private fun submitMicFinalization(
        listener: MicRecordingListener,
    ) {
        val executor = micRecognitionExecutor
        if (executor == null) {
            micSessionActive.set(false)
            return
        }
        executor.execute {
            var completed = false
            try {
                checkMicProcessingActive()
                val startTime = System.nanoTime()
                finalizeLiveAudio(::checkMicProcessingActive) { lines ->
                    notifyMicText(listener, lines, startTime)
                }
                micProcessingTimeNanos.addAndGet(System.nanoTime() - startTime)
                completed = true
            } catch (_: CancellationException) {
                // Option changes discard this session; the model owner will release it.
            } catch (e: Exception) {
                Log.e(TAG, "finalizeLiveAudio failed", e)
                listener.onError(e.message ?: "Failed to finalize live speech recognition")
            } finally {
                synchronized(this) {
                    if (micRecognitionExecutor === executor) {
                        micRecognitionExecutor = null
                    }
                }
                micSessionActive.set(false)
            }
            if (completed && isInitialized && finalizeMicInput.get()) {
                // Text has already been published after each transcribe; only signal completion.
                listener.onResult(emptyList(), true, micProcessingTimeNanos.get() / 1_000_000)
            }
        }
        executor.shutdown()
    }

    private fun notifyMicText(listener: MicRecordingListener, lines: List<String>, startTime: Long) {
        checkMicProcessingActive()
        val processingTimeMs = (micProcessingTimeNanos.get() + System.nanoTime() - startTime) / 1_000_000
        listener.onResult(lines, false, processingTimeMs)
    }

    /** 録音を停止する。通常停止時は未処理音声をすべて投入してから認識を確定する。 */
    fun stopMicRecording(finalize: Boolean = true) {
        if (!micSessionActive.get()) return
        finalizeMicInput.set(finalize)
        micRecording.set(false)
        releaseAudioRecord()
    }

    private fun checkMicProcessingActive() {
        if (!isInitialized || !finalizeMicInput.get()) {
            throw CancellationException("Microphone recognition cancelled")
        }
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
        // Called on the model owner before close/reinitialization. Let the reader enqueue
        // its cleanup first, then join recognition without interrupting native inference.
        var interrupted = false
        fun finish(executor: ExecutorService?) {
            executor ?: return
            executor.shutdown()
            while (true) {
                try {
                    if (executor.awaitTermination(100, TimeUnit.MILLISECONDS)) return
                } catch (_: InterruptedException) {
                    interrupted = true
                }
            }
        }
        finish(micReadExecutor)
        micReadExecutor = null
        finish(micRecognitionExecutor)
        micRecognitionExecutor = null
        if (interrupted) Thread.currentThread().interrupt()
    }

    private fun formatTimeStamp(sec: Float): String {
        val total = sec.toInt()
        return String.format(Locale.ROOT, "%02d:%02d", total / 60, total % 60)
    }

    /**
     * Collects transcript lines from the speech engine in the
     * meeting-minutes format used by ailia-models-flutter:
     * "[mm:ss - mm:ss] text". When diarization is enabled, each line
     * is prefixed with a compact speaker label such as "S0:".
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
                lines.add("$stamp S${text.speakerId}: ${text.text}")
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
