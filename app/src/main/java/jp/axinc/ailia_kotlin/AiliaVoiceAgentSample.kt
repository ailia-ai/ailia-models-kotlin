package jp.axinc.ailia_kotlin

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Orchestrates microphone turns and UI callbacks for Whisper → Gemma → GPT-SoVITS.
 * VoiceAgentModels owns inference/history; VoiceAgentPlayer owns PCM output.
 * Speech/LLM creation, inference and destruction run on the supplied model executor.
 * The executor must be the same one used for the application's other LLMs (QNN thread affinity).
 * Response synthesis owns a private voice model on a separate worker, joined before the next turn.
 * Microphone capture is stopped during recognition, generation and playback to avoid feedback.
 * Whisper is released before generation; Gemma and response synthesis can run concurrently.
 */
class AiliaVoiceAgentSample(context: Context, private val executor: ExecutorService) {
    enum class State {
        PREPARING, LISTENING, SYNTHESIZING_QUERY, WAITING_QUERY, PLAYING_QUERY,
        SUBMITTING, RECOGNIZING, THINKING, SYNTHESIZING, SPEAKING, CANCELLING, TEST_COMPLETE, STOPPING
    }
    enum class InputMode { MICROPHONE, SYNTHESIZED_QUERIES }
    data class Config(
        val speechEnvId: Int,
        val speechQnn: Boolean,
        val llmBackend: LLMBackend,
        val voiceEnvId: Int,
        val inputMode: InputMode = InputMode.MICROPHONE,
        val testQueries: List<String> = emptyList(),
        val speechOutputEnabled: Boolean = true,
        val speechModelType: SpeechModelType = SpeechModelType.WHISPER_MEDIUM,
        val llmModelType: LLMModelType = LLMModelType.GEMMA_4_E2B,
        val mtpEnabled: Boolean = llmModelType == LLMModelType.GEMMA_4_E2B,
    ) {
        init {
            require(speechModelType in setOf(SpeechModelType.WHISPER_MEDIUM, SpeechModelType.WHISPER_LARGE_V3,
                SpeechModelType.WHISPER_LARGE_V3_TURBO)) { "Whisper Medium、Large V3またはTurboを選択してください" }
            require(llmModelType in LLMModelType.GEMMA_4_MODELS) { "Gemma 4を選択してください" }
            require(inputMode != InputMode.SYNTHESIZED_QUERIES ||
                (testQueries.isNotEmpty() && testQueries.all { it.isNotBlank() })) {
                "テストクエリを1行に1件ずつ入力してください"
            }
        }
    }
    interface Listener {
        fun onState(state: State)
        fun onInput(text: String)
        fun onOutput(text: String)
        /** Provisional text; only onInput/onOutput carry the completed turn. */
        fun onIntermediateInput(text: String) = Unit
        fun onOutputPreview(text: String) = Unit
        fun onWaveform(samples: FloatArray)
        fun onDownload(fileName: String, downloaded: Long, total: Long)
        fun onStopped(error: String?)
        fun onAirConditionerTemperature(temperature: Double) = Unit
        fun onTakePhoto(onComplete: (Result<File>) -> Unit) {
            onComplete(Result.failure(IllegalStateException("カメラが接続されていません")))
        }
        fun onPhoto(file: File) { file.delete() }
    }

    private val context = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val generation = AtomicLong()
    private val models = VoiceAgentModels(this.context)
    private val player = VoiceAgentPlayer()
    @Volatile private var lastPlaybackTail: AudioUtil.WavFileData? = null
    private var listener: Listener? = null
    private var currentState = State.STOPPING
    @Volatile private var inputMode = InputMode.MICROPHONE
    private var speechOutputEnabled = true
    // Accessed on the UI thread, except for token-gated recorder callbacks.
    private class MicInput(val recorder: WavRecorder) {
        val hasSamples = AtomicBoolean(false)
        val manualSubmit = AtomicBoolean(false)
        val submitted = AtomicBoolean(false)
    }
    @Volatile private var micInput: MicInput? = null
    var isActive = false
        private set
    private var stopping = false

    fun start(config: Config, listener: Listener) {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (isActive || stopping) return
        this.listener = listener
        lastPlaybackTail = null
        inputMode = config.inputMode
        speechOutputEnabled = config.speechOutputEnabled
        isActive = true
        val token = generation.incrementAndGet()
        notifyState(State.PREPARING)
        executor.execute {
            try {
                models.prepare(config, { ensureActive(token) }) { fileName, downloaded, total ->
                    emit(token) { listener?.onDownload(fileName, downloaded, total) }
                }
                when (config.inputMode) {
                    InputMode.MICROPHONE -> emit(token) { listen(token) }
                    InputMode.SYNTHESIZED_QUERIES -> runTestQueries(token, config.testQueries)
                }
            } catch (_: CancellationException) {
                // stop() already queued resource release on this executor.
            } catch (e: Exception) {
                fail(token, e)
            }
        }
    }

    /** UI thread only. A fresh recorder prevents a cancelled capture from touching the next turn. */
    private fun listen(token: Long) {
        if (generation.get() != token) return
        notifyState(State.LISTENING)
        val gate = VoiceAgentInputGate(
            (player.remainingOutputSettleMs * WavRecorder.SAMPLE_RATE / 1000).toInt())
        val capture = WavRecorder()
        val turn = MicInput(capture)
        micInput = turn
        // Capture can start while a cancelled native call is still finishing. All SDK input
        // queues behind initialization on the owning worker, including manual submission.
        enqueue(token) { startListening(token) }
        val file = File(context.cacheDir, "voice_agent_$token.wav")
        capture.startRecording(file, object : WavRecorder.RecordingListener {
            override fun onWaveform(chunk: FloatArray, sampleRate: Int) {
                if (generation.get() != token || turn.submitted.get()) return
                val input = gate.accept(chunk)
                if (input.isEmpty()) return
                turn.hasSamples.set(true)
                emit(token) { listener?.onWaveform(input) }
                enqueue(token) {
                    if (!turn.submitted.get() && models.pushSpeech(input, 1, sampleRate)) {
                        Log.d("VoiceAgent", "Speech Buffered: finalize microphone turn")
                        finishMicrophoneTurn(token, turn)
                    }
                }
            }
            override fun onCompleted(file: File, audio: FloatArray, sampleRate: Int) {
                file.delete()
                if (generation.get() != token) return
                emit(token) { if (!turn.submitted.get()) notifyState(State.SUBMITTING) }
                // Also finalizes manual sends / the 30-second recording limit. The submitted
                // guard prevents late recorder callbacks from finalizing the same turn twice.
                enqueue(token) { finishMicrophoneTurn(token, turn) }
            }
            override fun onError(error: String) = fail(token, IllegalStateException(error))
        })
    }

    private fun finishMicrophoneTurn(token: Long, turn: MicInput) {
        if (!turn.submitted.compareAndSet(false, true)) return
        turn.recorder.stopRecording()
        state(token, State.SUBMITTING)
        respond(token)
        emit(token) { listen(token) }
    }

    private fun enqueue(token: Long, action: () -> Unit) {
        executor.execute {
            try {
                ensureActive(token)
                action()
            } catch (_: CancellationException) {
            } catch (error: Exception) {
                fail(token, error)
            }
        }
    }

    /** Finish the current recording immediately, including speech too short/quiet for endpointing. */
    fun submitMicrophoneInput(): Boolean {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (!isActive || inputMode != InputMode.MICROPHONE) return false
        val turn = micInput ?: return false
        if (!turn.hasSamples.get() || !turn.recorder.isRecording || !turn.manualSubmit.compareAndSet(false, true)) return false
        notifyState(State.SUBMITTING)
        turn.recorder.stopRecording()
        return true
    }

    /** Cancel this turn without ending the session. Native calls finish on their owning worker. */
    fun interruptResponse(resumeMicrophone: Boolean = true): Boolean {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (!isActive || currentState !in INTERRUPTIBLE_STATES) return false
        val discardInputPreview = currentState == State.RECOGNIZING
        val token = generation.incrementAndGet()
        inputMode = InputMode.MICROPHONE
        micInput?.recorder?.cancelRecording()
        micInput = null
        player.silence()
        notifyState(State.CANCELLING)
        if (discardInputPreview) listener?.onIntermediateInput("")
        listener?.onOutput("")
        // This runs after the cancelled turn's finally blocks and before any new inference.
        executor.execute { models.release(clearHistory = false) }
        if (resumeMicrophone) emit(token) { listen(token) }
        return true
    }

    /** Continue after microphone permission is granted, or after a completed automated test. */
    fun resumeMicrophoneInput() {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (isActive && currentState in setOf(State.CANCELLING, State.TEST_COMPLETE)) {
            inputMode = InputMode.MICROPHONE
            val token = generation.get()
            emit(token) { listen(token) }
        }
    }

    companion object {
        val INTERRUPTIBLE_STATES = setOf(State.SUBMITTING, State.RECOGNIZING, State.THINKING,
            State.SYNTHESIZING, State.SPEAKING, State.SYNTHESIZING_QUERY, State.WAITING_QUERY, State.PLAYING_QUERY)
    }

    /** Generated PCM is passed directly to Whisper; neither the microphone nor query text feeds the LLM. */
    private fun runTestQueries(token: Long, queries: List<String>) {
        for ((index, query) in queries.withIndex()) {
            val echo = lastPlaybackTail
            state(token, State.SYNTHESIZING_QUERY)
            val audio = models.synthesize(query) { ensureActive(token) }
            captureTestQuery(token, audio, echo)
            Log.d("VoiceAgent", "Test query ${index + 1}/${queries.size} submitted through VAD")
            state(token, State.SUBMITTING)
            respond(token)
        }
        // Keep the latest recognized query and answer visible until the user closes the session.
        state(token, State.TEST_COMPLETE)
    }

    /** Exercise the microphone endpoint with real synthesized PCM, echo, and inter-turn silence. */
    private fun captureTestQuery(token: Long, audio: AudioUtil.WavFileData,
                                 echo: AudioUtil.WavFileData?) {
        val valuesPerSecond = audio.sampleRate * audio.channels
        val guard = (valuesPerSecond * VoiceAgentPlayer.OUTPUT_SETTLE_MS / 1000).toInt()
        val gate = VoiceAgentInputGate(guard)
        val chunkSize = valuesPerSecond / 10
        state(token, State.WAITING_QUERY)
        startListening(token)
        fun feed(chunk: FloatArray): Boolean {
            ensureActive(token)
            val input = gate.accept(chunk)
            return input.isNotEmpty() && models.pushSpeech(input, audio.channels, audio.sampleRate)
        }
        // Silero sees silence after the playback tail is excluded, exactly as on microphone input.
        val tail = if (echo?.sampleRate == audio.sampleRate && echo.channels == audio.channels)
            echo.audioData.copyOf(guard) else FloatArray(guard)
        for (offset in tail.indices step chunkSize) {
            check(!feed(tail.copyOfRange(offset, minOf(offset + chunkSize, tail.size))))
            Thread.sleep(100)
        }
        repeat(12) {
            check(!feed(FloatArray(chunkSize))) { "再生残響でVADが作動しました" }
            Thread.sleep(100)
        }
        state(token, State.PLAYING_QUERY)
        play(token, audio)
        var buffered = false
        for (offset in audio.audioData.indices step chunkSize) {
            if (feed(audio.audioData.copyOfRange(offset, minOf(offset + chunkSize, audio.audioData.size)))) {
                buffered = true
                break
            }
        }
        repeat(20) { if (!buffered) buffered = feed(FloatArray(chunkSize)) }
        check(buffered) { "Silero VADがテストクエリの発話終了を検出できませんでした" }
    }

    private fun respond(token: Long) {
        models.withConversationTurn(isCancelled = { generation.get() != token }) {
            respondToAudio(token)
        }
    }

    private fun respondToAudio(token: Long) {
        state(token, State.RECOGNIZING)
        emit(token) { listener?.onIntermediateInput(""); listener?.onOutputPreview("") }
        val input = models.finishRecognition { ensureActive(token) }
        if (input.isBlank()) {
            check(inputMode != InputMode.SYNTHESIZED_QUERIES) { "テスト音声を認識できませんでした" }
            return
        }
        emit(token) { listener?.onInput(input); listener?.onOutput("") }
        state(token, State.THINKING)
        var firstPlayback = true
        val speech = if (speechOutputEnabled) VoiceAgentStreamingSpeech<AudioUtil.WavFileData>(
            checkActive = { ensureActive(token) },
            withSynthesis = { action ->
                models.withVoice({ ensureActive(token) }) { synthesize ->
                    action { segment ->
                        Log.d("VoiceAgent", "Synthesizing streamed segment: $segment")
                        synthesize(segment)
                    }
                }
            },
            play = { audio, checkActive ->
                if (firstPlayback) {
                    state(token, State.SPEAKING)
                    firstPlayback = false
                }
                Log.d("VoiceAgent", "Playing streamed segment")
                play(token, audio, checkActive)
            }) else null
        try {
            val response = models.generateResponse(input, {
                ensureActive(token)
                speech?.checkRunning()
            },
                onTemperature = { value -> emit(token) { listener?.onAirConditionerTemperature(value) } },
                takePhoto = { takePhoto(token) },
                onPhoto = { file ->
                    main.post {
                        val currentListener = listener
                        if (generation.get() == token && currentListener != null) currentListener.onPhoto(file)
                        else file.delete()
                    }
                },
                onSpeechText = { text, complete -> speech?.update(text, complete) }) { text ->
                emit(token) { listener?.onOutputPreview(text) }
            }
            ensureActive(token)
            emit(token) { listener?.onOutput(response) }
            if (speech != null) {
                Log.d("VoiceAgent", "Generation complete; draining streamed speech")
                state(token, State.SYNTHESIZING)
                speech.finish()
            }
        } finally {
            speech?.close()
        }
    }

    private fun startListening(token: Long) {
        models.startListening({ ensureActive(token) }) { text ->
            emit(token) { listener?.onIntermediateInput(text) }
        }
    }

    /** Wait off the UI thread; cancellation never waits for permission or a camera callback. */
    private fun takePhoto(token: Long): File {
        ensureActive(token)
        val results = ArrayBlockingQueue<Result<File>>(1)
        val lock = Any()
        var abandoned = false
        emit(token) {
            listener?.onTakePhoto { result ->
                synchronized(lock) {
                    if (abandoned) result.getOrNull()?.delete() else results.offer(result)
                }
            }
        }
        try {
            while (true) {
                ensureActive(token)
                val result = results.poll(100, TimeUnit.MILLISECONDS) ?: continue
                val file = result.getOrThrow()
                try { ensureActive(token) } catch (cancelled: CancellationException) {
                    file.delete()
                    throw cancelled
                }
                return file
            }
        } finally {
            synchronized(lock) {
                abandoned = true
                results.poll()?.getOrNull()?.delete()
            }
        }
    }

    private fun play(token: Long, audio: AudioUtil.WavFileData,
                     checkActive: () -> Unit = { ensureActive(token) }) {
        player.play(audio, checkActive) { chunk -> emit(token) { listener?.onWaveform(chunk) } }
        val tailSize = (audio.sampleRate * audio.channels * VoiceAgentPlayer.OUTPUT_SETTLE_MS / 1000).toInt()
        lastPlaybackTail = audio.copy(audioData = audio.audioData.copyOfRange(
            (audio.audioData.size - tailSize).coerceAtLeast(0), audio.audioData.size))
    }

    fun stop(error: String? = null) {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (!isActive || stopping) return
        isActive = false
        stopping = true
        val stopToken = generation.incrementAndGet()
        micInput?.recorder?.cancelRecording()
        micInput = null
        player.silence()
        notifyState(State.STOPPING)
        executor.execute {
            models.release()
            main.post {
                if (generation.get() == stopToken) {
                    stopping = false
                    listener?.onStopped(error)
                    listener = null
                }
            }
        }
    }

    private fun fail(token: Long, error: Exception) {
        Log.e("VoiceAgent", "Voice session failed", error)
        emit(token) { stop(error.message ?: "音声対話に失敗しました") }
    }
    private fun ensureActive(token: Long) { if (generation.get() != token) throw CancellationException() }
    private fun notifyState(state: State) { currentState = state; listener?.onState(state) }
    private fun state(token: Long, state: State) { ensureActive(token); emit(token) { notifyState(state) } }
    private fun emit(token: Long, action: () -> Unit) { main.post { if (generation.get() == token) action() } }
}
