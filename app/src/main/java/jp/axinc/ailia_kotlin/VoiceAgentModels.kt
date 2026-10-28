package jp.axinc.ailia_kotlin

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Base64
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import axip.ailia_llm.AiliaLLM
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.CancellationException

/**
 * Owns downloads, inference and conversation history. Call only on the session's model executor:
 * QNN model creation and destruction must stay on that thread. Response synthesis uses a separate,
 * worker-local voice instance so it can overlap generation without sharing a native model.
 */
internal class VoiceAgentModels(private val context: Context) {
    private val speech = AiliaSpeechSample(ModelDownloader.modelDirectory(context))
    private var llm: AiliaLLM? = null
    private var speechTurn: VoiceAgentSpeechTurn? = null
    private lateinit var config: AiliaVoiceAgentSample.Config
    private lateinit var llmModelFile: File
    private lateinit var projectorFile: File
    private var mtpModelFile: File? = null
    // Each entry is one complete voice turn, including tool calls/results and captured images.
    private val history = mutableListOf<MutableList<JSONObject>>()
    private val tools = VoiceAgentTools()

    fun prepare(config: AiliaVoiceAgentSample.Config, checkActive: () -> Unit,
                onDownload: (String, Long, Long) -> Unit) {
        this.config = config
        val progress = object : ModelDownloadListener {
            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                checkActive()
                onDownload(fileName, bytesDownloaded, totalBytes)
            }
            override fun onComplete() = Unit
            override fun onError(error: String) { checkActive(); error(error) }
        }
        checkActive()
        speech.currentModelType = config.speechModelType
        speech.useQnn = config.speechQnn
        speech.language = "ja"
        check(speech.downloadModel(listener = progress)) { "Whisperモデルのダウンロードに失敗しました" }
        checkActive()

        val modelType = config.llmModelType
        val qnnFile = if (config.llmBackend == LLMBackend.QNN) {
            checkNotNull(QnnSupport.llmQnnFileName(modelType)) { "この端末ではGemmaのQNNを使用できません" }
        } else null
        val fileName = qnnFile ?: modelType.fileName
        fun llmProgress(name: String) = object : ModelDownloader.DownloadListener {
            override fun onProgress(bytesDownloaded: Long, totalBytes: Long) = progress.onProgress(name, bytesDownloaded, totalBytes)
            override fun onComplete(file: File) = Unit
            override fun onError(error: String) = progress.onError(error)
        }
        val modelFile = if (qnnFile != null) ModelDownloader.downloadQnnLLMModel(context, fileName, llmProgress(fileName))
            else ModelDownloader.downloadLLMModel(context, fileName, llmProgress(fileName))
        checkActive()
        llmModelFile = checkNotNull(modelFile) { "Gemmaモデルのダウンロードに失敗しました" }

        val projectorName = checkNotNull(SharedLLM.projectorFileName(modelType, config.llmBackend)) {
            "このモデルでは画像入力を使用できません"
        }
        projectorFile = checkNotNull(SharedLLM.download(context, projectorName, config.llmBackend, llmProgress(projectorName))) {
            "画像エンコーダのダウンロードに失敗しました"
        }
        checkActive()

        val mtpName = if (config.mtpEnabled) LLMMtp.assistantFileName(modelType, config.llmBackend) else null
        mtpModelFile = mtpName?.let { name ->
            checkNotNull(LLMMtp.download(context, name, config.llmBackend, llmProgress(name))) {
                "MTP Assistantのダウンロードに失敗しました"
            }
        }
        checkActive()

        if (config.speechOutputEnabled || config.inputMode == AiliaVoiceAgentSample.InputMode.SYNTHESIZED_QUERIES) {
            val voice = createVoice()
            check(voice.downloadModel(progress)) { "GPT-SoVITSモデルのダウンロードに失敗しました" }
        }
        checkActive()
        history.clear()
        tools.reset()
    }

    /** Keep generated replies on cancellation; roll back only turns with no assistant content or calls. */
    fun withConversationTurn(isCancelled: () -> Boolean, action: () -> Unit) {
        val historyBeforeTurn = history.toList()
        try {
            action()
        } finally {
            if (isCancelled()) {
                val current = history.lastOrNull()
                val hasReply = current != null && historyBeforeTurn.none { it === current } &&
                    current.any { it.optString("role") == "assistant" }
                if (!hasReply) {
                    history.clear()
                    history.addAll(historyBeforeTurn)
                }
            }
        }
    }

    fun startListening(checkActive: () -> Unit, onIntermediateInput: ((String) -> Unit)? = null) {
        checkActive()
        check(speech.initializeSpeech(config.speechEnvId, liveMode = false)) { "Whisperの初期化に失敗しました" }
        checkActive()
        speechTurn = speech.createBufferedTurn(onIntermediateInput, checkActive)
    }

    fun pushSpeech(audio: FloatArray, channels: Int, sampleRate: Int): Boolean =
        checkNotNull(speechTurn) { "Speech input not initialized" }.push(audio, channels, sampleRate)

    fun finishRecognition(checkActive: () -> Unit): String {
        val input = try {
            checkActive()
            voiceAgentTranscript(checkNotNull(speechTurn).finish(checkActive))
        } catch (error: Exception) {
            // An intermediate callback can abort native decoding after the turn was cancelled.
            checkActive()
            throw error
        } finally {
            speechTurn = null
            speech.releaseSpeech()
        }
        checkActive()
        return input
    }

    fun generateResponse(input: String, checkActive: () -> Unit, onTemperature: (Double) -> Unit,
                         takePhoto: () -> File, onPhoto: (File) -> Unit,
                         onSpeechText: (text: String, complete: Boolean) -> Unit = { _, _ -> },
                         onOutput: (String) -> Unit): String {
        val model = AiliaLLM()
        llm = model
        var currentTurn: MutableList<JSONObject>? = null
        var partialReply = ""
        val photos = mutableListOf<JSONObject>()
        try {
            model.openModelFile(llmModelFile.absolutePath, if (config.llmBackend == LLMBackend.QNN) 0 else 4096)
            checkActive()
            mtpModelFile?.let { file ->
                check(LLMMtp.open(model, file.absolutePath)) { "MTPの初期化に失敗しました" }
            }
            model.openMultimodalProjectorFile(projectorFile.absolutePath)
            checkActive()
            // Keep whole user turns: assistant tool calls and their results must stay together.
            while (history.size >= 4) removeOldestTurn()
            val request = JSONObject().put("role", "user").put("content", input)
            val turn = mutableListOf(request)
            history += turn
            currentTurn = turn
            val contextSize = model.getContextSize()
            val definitions = tools.definitions()
            val fixedTokens = model.getTokenCount(definitions) + model.getTokenCount(SYSTEM_PROMPT + CONTINUATION_INSTRUCTION) + 64
            model.setSamplingParams(40, 0.9f, 0.0f, 1234)
            model.setTools(definitions)
            model.setThinking(false)
            val continuedReply = StringBuilder()
            var toolRounds = 0
            var emptyContinuations = 0
            var compactedForPrompt = false
            var hasGeneratedReply = false
            fun compactHistory() {
                val before = history.flatten().sumOf { messageTokens(model, it) }
                request.put("content", input)
                val available = (contextSize - fixedTokens - 128).coerceAtLeast(192)
                val budget = (before / 8).coerceAtLeast(192).coerceAtMost(available)
                voiceAgentCompactHistory(history, turn, budget,
                    maximumBudget = if (emptyContinuations < 2) available else budget,
                    request = request.takeIf { hasGeneratedReply }) { messageTokens(model, it) }
                Log.i("VoiceAgent", "Context compacted: $before -> ${turn.sumOf { messageTokens(model, it) }} history tokens")
            }
            fun continueResponse() {
                if (partialReply.isNotBlank()) {
                    hasGeneratedReply = true
                    turn += JSONObject().put("role", "assistant").put("content", partialReply)
                    continuedReply.append(partialReply)
                    partialReply = ""
                    emptyContinuations = 0
                } else {
                    emptyContinuations++
                }
                compactHistory()
                compactedForPrompt = true
                if (continuedReply.isNotEmpty() && turn.lastOrNull()?.optString("content") != CONTINUE_PROMPT) {
                    turn += JSONObject().put("role", "user").put("content", CONTINUE_PROMPT)
                }
                // Supported models have ample room for tools plus a short history. Repeated
                // CONTEXT_FULL without producing text indicates an unusable model/context setup.
                check(emptyContinuations <= 3) { "短縮した履歴でもモデルのコンテキストに収まりません" }
                Log.i("VoiceAgent", "Continuing response after context boundary")
            }
            while (true) {
                checkActive()
                if (!compactedForPrompt && fixedTokens + history.flatten().sumOf { messageTokens(model, it) } > contextSize - 256) compactHistory()
                compactedForPrompt = false
                var complete = false
                try {
                    val prompt = SYSTEM_PROMPT + if (continuedReply.isNotEmpty()) CONTINUATION_INSTRUCTION else ""
                    model.setPromptJson(JSONArray().put(JSONObject().put("role", "system").put("content", prompt)).apply {
                        history.flatten().forEach { put(it) }
                    }.toString())
                    val promptTokens = model.getPromptTokenCount()
                    Log.d("VoiceAgent", "Tool prompt tokens: $promptTokens")
                    // Stop before the native window is exhausted, reserving a full MTP batch
                    // and template/EOS space. The former 512-token error is no longer an end point.
                    val generationLimit = (contextSize - promptTokens - 16).coerceAtLeast(0)
                    val preview = VoiceAgentTextStream(trimLeadingWhitespace = continuedReply.isEmpty())
                    val generated = StringBuilder()
                    var generatedTokens = 0
                    val batchLimit = if (mtpModelFile != null) LLMMtp.N_DRAFT + 1 else 1
                    if (generationLimit >= 64) {
                        for (step in 0 until generationLimit) {
                            if (generatedTokens + batchLimit > generationLimit) break
                            checkActive()
                            complete = model.generate()
                            if (complete) break
                            val delta = model.getDeltaText()
                            generated.append(delta)
                            generatedTokens = model.getTokenCount(generated.toString())
                            val text = preview.append(delta)
                            if (text != partialReply) {
                                checkActive()
                                partialReply = text
                                val visible = continuedReply.toString() + text
                                onOutput(visible)
                                onSpeechText(visible, false)
                            }
                            // Prefer a complete sentence/line when nearing the window boundary.
                            // Unexpected native CONTEXT_FULL still recovers from any partial text.
                            if (generatedTokens >= generationLimit - maxOf(32, generationLimit / 8) &&
                                RESPONSE_BOUNDARY.containsMatchIn(partialReply)) break
                        }
                    }
                } catch (error: RuntimeException) {
                    checkActive()
                    if (!voiceAgentIsContextFull(error)) throw error
                    Log.i("VoiceAgent", "Native context full; retaining generated text", error)
                }
                checkActive()
                if (!complete) {
                    continueResponse()
                    continue
                }
                val response = JSONObject(model.getResponseJson())
                Log.d("VoiceAgent", "Response JSON: $response")
                turn += response
                hasGeneratedReply = true
                partialReply = ""
                val text = response.optString("content").trimEnd().let {
                    if (continuedReply.isEmpty()) it.trimStart() else it
                }
                val visible = continuedReply.toString() + text
                onSpeechText(visible, true)
                val calls = response.optJSONArray("tool_calls")
                if (calls == null || calls.length() == 0) {
                    check(visible.isNotBlank()) { "応答が空でした。対話を開始し直してください" }
                    onOutput(visible)
                    return visible
                }
                check(++toolRounds <= 8) { "ツールの呼び出しが上限に達しました。もう一度短く話しかけてください" }
                continuedReply.setLength(0)
                emptyContinuations = 0
                // A tool round may contain introductory prose; the next round starts a new preview.
                onOutput("")
                for (index in 0 until calls.length()) {
                    checkActive()
                    val call = calls.getJSONObject(index)
                    val function = call.getJSONObject("function")
                    val name = function.getString("name")
                    val result = tools.execute(name, function.getString("arguments"), onTemperature) {
                        val file = takePhoto()
                        try {
                            checkActive()
                            val image = photoPart(file)
                            checkActive()
                            onPhoto(file)
                            photos += image
                            file
                        } catch (error: Exception) {
                            file.delete()
                            throw error
                        }
                    }
                    Log.i("VoiceAgent", "Tool call: $name -> $result")
                    turn += JSONObject().put("role", "tool")
                        .put("tool_call_id", call.getString("id")).put("content", result)
                }
                appendPhotos(turn, photos)
            }
        } catch (cancelled: CancellationException) {
            currentTurn?.let { turn ->
                completeInterruptedTools(turn)
                appendPhotos(turn, photos)
                if (partialReply.isNotBlank()) {
                    turn += JSONObject().put("role", "assistant").put("content", partialReply.trimEnd())
                }
            }
            throw cancelled
        } finally {
            model.destroy()
            llm = null
        }
    }

    private fun appendPhotos(turn: MutableList<JSONObject>, photos: MutableList<JSONObject>) {
        if (photos.isEmpty()) return
        // The SDK accepts media in user messages, following the matched tool results.
        turn += JSONObject().put("role", "user").put("content", JSONArray()
            .put(JSONObject().put("type", "text").put("text", "撮影した写真です。元の依頼に、この写真を踏まえて答えてください。"))
            .apply { photos.forEach { put(it) } })
        photos.clear()
    }

    /** Keep already executed calls, and resolve interrupted/unstarted calls without executing them. */
    private fun completeInterruptedTools(turn: MutableList<JSONObject>) {
        val assistant = turn.indexOfLast { it.optString("role") == "assistant" }
        if (assistant < 0) return
        val calls = turn[assistant].optJSONArray("tool_calls") ?: return
        val results = turn.drop(assistant + 1).takeWhile { it.optString("role") == "tool" }
        val returnedIds = results.map { it.optString("tool_call_id") }.toSet()
        var position = assistant + 1 + results.size
        for (index in 0 until calls.length()) {
            val id = calls.getJSONObject(index).getString("id")
            if (id !in returnedIds) {
                val result = JSONObject().put("error", "ユーザーが応答を中断したため、ツールの完了を確認できませんでした。")
                turn.add(position++, JSONObject().put("role", "tool").put("tool_call_id", id).put("content", result.toString()))
            }
        }
    }

    private fun removeOldestTurn(): Boolean {
        if (history.size <= 1) return false
        history.removeAt(0)
        return true
    }

    /** Reserve image tokens without tokenizing the much larger Base64 payload as text. */
    private fun messageTokens(model: AiliaLLM, message: JSONObject): Int {
        val content = message.optJSONArray("content") ?: return model.getTokenCount(message.toString()) + 32
        var count = 32
        for (index in 0 until content.length()) {
            val part = content.getJSONObject(index)
            count += if (part.optString("type") == "image") 512 else model.getTokenCount(part.optString("text"))
        }
        return count
    }

    /** Store bounded, upright JPEG data in history independently of the preview file lifetime. */
    private fun photoPart(file: File): JSONObject {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true; inSampleSize = 1 }
        BitmapFactory.decodeFile(file.absolutePath, options)
        require(options.outWidth > 0 && options.outHeight > 0) { "写真を読み込めませんでした" }
        while (maxOf(options.outWidth, options.outHeight) / options.inSampleSize > 512) options.inSampleSize *= 2
        options.inJustDecodeBounds = false
        val bitmap = checkNotNull(BitmapFactory.decodeFile(file.absolutePath, options)) { "写真を読み込めませんでした" }
        var upright: Bitmap? = null
        try {
            val exif = ExifInterface(file.absolutePath)
            val matrix = Matrix().apply {
                postRotate(exif.rotationDegrees.toFloat())
                if (exif.isFlipped) postScale(-1f, 1f)
            }
            upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            val bytes = ByteArrayOutputStream().use { output ->
                check(upright.compress(Bitmap.CompressFormat.JPEG, 85, output)) { "写真の変換に失敗しました" }
                output.toByteArray()
            }
            return JSONObject().put("type", "image").put("data", Base64.encodeToString(bytes, Base64.NO_WRAP))
        } finally {
            if (upright !== bitmap) upright?.recycle()
            bitmap.recycle()
        }
    }

    companion object {
        private val RESPONSE_BOUNDARY = Regex("[。！？!?\\n\\r][」』）)\\]】〉》”’\\\"'\\s]*$")
        private const val CONTINUE_PROMPT = "続けてください"
        private const val CONTINUATION_INSTRUCTION = " Continue exactly from the last assistant message. Complete its unfinished final sentence or line before moving on. Do not repeat, summarize, or restart the answer."
        private const val SYSTEM_PROMPT = "You are a helpful assistant. Use tools to perform requested actions. Reply briefly in Japanese after receiving tool results."
    }

    fun synthesize(text: String, checkActive: () -> Unit): AudioUtil.WavFileData =
        withVoice(checkActive) { synthesize -> synthesize(text) }

    private fun createVoice() = AiliaVoiceSample(ModelDownloader.modelDirectory(context)).apply {
        modelType = VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE
    }

    /** Create, use and release a private voice instance on the calling synthesis worker. */
    fun <T> withVoice(checkActive: () -> Unit, action: (synthesize: (String) -> AudioUtil.WavFileData) -> T): T {
        checkActive()
        val voice = createVoice()
        val reference = AudioUtil().loadRawAudio(context.resources.openRawResource(R.raw.reference_audio_girl))
        try {
            check(voice.initializeVoice(config.voiceEnvId)) { "GPT-SoVITSの初期化に失敗しました" }
            checkActive()
            check(voice.setReferenceAudio(reference.audioData, reference.channels, reference.sampleRate,
                "水をマレーシアから買わなくてはならない。", "ja")) { "参照音声の設定に失敗しました" }
            checkActive()
            return action { text -> synthesizeInitialized(voice, text, checkActive) }
        } finally {
            voice.releaseVoice()
        }
    }

    private fun synthesizeInitialized(voice: AiliaVoiceSample, text: String,
                                      checkActive: () -> Unit): AudioUtil.WavFileData {
        checkActive()
        check(voice.synthesizeText(text, "ja", autoPlay = false) >= 0) {
            "音声合成に失敗しました"
        }
        checkActive()
        return AudioUtil.WavFileData(voice.lastAudioSampleRate, voice.lastAudioChannels, checkNotNull(voice.lastAudioData))
    }

    fun release(clearHistory: Boolean = true) {
        speechTurn = null
        speech.releaseSpeech()
        runCatching { llm?.destroy() }.onFailure { Log.w("VoiceAgent", "Failed to release LLM", it) }
        llm = null
        if (clearHistory) history.clear()
    }
}
