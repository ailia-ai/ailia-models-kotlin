package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Log
import axip.ailia_llm.AiliaLLM
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Sample class demonstrating ailia LLM Tool Use (Function Calling).
 *
 * Gemma 4 (E2B / E4B)に「エアコンの温度を設定するツール」を渡し、
 * 「エアコンの温度を20度にしてください」のような指示からツール呼び出しを生成させる。
 *
 * Tool Useでは最初のターンからJSON履歴([AiliaLLM.setPromptJson])を使用し、
 * 生成結果は[AiliaLLM.getResponseJson]で構造化JSONとして受け取る。
 * Thinking(推論過程の出力)の有無は要求ごとに[chat]で指定する。
 */
class AiliaToolUseSample {
    private var llm: AiliaLLM? = null
    private var isInitialized = false
    private var modelPath: String? = null
    private var lastResult: String = ""
    /** JSON履歴。assistantはSDKが返したオブジェクトをそのまま積む。 */
    private var conversationHistory = JSONArray()
    private val cancelRequested = AtomicBoolean(false)

    /** 実行バックエンド。QNNは対応SoCの場合のみ選択できる。 */
    var backend: LLMBackend = LLMBackend.CPU

    /** 使用するモデル。ツール呼び出しに対応するGemma 4のみ。 */
    var modelType: LLMModelType = LLMModelType.GEMMA_4_E2B

    /** MTPを使うか。 */
    var mtpEnabled: Boolean = true

    /** 初期化時にMTPを有効にできたか。 */
    var isMtpActive: Boolean = false
        private set

    /** ダウンロード中のファイル名。進捗表示に使う。 */
    @Volatile
    var downloadingFileName: String? = null
        private set

    /** 直前の[initialize]でモデルのダウンロードに失敗した理由。失敗していなければnull。 */
    @Volatile
    var downloadError: String? = null
        private set

    /** ツールで設定したエアコンの温度(摂氏)。未設定ならnull。 */
    var airConditionerTemperature: Double? = null
        private set

    companion object {
        private const val TAG = "AiliaToolUseSample"
        private const val N_CTX = 8192 // Context window size
        // QNNモデルはコンテキスト長が変換時に固定されるため、0を指定してモデル内の値を使う
        private const val N_CTX_QNN = 0
        private const val MAX_GENERATION_STEPS = 4096

        /** ツール呼び出しと結果返却の往復回数の上限。 */
        private const val MAX_TOOL_TURNS = 8

        /** Tool Useのサンプリング温度。 */
        private const val TEMPERATURE = 0.0f

        /** エアコンの温度を設定するツールの名前。 */
        const val TOOL_NAME = "set_air_conditioner_temperature"

        /** チャット入力欄の初期値。 */
        const val DEFAULT_PROMPT = "エアコンの温度を20度にしてください"

        /** OpenAI互換のツール定義。 */
        private val TOOLS_JSON = """
            [{"type":"function","function":{
                "name":"$TOOL_NAME",
                "description":"エアコンの設定温度を変更します。",
                "parameters":{"type":"object","properties":{
                    "temperature":{"type":"number","description":"設定温度(摂氏)"}},
                    "required":["temperature"]}
            }}]
        """.trimIndent()
    }

    interface ToolUseListener {
        fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onStatus(status: String)

        /** assistantの新しいターンが始まった(ツール実行後の再生成を含む)。 */
        fun onTurnStart()

        /** 生成中のテキスト。Thinkingやツール呼び出し構文を含む生のプレビュー。 */
        fun onToken(token: String)

        /**
         * assistantのターンが終わった。SDKが解析した本文と、Thinking有効時の推論過程を渡す。
         * ツール呼び出しだけのターンでは両方とも空になる。
         */
        fun onTurnComplete(content: String, reasoning: String)

        /** ツールを実行した。[arguments]と[result]はJSON文字列。 */
        fun onToolCall(name: String, arguments: String, result: String)

        fun onComplete(fullResponse: String)
        fun onError(error: String)
    }

    /**
     * Downloads and initializes the model, then sets the tool definition.
     * This is a blocking operation that should be called on a background thread.
     */
    fun initialize(
        context: Context,
        progressListener: ModelDownloader.DownloadListener? = null,
    ): Boolean {
        downloadError = null
        val listener = ModelDownloader.recordingErrors(progressListener) { downloadError = it }
        return try {
            if (isInitialized) {
                release()
            }

            val qnnFileName = qnnFileNameOrNull()
            if (backend == LLMBackend.QNN && qnnFileName == null) {
                Log.e(TAG, "QNN model is not available for ${modelType.displayName} on ${QnnSupport.socName}")
                return false
            }

            val fileName = qnnFileName ?: modelType.fileName
            Log.i(TAG, "Downloading ${modelType.displayName} model ($fileName) for ${backend.displayName}...")
            downloadingFileName = fileName
            val modelFile = if (qnnFileName != null) {
                ModelDownloader.downloadQnnLLMModel(context, qnnFileName, listener)
            } else {
                ModelDownloader.downloadLLMModel(context, modelType.fileName, listener)
            }
            if (modelFile == null) {
                Log.e(TAG, "Failed to download model")
                downloadError = downloadError ?: "Failed to download $fileName"
                return false
            }
            modelPath = modelFile.absolutePath

            val mtpFileName = mtpFileNameOrNull()
            val mtpFile = mtpFileName?.let {
                Log.i(TAG, "Downloading MTP assistant ($it) for ${backend.displayName}...")
                downloadingFileName = it
                LLMMtp.download(context, it, backend, listener)
            }
            if (mtpFileName != null && mtpFile == null) {
                Log.e(TAG, "Failed to download MTP assistant")
                downloadError = downloadError ?: "Failed to download $mtpFileName"
                return false
            }

            // mmprojは後から開けないため、LLM系のどの画面でも使えるよう常に開く
            val projectorFileName = SharedLLM.projectorFileName(modelType, backend)
            val projectorFile = projectorFileName?.let {
                Log.i(TAG, "Downloading multimodal projector ($it) for ${backend.displayName}...")
                downloadingFileName = it
                SharedLLM.download(context, it, backend, listener)
            }
            if (projectorFileName != null && projectorFile == null) {
                Log.e(TAG, "Failed to download multimodal projector")
                downloadError = downloadError ?: "Failed to download $projectorFileName"
                return false
            }

            llm = SharedLLM.acquire(
                SharedLLM.Config(
                    modelPath = modelPath!!,
                    nCtx = if (qnnFileName != null) N_CTX_QNN else N_CTX,
                    mtpPath = mtpFile?.absolutePath,
                    projectorPath = projectorFile?.absolutePath,
                )
            )
            isMtpActive = SharedLLM.isMtpActive

            conversationHistory = JSONArray()
            airConditionerTemperature = null

            isInitialized = true
            Log.i(TAG, "Tool Use initialized. Context size: ${llm!!.getContextSize()}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Tool Use: ${e.message}", e)
            release()
            false
        }
    }

    /** Checks if the model is already downloaded. */
    fun isModelDownloaded(context: Context): Boolean {
        val qnnFileName = qnnFileNameOrNull()
        val modelDownloaded = if (qnnFileName != null) {
            ModelDownloader.isQnnLLMModelDownloaded(context, qnnFileName)
        } else {
            ModelDownloader.isLLMModelDownloaded(context, modelType.fileName)
        }
        val mtpDownloaded = mtpFileNameOrNull()?.let { LLMMtp.isDownloaded(context, it, backend) } ?: true
        val projectorDownloaded = SharedLLM.projectorFileName(modelType, backend)
            ?.let { SharedLLM.isDownloaded(context, it, backend) } ?: true
        return modelDownloaded && mtpDownloaded && projectorDownloaded
    }

    /** 現在のバックエンドでダウンロードするモデルファイル名。 */
    fun modelFileName(): String = qnnFileNameOrNull() ?: modelType.fileName

    private fun qnnFileNameOrNull(): String? =
        if (backend == LLMBackend.QNN) QnnSupport.llmQnnFileName(modelType) else null

    /** MTPを使う場合のAssistantのファイル名。MTP無効時や対応しないモデルではnull。 */
    private fun mtpFileNameOrNull(): String? =
        if (mtpEnabled) LLMMtp.assistantFileName(modelType, backend) else null

    /**
     * Runs one user request, executing tool calls until the model answers.
     * This is a blocking operation that should be called on a background thread.
     *
     * @param thinking trueならThinking(推論過程)を出力させる
     * @return The processing time in milliseconds, or -1 on failure
     */
    fun chat(userInput: String, thinking: Boolean, listener: ToolUseListener? = null): Long {
        val model = llm
        if (!isInitialized || model == null) {
            Log.e(TAG, "Tool Use not initialized")
            listener?.onError("Tool Use not initialized")
            return -1
        }

        val historySizeBeforeRequest = conversationHistory.length()
        return try {
            cancelRequested.set(false)
            val startTime = System.nanoTime()

            // モデルは他の画面と共有しているため、この画面の設定を毎回設定し直す。
            // ツール呼び出しの引数がぶれないよう、Tool Useではtemperatureを0にする
            model.setSamplingParams(40, 0.9f, TEMPERATURE, 1234)
            model.setTools(TOOLS_JSON)
            // Thinkingはプロンプトを設定する前に切り替える
            model.setThinking(thinking)

            conversationHistory.put(
                JSONObject().put("role", "user").put("content", userInput)
            )

            var finalResponse = ""
            var answered = false
            for (turn in 0 until MAX_TOOL_TURNS) {
                listener?.onTurnStart()
                model.setPromptJson(conversationHistory.toString())

                var done = false
                var generationSteps = 0
                val partialResponse = StringBuilder()
                while (!done && !cancelRequested.get() && generationSteps < MAX_GENERATION_STEPS) {
                    done = model.generate()
                    generationSteps++
                    val token = model.getDeltaText()
                    if (token.isNotEmpty()) {
                        partialResponse.append(token)
                        listener?.onToken(token)
                    }
                }
                if (cancelRequested.get()) {
                    // 中断した回も画面に残るため、途中までの応答を履歴に積んで次の発言の文脈にする。
                    // 実行済みのツール呼び出しと結果も残す(エアコンの状態は既に変わっている)。
                    // このリクエストで何も生成していなければ送信前の状態に戻す
                    if (partialResponse.isNotEmpty()) {
                        conversationHistory.put(
                            JSONObject().put("role", "assistant").put("content", partialResponse.toString())
                        )
                    } else if (turn == 0) {
                        rollbackHistory(historySizeBeforeRequest)
                    }
                    listener?.onError("Generation cancelled")
                    return -1
                }
                if (!done) {
                    rollbackHistory(historySizeBeforeRequest)
                    listener?.onError("Generation stopped after $MAX_GENERATION_STEPS steps")
                    return -1
                }

                // 生成が不完全な場合はここで例外になる。ツールは実行しない。
                val response = JSONObject(model.getResponseJson())
                Log.d(TAG, "Response JSON (turn $turn): $response")
                conversationHistory.put(response)
                listener?.onTurnComplete(
                    response.optString("content"),
                    response.optString("reasoning_content"),
                )

                val calls = response.optJSONArray("tool_calls")
                if (calls == null || calls.length() == 0) {
                    finalResponse = response.optString("content")
                    answered = true
                    break
                }

                listener?.onStatus("Running tool...")
                for (i in 0 until calls.length()) {
                    val call = calls.getJSONObject(i)
                    val function = call.getJSONObject("function")
                    val name = function.getString("name")
                    val arguments = function.getString("arguments")
                    val result = executeTool(name, arguments)
                    Log.i(TAG, "Tool call: $name($arguments) -> $result")
                    listener?.onToolCall(name, arguments, result)
                    conversationHistory.put(
                        JSONObject()
                            .put("role", "tool")
                            .put("tool_call_id", call.getString("id"))
                            // ツール結果のcontentは文字列で渡す
                            .put("content", result)
                    )
                }
            }

            if (!answered) {
                rollbackHistory(historySizeBeforeRequest)
                listener?.onError("Tool call did not finish in $MAX_TOOL_TURNS turns")
                return -1
            }

            lastResult = finalResponse
            val processingTime = (System.nanoTime() - startTime) / 1000000
            listener?.onComplete(finalResponse)
            Log.i(TAG, "Tool use completed in ${processingTime}ms. Response: $finalResponse")
            processingTime
        } catch (e: Exception) {
            rollbackHistory(historySizeBeforeRequest)
            Log.e(TAG, "Failed to run tool use: ${e.message}", e)
            listener?.onError("Failed to generate: ${e.message}")
            -1
        }
    }

    /**
     * ツールの実装。実際の機器の代わりに設定温度を保持する。
     * @return ツール結果のJSON文字列
     */
    private fun executeTool(name: String, arguments: String): String {
        if (name != TOOL_NAME) {
            return JSONObject().put("error", "unknown tool: $name").toString()
        }
        return try {
            val temperature = JSONObject(arguments).getDouble("temperature")
            airConditionerTemperature = temperature
            JSONObject()
                .put("status", "ok")
                .put("temperature", temperature)
                .toString()
        } catch (e: Exception) {
            Log.e(TAG, "Invalid tool arguments: $arguments", e)
            JSONObject().put("error", "invalid arguments: $arguments").toString()
        }
    }

    /** 失敗・中断時に、この要求で追加したメッセージを履歴から取り除く。 */
    private fun rollbackHistory(sizeBeforeRequest: Int) {
        while (conversationHistory.length() > sizeBeforeRequest) {
            conversationHistory.remove(conversationHistory.length() - 1)
        }
    }

    /** 表示用のエアコンの状態。 */
    fun airConditionerStatus(): String = airConditionerTemperature?.let {
        String.format(Locale.ROOT, "Air conditioner: %.1f C", it)
    } ?: "Air conditioner: not set"

    /** Requests the blocking generation loop to stop at the next token boundary. */
    fun cancelGeneration() {
        cancelRequested.set(true)
    }

    /** Clears the conversation history. */
    fun clearHistory() {
        conversationHistory = JSONArray()
        Log.i(TAG, "Conversation history cleared")
    }

    fun getLastResult(): String = lastResult

    /** この画面の会話を破棄する。モデル自体は[SharedLLM]が保持し、他のLLM系の画面と使い回す。 */
    fun release() {
        cancelGeneration()
        // モデル自体は[SharedLLM]が保持する。ツール定義は他の画面が使う前に各画面で設定し直す
        llm = null
        isInitialized = false
        isMtpActive = false
        modelPath = null
        conversationHistory = JSONArray()
        Log.i(TAG, "Tool Use released")
    }
}
