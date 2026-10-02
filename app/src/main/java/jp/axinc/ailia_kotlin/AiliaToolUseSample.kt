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

    /** 使用するモデル。ツール呼び出しに対応するGemma 4のみ。 */
    var modelType: LLMModelType = LLMModelType.GEMMA_4_E2B

    /** ツールで設定したエアコンの温度(摂氏)。未設定ならnull。 */
    var airConditionerTemperature: Double? = null
        private set

    companion object {
        private const val TAG = "AiliaToolUseSample"
        private const val N_CTX = 8192 // Context window size
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
        return try {
            if (isInitialized) {
                release()
            }

            Log.i(TAG, "Downloading ${modelType.displayName} model (${modelType.fileName})...")
            val modelFile = ModelDownloader.downloadLLMModel(context, modelType.fileName, progressListener)
            if (modelFile == null) {
                Log.e(TAG, "Failed to download model")
                return false
            }
            modelPath = modelFile.absolutePath

            Log.i(TAG, "Creating AiliaLLM instance...")
            llm = AiliaLLM()

            Log.i(TAG, "Opening model file: $modelPath")
            llm!!.openModelFile(modelPath!!, N_CTX)
            // ツール呼び出しの引数がぶれないよう、Tool Useではtemperatureを0にする
            llm!!.setSamplingParams(40, 0.9f, TEMPERATURE, 1234)

            llm!!.setTools(TOOLS_JSON)

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
    fun isModelDownloaded(context: Context): Boolean =
        ModelDownloader.isLLMModelDownloaded(context, modelType.fileName)

    /** ダウンロードするモデルファイル名。 */
    fun modelFileName(): String = modelType.fileName

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
                while (!done && !cancelRequested.get() && generationSteps < MAX_GENERATION_STEPS) {
                    done = model.generate()
                    generationSteps++
                    val token = model.getDeltaText()
                    if (token.isNotEmpty()) {
                        listener?.onToken(token)
                    }
                }
                if (cancelRequested.get()) {
                    rollbackHistory(historySizeBeforeRequest)
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

    /** Releases the LLM resources. */
    fun release() {
        cancelGeneration()
        try {
            // ツール設定は必ず解除してから破棄する
            llm?.setTools(null)
        } catch (e: Exception) {
            Log.e(TAG, "Error clearing tools: ${e.message}")
        }
        try {
            llm?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing Tool Use: ${e.message}")
        } finally {
            llm = null
            isInitialized = false
            modelPath = null
            conversationHistory = JSONArray()
            Log.i(TAG, "Tool Use released")
        }
    }
}
