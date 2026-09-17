package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Log
import axip.ailia_llm.AiliaLLM
import axip.ailia_llm.AiliaLLMChatMessage
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Available LLM models (URLs follow ailia-models-flutter: /gemma/<fileName>).
 */
enum class LLMModelType(val displayName: String, val fileName: String) {
    GEMMA_4_E2B("Gemma 4 E2B", "gemma-4-E2B-it-Q4_K_M.gguf"),
    GEMMA_4_E4B("Gemma 4 E4B", "gemma-4-E4B-it-Q4_K_M.gguf"),
    GEMMA_2_2B("Gemma 2 2B", "gemma-2-2b-it-Q4_K_M.gguf"),
}

/**
 * LLMの実行バックエンド。
 * QNNはSoC固有の変換済みモデル(.qnn)を使うため、対応SoC/モデルの場合のみ選択できる。
 */
enum class LLMBackend(val displayName: String) {
    CPU("CPU"),
    QNN("QNN (NPU)"),
}

/**
 * 1回の生成で計測したPrefill / Decodeの性能。
 *
 * Prefillは最初のgenerate()呼び出しで実行されるため、PPS(prefill tokens/s)は
 * 「プロンプトのトークン数 / 最初のgenerate()の時間」で求める。
 */
data class LLMPerformance(
    val promptTokens: Int,
    val prefillMs: Long,
    val generatedTokens: Int,
    val decodeMs: Long,
) {
    /** Prefillスループット (tokens/s)。 */
    val prefillTokensPerSecond: Double =
        if (prefillMs > 0) promptTokens * 1000.0 / prefillMs else 0.0

    /** Decodeスループット (tokens/s)。最初のトークンはPrefillに含めるため除外する。 */
    val decodeTokensPerSecond: Double =
        if (decodeMs > 0) generatedTokens * 1000.0 / decodeMs else 0.0

    /** UIに表示する1行のサマリ。 */
    fun summary(): String = String.format(
        java.util.Locale.ROOT,
        "Prefill %d tokens %.2f tokens/s (%d ms) / Decode %d tokens %.2f tokens/s",
        promptTokens, prefillTokensPerSecond, prefillMs, generatedTokens, decodeTokensPerSecond,
    )
}

/**
 * Sample class demonstrating ailia LLM inference for text generation.
 */
class AiliaLLMSample {
    private var llm: AiliaLLM? = null
    private var isInitialized = false
    private var lastResult: String = ""
    private var modelPath: String? = null
    /** 直近の生成で計測したPrefill / Decodeの性能。 */
    var lastPerformance: LLMPerformance? = null
        private set
    private val conversationHistory = mutableListOf<AiliaLLMChatMessage>()
    private val cancelRequested = AtomicBoolean(false)

    var modelType: LLMModelType = LLMModelType.GEMMA_4_E2B
    var backend: LLMBackend = LLMBackend.CPU

    companion object {
        private const val TAG = "AiliaLLMSample"
        private const val N_CTX = 8192 // Context window size
        // QNNモデルはコンテキスト長が変換時に固定されるため、0を指定してモデル内の値を使う
        private const val N_CTX_QNN = 0
        private const val MAX_GENERATION_STEPS = 4096
    }

    interface LLMListener {
        fun onToken(token: String)
        fun onComplete(fullResponse: String)
        fun onError(error: String)
    }

    /**
     * Downloads and initializes the selected LLM model ([modelType]).
     * This is a blocking operation that should be called on a background thread.
     *
     * @param context The Android context
     * @param progressListener Optional listener for download progress
     * @return true if initialization succeeded, false otherwise
     */
    fun initialize(
        context: Context,
        progressListener: ModelDownloader.DownloadListener? = null
    ): Boolean {
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
            val modelFile = if (qnnFileName != null) {
                ModelDownloader.downloadQnnLLMModel(context, qnnFileName, progressListener)
            } else {
                ModelDownloader.downloadLLMModel(context, modelType.fileName, progressListener)
            }
            if (modelFile == null) {
                Log.e(TAG, "Failed to download model")
                return false
            }
            modelPath = modelFile.absolutePath

            Log.i(TAG, "Creating AiliaLLM instance...")
            llm = AiliaLLM()

            Log.i(TAG, "Opening model file: $modelPath")
            llm!!.openModelFile(modelPath!!, if (qnnFileName != null) N_CTX_QNN else N_CTX)

            // Set default sampling parameters
            llm!!.setSamplingParams(40, 0.9f, 0.4f, 1234)

            // Add system prompt
            conversationHistory.clear()
            conversationHistory.add(AiliaLLMChatMessage("system", "You are a helpful assistant. Keep your responses brief and concise."))

            isInitialized = true
            Log.i(TAG, "LLM initialized successfully. Context size: ${llm!!.getContextSize()}")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM: ${e.message}", e)
            release()
            false
        }
    }

    /**
     * Checks if the model is already downloaded.
     */
    fun isModelDownloaded(context: Context): Boolean {
        val qnnFileName = qnnFileNameOrNull()
        return if (qnnFileName != null) {
            ModelDownloader.isQnnLLMModelDownloaded(context, qnnFileName)
        } else {
            ModelDownloader.isLLMModelDownloaded(context, modelType.fileName)
        }
    }

    /**
     * PPS計測用の評価テキストを作る。
     * モデルが初期化済みならモデルのトークナイザで正確なトークン数に合わせる。
     */
    fun buildBenchmarkPrompt(
        context: Context,
        targetTokens: Int = BenchmarkPrompt.DEFAULT_TARGET_TOKENS,
    ): BenchmarkPrompt.Result {
        val model = llm.takeIf { isInitialized } ?: return BenchmarkPrompt.build(context, targetTokens)
        return BenchmarkPrompt.build(context, targetTokens) { text -> model.getTokenCount(text) }
    }

    /** 現在のバックエンド/モデルでダウンロードするモデルファイル名。 */
    fun modelFileName(): String = qnnFileNameOrNull() ?: modelType.fileName

    /** QNNを選択している場合のQNNモデルファイル名。CPU時や未対応の組み合わせではnull。 */
    private fun qnnFileNameOrNull(): String? =
        if (backend == LLMBackend.QNN) QnnSupport.llmQnnFileName(modelType) else null

    /**
     * Generates a response for the given user input.
     * This is a blocking operation that should be called on a background thread.
     *
     * @param userInput The user's message
     * @param listener Optional listener for streaming tokens
     * @return The processing time in milliseconds
     */
    fun chat(userInput: String, listener: LLMListener? = null): Long {
        if (!isInitialized || llm == null) {
            Log.e(TAG, "LLM not initialized")
            listener?.onError("LLM not initialized")
            return -1
        }

        val historySizeBeforeRequest = conversationHistory.size
        return try {
            cancelRequested.set(false)
            lastPerformance = null
            val startTime = System.nanoTime()

            // Add user message to conversation history
            conversationHistory.add(AiliaLLMChatMessage("user", userInput))

            // Set the prompt
            llm!!.setPrompt(conversationHistory.toTypedArray())
            val promptTokens = llm!!.getPromptTokenCount()

            // Generate response token by token
            val responseBuilder = StringBuilder()
            var done = false
            var generationSteps = 0
            // Prefillは最初のgenerate()で実行されるため、その時間を分けて計測する
            var prefillNanos = 0L
            var decodeStartNanos = 0L

            while (!done && !cancelRequested.get() && generationSteps < MAX_GENERATION_STEPS) {
                val stepStart = System.nanoTime()
                done = llm!!.generate()
                if (generationSteps == 0) {
                    prefillNanos = System.nanoTime() - stepStart
                    decodeStartNanos = System.nanoTime()
                }
                generationSteps++
                val token = llm!!.getDeltaText()
                if (token.isNotEmpty()) {
                    responseBuilder.append(token)
                    listener?.onToken(token)
                }
            }
            val decodeNanos = if (decodeStartNanos > 0) System.nanoTime() - decodeStartNanos else 0L

            if (cancelRequested.get()) {
                while (conversationHistory.size > historySizeBeforeRequest) conversationHistory.removeAt(conversationHistory.lastIndex)
                listener?.onError("Generation cancelled")
                return -1
            }
            if (!done) {
                while (conversationHistory.size > historySizeBeforeRequest) conversationHistory.removeAt(conversationHistory.lastIndex)
                listener?.onError("Generation stopped after $MAX_GENERATION_STEPS steps")
                return -1
            }

            val fullResponse = responseBuilder.toString()
            lastResult = fullResponse

            // Add assistant response to conversation history
            conversationHistory.add(AiliaLLMChatMessage("assistant", fullResponse))

            val endTime = System.nanoTime()
            val processingTime = (endTime - startTime) / 1000000

            val performance = LLMPerformance(
                promptTokens = promptTokens,
                prefillMs = prefillNanos / 1000000,
                // 最初のトークンはPrefillに含まれるため、Decodeのトークン数から除く
                generatedTokens = (generationSteps - 1).coerceAtLeast(0),
                decodeMs = decodeNanos / 1000000,
            )
            lastPerformance = performance

            listener?.onComplete(fullResponse)
            Log.i(TAG, "Chat completed in ${processingTime}ms. ${performance.summary()}")
            Log.i(TAG, "Response: $fullResponse")

            processingTime

        } catch (e: Exception) {
            while (conversationHistory.size > historySizeBeforeRequest) conversationHistory.removeAt(conversationHistory.lastIndex)
            Log.e(TAG, "Failed to generate response: ${e.message}", e)
            listener?.onError("Failed to generate: ${e.message}")
            -1
        }
    }

    /** Requests the blocking generation loop to stop at the next token boundary. */
    fun cancelGeneration() {
        cancelRequested.set(true)
    }

    /**
     * Clears the conversation history and resets the context.
     */
    fun clearHistory() {
        conversationHistory.clear()
        conversationHistory.add(AiliaLLMChatMessage("system", "You are a helpful assistant."))
        Log.i(TAG, "Conversation history cleared")
    }

    /**
     * Gets the last generated response.
     */
    fun getLastResult(): String {
        return lastResult
    }

    /**
     * Gets the number of backends available.
     */
    fun getBackendCount(): Int {
        return try {
            AiliaLLM.getBackendCount()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get backend count: ${e.message}")
            0
        }
    }

    /**
     * Gets the name of a backend by index.
     */
    fun getBackendName(index: Int): String {
        return try {
            AiliaLLM.getBackendName(index)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get backend name: ${e.message}")
            "Unknown"
        }
    }

    /**
     * Releases the LLM resources.
     */
    fun release() {
        cancelGeneration()
        try {
            llm?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing LLM: ${e.message}")
        } finally {
            llm = null
            isInitialized = false
            modelPath = null
            conversationHistory.clear()
            Log.i(TAG, "LLM released")
        }
    }
}
