package jp.axinc.ailia_kotlin

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import axip.ailia_llm.AiliaLLM
import axip.ailia_llm.AiliaLLMMediaData
import axip.ailia_llm.AiliaLLMMultimodalChatMessage
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean

/**
 * マルチモーダル入力の種類。
 * VLM(画像)とALM(音声)で同じGemma 4 E2B + mmprojを使い、入力メディアのみが異なる。
 *
 * @property mediaBeforeText メディアを質問より前に置くか。Gemma 4のモデルカードの推奨
 *   (画像はテキストの前、音声はテキストの後)に従う。
 */
enum class MultimodalMediaType(val mediaType: String, val mediaBeforeText: Boolean) {
    IMAGE("image", mediaBeforeText = true),
    AUDIO("audio", mediaBeforeText = false),
}

/**
 * VLM / ALMの1回の生成で計測した性能。
 *
 * 画像・音声のエンコードはsetPrompt()、Prefillは最初のgenerate()で実行されるため、
 * TTFTはsetPrompt()の開始から最初のgenerate()が返るまでの時間で求める。
 */
data class MultimodalLLMPerformance(
    val ttftMs: Long,
    val generatedTokens: Int,
    val decodeMs: Long,
) {
    /** Decodeスループット (tokens/s)。最初のトークンはTTFTに含めるため除外する。 */
    val decodeTokensPerSecond: Double =
        if (decodeMs > 0) generatedTokens * 1000.0 / decodeMs else 0.0

    /** UIに表示する1行のサマリ。 */
    fun summary(): String = String.format(
        java.util.Locale.ROOT,
        "TTFT %d ms / Decode %d tokens %.2f tokens/s",
        ttftMs, generatedTokens, decodeTokensPerSecond,
    )
}

/**
 * Sample class demonstrating ailia Multimodal LLM inference (VLM / ALM).
 *
 * Gemma 4 (E2B / E4B)をテキストモデルとして使い、mmprojで画像または音声を入力する。
 */
class AiliaMultimodalLLMSample(
    private val mediaType: MultimodalMediaType = MultimodalMediaType.IMAGE,
) {
    private var llm: AiliaLLM? = null
    private var isInitialized = false
    private var lastResult: String = ""
    private var modelPath: String? = null
    private var projectorPath: String? = null
    private var sampleMediaPath: String? = null
    private val conversationHistory = mutableListOf<AiliaLLMMultimodalChatMessage>()
    /** 会話履歴にメディアを送ったか。メディアは会話の最初の質問だけで送る。 */
    private var mediaSent = false
    private val cancelRequested = AtomicBoolean(false)

    /** 使用するモデル。mmprojを持つGemma 4のみ。 */
    var modelType: LLMModelType = LLMModelType.GEMMA_4_E2B

    /** MTPを使うか。 */
    var mtpEnabled: Boolean = true

    /** 初期化時にMTPを有効にできたか。 */
    var isMtpActive: Boolean = false
        private set

    /** 直近の生成で計測したTTFT / Decodeの性能。 */
    var lastPerformance: MultimodalLLMPerformance? = null
        private set

    companion object {
        private const val TAG = "AiliaMultimodalLLM"
        private const val N_CTX = 8192 // Context window size
        private const val MAX_GENERATION_STEPS = 4096
    }

    interface MultimodalLLMListener {
        fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onStatus(status: String)
        fun onToken(token: String)
        fun onComplete(fullResponse: String)
        fun onError(error: String)
    }

    /**
     * Downloads and initializes the selected Gemma 4 multimodal model ([modelType]).
     * This is a blocking operation that should be called on a background thread.
     *
     * @param context The Android context
     * @param listener Optional listener for progress and results
     * @return true if initialization succeeded, false otherwise
     */
    fun initialize(
        context: Context,
        listener: MultimodalLLMListener? = null,
    ): Boolean {
        return try {
            if (isInitialized) {
                release()
            }

            val modelFileName = modelFileName()
            val projectorFileName = projectorFileName()
            if (projectorFileName == null) {
                listener?.onError("${modelType.displayName} does not support image or audio input")
                return false
            }

            // Download model file
            Log.i(TAG, "Downloading $modelFileName...")
            val modelFile = downloadModelFile(context, modelFileName, listener)
            if (modelFile == null) {
                listener?.onError("Failed to download model")
                return false
            }
            modelPath = modelFile.absolutePath

            // Download projector (mmproj) file
            Log.i(TAG, "Downloading $projectorFileName...")
            val projectorFile = downloadModelFile(context, projectorFileName, listener)
            if (projectorFile == null) {
                listener?.onError("Failed to download projector")
                return false
            }
            projectorPath = projectorFile.absolutePath

            // Download MTP assistant file
            val mtpFileName = mtpFileName()
            val mtpFile = mtpFileName?.let {
                Log.i(TAG, "Downloading $it...")
                downloadModelFile(context, it, listener)
            }
            if (mtpFileName != null && mtpFile == null) {
                listener?.onError("Failed to download MTP assistant")
                return false
            }

            // Use the built-in sample media (R.raw.person / R.raw.demo) instead of downloading
            Log.i(TAG, "Preparing sample media from resources...")
            val sampleMediaFile = prepareSampleMediaFromResources(context)
            if (sampleMediaFile == null) {
                listener?.onError("Failed to prepare sample ${mediaType.mediaType}")
                return false
            }
            sampleMediaPath = sampleMediaFile.absolutePath
            Log.i(TAG, "Sample media ready: $sampleMediaPath")

            // Create AiliaLLM instance
            Log.i(TAG, "Creating AiliaLLM instance...")
            llm = AiliaLLM()

            // Open model file
            Log.i(TAG, "Opening model file: $modelPath")
            llm!!.openModelFile(modelPath!!, N_CTX)

            // MTP assistant must be opened before the multimodal projector and the first prompt
            isMtpActive = mtpFile != null && LLMMtp.open(llm!!, mtpFile.absolutePath)

            // Open multimodal projector
            Log.i(TAG, "Opening multimodal projector: $projectorPath")
            llm!!.openMultimodalProjectorFile(projectorPath!!)

            // Set default sampling parameters
            llm!!.setSamplingParams(40, 0.9f, 0.4f, 1234)

            // Check multimodal capabilities
            val capabilities = llm!!.getMultimodalCapabilities()
            Log.i(TAG, "Multimodal capabilities - Vision: ${capabilities.visionSupport}, Audio: ${capabilities.audioSupport}")
            val supported = when (mediaType) {
                MultimodalMediaType.IMAGE -> capabilities.visionSupport
                MultimodalMediaType.AUDIO -> capabilities.audioSupport
            }
            if (!supported) {
                listener?.onError("This model does not support ${mediaType.mediaType} input")
                release()
                return false
            }

            // システムプロンプトは設定しない(「説明せよ」などの指示が最初の質問より優先されるため)
            conversationHistory.clear()
            mediaSent = false

            isInitialized = true
            Log.i(TAG, "Multimodal LLM initialized successfully. Context size: ${llm!!.getContextSize()}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize multimodal LLM: ${e.message}", e)
            listener?.onError("Failed to initialize: ${e.message}")
            release()
            false
        }
    }

    /** 使用するテキストモデルのファイル名。 */
    fun modelFileName(): String = modelType.fileName

    /** 使用するmmprojのファイル名。画像・音声に対応しないモデルではnull。 */
    fun projectorFileName(): String? = modelType.mmprojFileName

    /** MTPを使う場合のAssistantのファイル名。MTP無効時や対応しない組み合わせではnull。 */
    fun mtpFileName(): String? = if (mtpEnabled) modelType.mtpFileName else null

    private fun downloadModelFile(
        context: Context,
        fileName: String,
        listener: MultimodalLLMListener?,
    ): File? {
        val downloadListener = object : ModelDownloader.DownloadListener {
            override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                listener?.onDownloadProgress(fileName, bytesDownloaded, totalBytes)
            }

            override fun onComplete(file: File) {
                Log.i(TAG, "Download complete: ${file.absolutePath}")
            }

            override fun onError(error: String) {
                Log.e(TAG, "Download error: $error")
            }
        }
        return ModelDownloader.downloadLLMModel(context, fileName, downloadListener)
    }

    /**
     * Checks if all required files are already downloaded.
     */
    fun areFilesDownloaded(context: Context): Boolean {
        val projectorFileName = projectorFileName() ?: return false
        return listOfNotNull(modelFileName(), projectorFileName, mtpFileName()).all {
            ModelDownloader.isLLMModelDownloaded(context, it)
        }
    }

    /**
     * Generates a response for the given media file and user input.
     * This is a blocking operation that should be called on a background thread.
     *
     * @param mediaPath Path to the image or audio file (if null, uses the built-in sample)
     * @param userInput The user's question about the media
     * @param listener Optional listener for streaming tokens
     * @return The processing time in milliseconds
     */
    fun chatWithMedia(
        mediaPath: String? = null,
        userInput: String,
        listener: MultimodalLLMListener? = null,
    ): Long {
        if (!isInitialized || llm == null) {
            Log.e(TAG, "Multimodal LLM not initialized")
            listener?.onError("Multimodal LLM not initialized")
            return -1
        }

        val mediaToUse = mediaPath ?: sampleMediaPath
        if (mediaToUse == null) {
            Log.e(TAG, "No ${mediaType.mediaType} available")
            listener?.onError("No ${mediaType.mediaType} available")
            return -1
        }

        val historySizeBeforeRequest = conversationHistory.size
        return try {
            cancelRequested.set(false)
            lastPerformance = null
            val startTime = System.nanoTime()

            // メディアを毎ターン履歴に積むとエンコード時間とコンテキストが増えるため、
            // 最初の質問だけメディアを送り、以降はテキストだけを送る。
            // 入力を切り替えた場合は呼び出し側でclearHistory()して送り直す
            val attachMedia = !mediaSent
            val userMessage = if (attachMedia) {
                Log.i(TAG, "chatWithMedia: attaching ${mediaType.mediaType}: $mediaToUse")
                val mediaData = AiliaLLMMediaData(mediaType.mediaType, mediaToUse)
                // The <__media__> placeholder will be replaced with the image or audio
                val content = if (mediaType.mediaBeforeText) {
                    "<__media__>\n$userInput"
                } else {
                    "$userInput\n<__media__>"
                }
                AiliaLLMMultimodalChatMessage("user", content, mediaData)
            } else {
                Log.i(TAG, "chatWithMedia: ${mediaType.mediaType} already sent, sending text only")
                AiliaLLMMultimodalChatMessage("user", userInput)
            }
            conversationHistory.add(userMessage)

            // Set the multimodal prompt (media encoding takes a while on mobile CPUs)
            Log.i(TAG, "chatWithMedia: calling setPrompt with ${conversationHistory.size} messages...")
            if (attachMedia) listener?.onStatus("Processing ${mediaType.mediaType}...")
            val promptStart = System.nanoTime()
            llm!!.setPrompt(conversationHistory.toTypedArray())
            val promptTime = (System.nanoTime() - promptStart) / 1000000
            Log.i(TAG, "chatWithMedia: setPrompt completed in ${promptTime}ms")
            listener?.onStatus(
                if (attachMedia) "Generating... (${mediaType.mediaType} processed in ${promptTime / 1000}s)"
                else "Generating..."
            )

            // Generate response token by token
            val responseBuilder = StringBuilder()
            var done = false
            var tokenCount = 0
            var generationSteps = 0
            var ttftNanos = 0L
            var decodeStartNanos = 0L

            Log.i(TAG, "chatWithMedia: starting generate loop...")
            while (!done && !cancelRequested.get() && generationSteps < MAX_GENERATION_STEPS) {
                done = llm!!.generate()
                if (generationSteps == 0) {
                    // Prefillは最初のgenerate()で実行されるため、ここまでをTTFTとする
                    decodeStartNanos = System.nanoTime()
                    ttftNanos = decodeStartNanos - promptStart
                }
                generationSteps++
                val token = llm!!.getDeltaText()
                if (token.isNotEmpty()) {
                    tokenCount++
                    responseBuilder.append(token)
                    listener?.onToken(token)
                    if (tokenCount <= 5 || tokenCount % 10 == 0) {
                        Log.i(TAG, "chatWithMedia: token[$tokenCount]='$token'")
                    }
                }
            }
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
            val decodeNanos = if (decodeStartNanos > 0) System.nanoTime() - decodeStartNanos else 0L
            Log.i(TAG, "chatWithMedia: generate loop done, total tokens=$tokenCount")

            val fullResponse = responseBuilder.toString()
            lastResult = fullResponse

            // Add assistant response to conversation history
            conversationHistory.add(AiliaLLMMultimodalChatMessage("assistant", fullResponse))
            // 失敗時はユーザー発言ごと履歴から外すため、成功したときだけ送信済みとして記録する
            mediaSent = true

            val endTime = System.nanoTime()
            val processingTime = (endTime - startTime) / 1000000

            val performance = MultimodalLLMPerformance(
                ttftMs = ttftNanos / 1000000,
                // 最初のトークンはTTFTに含まれるため、Decodeのトークン数から除く
                generatedTokens = (generationSteps - 1).coerceAtLeast(0),
                decodeMs = decodeNanos / 1000000,
            )
            lastPerformance = performance

            listener?.onComplete(fullResponse)
            Log.i(TAG, "Multimodal chat completed in ${processingTime}ms. ${performance.summary()}")
            Log.i(TAG, "Response: $fullResponse")

            processingTime
        } catch (e: Exception) {
            while (conversationHistory.size > historySizeBeforeRequest) conversationHistory.removeAt(conversationHistory.lastIndex)
            Log.e(TAG, "Failed to generate response: ${e.message}", e)
            listener?.onError("Failed to generate: ${e.message}")
            -1
        }
    }

    /** 画像入力用のエイリアス。 */
    fun chatWithImage(
        imagePath: String? = null,
        userInput: String,
        listener: MultimodalLLMListener? = null,
    ): Long = chatWithMedia(imagePath, userInput, listener)

    /** Requests the blocking generation loop to stop at the next token boundary. */
    fun cancelGeneration() {
        cancelRequested.set(true)
    }

    /**
     * Gets the path to the built-in sample image or audio.
     */
    fun getSampleMediaPath(): String? {
        return sampleMediaPath
    }

    /**
     * Clears the conversation history and resets the context.
     */
    fun clearHistory() {
        conversationHistory.clear()
        mediaSent = false
        Log.i(TAG, "Conversation history cleared")
    }

    /**
     * Gets the last generated response.
     */
    fun getLastResult(): String {
        return lastResult
    }

    /**
     * Releases the LLM resources.
     */
    fun release() {
        cancelGeneration()
        try {
            llm?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing multimodal LLM: ${e.message}")
        } finally {
            llm = null
            isInitialized = false
            isMtpActive = false
            modelPath = null
            projectorPath = null
            sampleMediaPath = null
            conversationHistory.clear()
            mediaSent = false
            Log.i(TAG, "Multimodal LLM released")
        }
    }

    /**
     * Prepares the built-in sample media from resources.
     * Images are re-encoded to JPEG and audio is copied as-is, because AiliaLLM takes a file path.
     */
    private fun prepareSampleMediaFromResources(context: Context): File? = when (mediaType) {
        MultimodalMediaType.IMAGE -> prepareSampleImage(context)
        MultimodalMediaType.AUDIO -> prepareSampleAudio(context)
    }

    private fun prepareSampleImage(context: Context): File? {
        return try {
            val options = BitmapFactory.Options().apply {
                inScaled = false
            }
            val bitmap = BitmapFactory.decodeResource(context.resources, R.raw.person, options)
            if (bitmap == null) {
                Log.e(TAG, "Failed to decode R.raw.person")
                return null
            }

            val file = File(context.cacheDir, "sample_image.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            bitmap.recycle()

            Log.i(TAG, "Sample image saved to: ${file.absolutePath}")
            file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to prepare sample image: ${e.message}", e)
            null
        }
    }

    private fun prepareSampleAudio(context: Context): File? {
        return try {
            val file = File(context.cacheDir, "sample_audio.wav")
            context.resources.openRawResource(R.raw.demo).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Sample audio saved to: ${file.absolutePath}")
            file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to prepare sample audio: ${e.message}", e)
            null
        }
    }
}
