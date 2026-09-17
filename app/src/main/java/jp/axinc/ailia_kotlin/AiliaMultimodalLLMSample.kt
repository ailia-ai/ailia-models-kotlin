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
 */
enum class MultimodalMediaType(val mediaType: String, val systemPrompt: String) {
    IMAGE(
        "image",
        "You are a helpful assistant that can understand images. Describe images briefly and concisely.",
    ),
    AUDIO(
        "audio",
        "You are a helpful assistant that can understand audio. Describe audio briefly and concisely.",
    ),
}

/**
 * Sample class demonstrating ailia Multimodal LLM inference (VLM / ALM).
 *
 * Gemma 4 E2Bをテキストモデルとして使い、mmprojで画像または音声を入力する。
 * バックエンドにQNNを選ぶと、SoC固有の変換済みモデル(.qnn)をNPUで実行する。
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
    private val cancelRequested = AtomicBoolean(false)

    /** 実行バックエンド。QNNは対応SoCの場合のみ選択できる。 */
    var backend: LLMBackend = LLMBackend.CPU

    companion object {
        private const val TAG = "AiliaMultimodalLLM"
        private const val N_CTX = 8192 // Context window size
        // QNNモデルはコンテキスト長が変換時に固定されるため、0を指定してモデル内の値を使う
        private const val N_CTX_QNN = 0
        private const val MAX_GENERATION_STEPS = 4096

        /** CPU(GGUF)で使用するGemma 4 E2Bのテキストモデル。 */
        const val GGUF_MODEL_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"

        /** CPU(GGUF)で使用するGemma 4 E2Bの画像/音声エンコーダ(mmproj)。 */
        const val GGUF_PROJECTOR_FILE = "gemma-4-E2B-it-mmproj-F16.gguf"

        /** QNNモデルの対象となるLLMモデル。 */
        private val QNN_MODEL_TYPE = LLMModelType.GEMMA_4_E2B
    }

    interface MultimodalLLMListener {
        fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onStatus(status: String)
        fun onToken(token: String)
        fun onComplete(fullResponse: String)
        fun onError(error: String)
    }

    /**
     * Downloads and initializes the Gemma 4 E2B multimodal model.
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

            val useQnn = backend == LLMBackend.QNN
            val modelFileName = modelFileName()
            val projectorFileName = projectorFileName()
            if (useQnn && (modelFileName == null || projectorFileName == null)) {
                listener?.onError("QNN model is not available on this SoC")
                return false
            }

            // Download model file
            Log.i(TAG, "Downloading $modelFileName for ${backend.displayName}...")
            val modelFile = downloadModelFile(context, modelFileName!!, useQnn, listener)
            if (modelFile == null) {
                listener?.onError("Failed to download model")
                return false
            }
            modelPath = modelFile.absolutePath

            // Download projector (mmproj) file
            Log.i(TAG, "Downloading $projectorFileName for ${backend.displayName}...")
            val projectorFile = downloadModelFile(context, projectorFileName!!, useQnn, listener)
            if (projectorFile == null) {
                listener?.onError("Failed to download projector")
                return false
            }
            projectorPath = projectorFile.absolutePath

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
            llm!!.openModelFile(modelPath!!, if (useQnn) N_CTX_QNN else N_CTX)

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

            // Add system prompt
            conversationHistory.clear()
            conversationHistory.add(AiliaLLMMultimodalChatMessage("system", mediaType.systemPrompt))

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

    /** 現在のバックエンドで使用するテキストモデルのファイル名。QNN未対応SoCではnull。 */
    fun modelFileName(): String? = when (backend) {
        LLMBackend.CPU -> GGUF_MODEL_FILE
        LLMBackend.QNN -> QnnSupport.llmQnnFileName(QNN_MODEL_TYPE)
    }

    /** 現在のバックエンドで使用するmmprojのファイル名。QNN未対応SoCではnull。 */
    fun projectorFileName(): String? = when (backend) {
        LLMBackend.CPU -> GGUF_PROJECTOR_FILE
        LLMBackend.QNN -> QnnSupport.llmQnnMmprojFileName(QNN_MODEL_TYPE)
    }

    private fun downloadModelFile(
        context: Context,
        fileName: String,
        useQnn: Boolean,
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
        return if (useQnn) {
            ModelDownloader.downloadQnnLLMModel(context, fileName, downloadListener)
        } else {
            ModelDownloader.downloadLLMModel(context, fileName, downloadListener)
        }
    }

    /**
     * Checks if all required files are already downloaded.
     */
    fun areFilesDownloaded(context: Context): Boolean {
        val modelFileName = modelFileName() ?: return false
        val projectorFileName = projectorFileName() ?: return false
        return if (backend == LLMBackend.QNN) {
            ModelDownloader.isQnnLLMModelDownloaded(context, modelFileName) &&
                ModelDownloader.isQnnLLMModelDownloaded(context, projectorFileName)
        } else {
            ModelDownloader.isLLMModelDownloaded(context, modelFileName) &&
                ModelDownloader.isLLMModelDownloaded(context, projectorFileName)
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
            val startTime = System.nanoTime()

            // Create media data for the image or audio
            Log.i(TAG, "chatWithMedia: creating media data for: $mediaToUse")
            val mediaData = AiliaLLMMediaData(mediaType.mediaType, mediaToUse)

            // Create user message with media placeholder
            // The <__media__> placeholder will be replaced with the image or audio
            val messageContent = "<__media__>\n$userInput"
            val userMessage = AiliaLLMMultimodalChatMessage("user", messageContent, mediaData)
            conversationHistory.add(userMessage)

            // Set the multimodal prompt (media encoding takes a while on mobile CPUs)
            Log.i(TAG, "chatWithMedia: calling setPrompt with ${conversationHistory.size} messages...")
            listener?.onStatus("Processing ${mediaType.mediaType}...")
            val promptStart = System.nanoTime()
            llm!!.setPrompt(conversationHistory.toTypedArray())
            val promptTime = (System.nanoTime() - promptStart) / 1000000
            Log.i(TAG, "chatWithMedia: setPrompt completed in ${promptTime}ms")
            listener?.onStatus("Generating... (${mediaType.mediaType} processed in ${promptTime / 1000}s)")

            // Generate response token by token
            val responseBuilder = StringBuilder()
            var done = false
            var tokenCount = 0
            var generationSteps = 0

            Log.i(TAG, "chatWithMedia: starting generate loop...")
            while (!done && !cancelRequested.get() && generationSteps < MAX_GENERATION_STEPS) {
                done = llm!!.generate()
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
            Log.i(TAG, "chatWithMedia: generate loop done, total tokens=$tokenCount")

            val fullResponse = responseBuilder.toString()
            lastResult = fullResponse

            // Add assistant response to conversation history
            conversationHistory.add(AiliaLLMMultimodalChatMessage("assistant", fullResponse))

            val endTime = System.nanoTime()
            val processingTime = (endTime - startTime) / 1000000

            listener?.onComplete(fullResponse)
            Log.i(TAG, "Multimodal chat completed in ${processingTime}ms. Response: $fullResponse")

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
        conversationHistory.add(AiliaLLMMultimodalChatMessage("system", mediaType.systemPrompt))
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
            modelPath = null
            projectorPath = null
            sampleMediaPath = null
            conversationHistory.clear()
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
