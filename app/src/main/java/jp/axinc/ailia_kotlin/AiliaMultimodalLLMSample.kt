package jp.axinc.ailia_kotlin

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import ai.ailia.llm.AiliaLLM
import ai.ailia.llm.AiliaLLMMediaData
import ai.ailia.llm.AiliaLLMMultimodalChatMessage
import java.io.File
import java.io.FileOutputStream

/**
 * Sample class demonstrating ailia Multimodal LLM inference for image understanding.
 */
class AiliaMultimodalLLMSample {
    private var llm: AiliaLLM? = null
    private var isInitialized = false
    private var lastResult: String = ""
    private var modelPath: String? = null
    private var projectorPath: String? = null
    private var sampleImagePath: String? = null
    private val conversationHistory = mutableListOf<AiliaLLMMultimodalChatMessage>()

    companion object {
        private const val TAG = "AiliaMultimodalLLM"
        private const val N_CTX = 8192 // Context window size
    }

    interface MultimodalLLMListener {
        fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onToken(token: String)
        fun onComplete(fullResponse: String)
        fun onError(error: String)
    }

    /**
     * Downloads and initializes the Gemma 3 multimodal LLM model.
     * This is a blocking operation that should be called on a background thread.
     *
     * @param context The Android context
     * @param listener Optional listener for progress and results
     * @return true if initialization succeeded, false otherwise
     */
    fun initialize(
        context: Context,
        listener: MultimodalLLMListener? = null
    ): Boolean {
        return try {
            if (isInitialized) {
                release()
            }

            // Download model file
            Log.i(TAG, "Downloading Gemma 3 model...")
            val modelFile = ModelDownloader.downloadGemma3Model(context, object : ModelDownloader.DownloadListener {
                override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                    listener?.onDownloadProgress("gemma-3-4b-it-Q4_K_M.gguf", bytesDownloaded, totalBytes)
                }
                override fun onComplete(file: java.io.File) {
                    Log.i(TAG, "Model download complete: ${file.absolutePath}")
                }
                override fun onError(error: String) {
                    Log.e(TAG, "Model download error: $error")
                }
            })
            if (modelFile == null) {
                listener?.onError("Failed to download model")
                return false
            }
            modelPath = modelFile.absolutePath

            // Download projector file
            Log.i(TAG, "Downloading Gemma 3 projector...")
            val projectorFile = ModelDownloader.downloadGemma3Projector(context, object : ModelDownloader.DownloadListener {
                override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                    listener?.onDownloadProgress("gemma-3-4b-it-GGUF_mmproj-model-f16.gguf", bytesDownloaded, totalBytes)
                }
                override fun onComplete(file: java.io.File) {
                    Log.i(TAG, "Projector download complete: ${file.absolutePath}")
                }
                override fun onError(error: String) {
                    Log.e(TAG, "Projector download error: $error")
                }
            })
            if (projectorFile == null) {
                listener?.onError("Failed to download projector")
                return false
            }
            projectorPath = projectorFile.absolutePath

            // Use built-in sample image (R.raw.person) instead of downloading
            Log.i(TAG, "Preparing sample image from resources...")
            val sampleImageFile = prepareSampleImageFromResources(context)
            if (sampleImageFile == null) {
                listener?.onError("Failed to prepare sample image")
                return false
            }
            sampleImagePath = sampleImageFile.absolutePath
            Log.i(TAG, "Sample image ready: $sampleImagePath")

            // Create AiliaLLM instance
            Log.i(TAG, "Creating AiliaLLM instance...")
            llm = AiliaLLM()

            // Open model file
            Log.i(TAG, "Opening model file: $modelPath")
            llm!!.openModelFile(modelPath!!, N_CTX)

            // Open multimodal projector
            Log.i(TAG, "Opening multimodal projector: $projectorPath")
            llm!!.openMultimodalProjectorFile(projectorPath!!)

            // Set default sampling parameters
            llm!!.setSamplingParams(40, 0.9f, 0.4f, 1234)

            // Check multimodal capabilities
            val capabilities = llm!!.getMultimodalCapabilities()
            Log.i(TAG, "Multimodal capabilities - Vision: ${capabilities.visionSupport}, Audio: ${capabilities.audioSupport}")

            // Add system prompt
            conversationHistory.clear()
            conversationHistory.add(AiliaLLMMultimodalChatMessage("system", "You are a helpful assistant that can understand images. Describe images briefly and concisely."))

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

    /**
     * Checks if all required files are already downloaded.
     */
    fun areFilesDownloaded(context: Context): Boolean {
        return ModelDownloader.isGemma3ModelDownloaded(context) &&
               ModelDownloader.isGemma3ProjectorDownloaded(context)
    }

    /**
     * Generates a response for the given image and user input.
     * This is a blocking operation that should be called on a background thread.
     *
     * @param imagePath Path to the image file (if null, uses the sample image)
     * @param userInput The user's question about the image
     * @param listener Optional listener for streaming tokens
     * @return The processing time in milliseconds
     */
    fun chatWithImage(
        imagePath: String? = null,
        userInput: String,
        listener: MultimodalLLMListener? = null
    ): Long {
        if (!isInitialized || llm == null) {
            Log.e(TAG, "Multimodal LLM not initialized")
            listener?.onError("Multimodal LLM not initialized")
            return -1
        }

        val imageToUse = imagePath ?: sampleImagePath
        if (imageToUse == null) {
            Log.e(TAG, "No image available")
            listener?.onError("No image available")
            return -1
        }

        return try {
            val startTime = System.nanoTime()

            // Create media data for the image
            val mediaData = AiliaLLMMediaData("image", imageToUse)

            // Create user message with image placeholder
            // The <__media__> placeholder will be replaced with the image
            val messageContent = "<__media__>\n$userInput"
            val userMessage = AiliaLLMMultimodalChatMessage("user", messageContent, mediaData)
            conversationHistory.add(userMessage)

            // Set the multimodal prompt
            llm!!.setMultimodalPrompt(conversationHistory.toTypedArray())

            // Generate response token by token
            val responseBuilder = StringBuilder()
            var done = false

            while (!done) {
                done = llm!!.generate()
                val token = llm!!.getDeltaText()
                if (token.isNotEmpty()) {
                    responseBuilder.append(token)
                    listener?.onToken(token)
                }
            }

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
            Log.e(TAG, "Failed to generate response: ${e.message}", e)
            listener?.onError("Failed to generate: ${e.message}")
            -1
        }
    }

    /**
     * Gets the path to the sample image.
     */
    fun getSampleImagePath(): String? {
        return sampleImagePath
    }

    /**
     * Clears the conversation history and resets the context.
     */
    fun clearHistory() {
        conversationHistory.clear()
        conversationHistory.add(AiliaLLMMultimodalChatMessage("system", "You are a helpful assistant that can understand images."))
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
        try {
            llm?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing multimodal LLM: ${e.message}")
        } finally {
            llm = null
            isInitialized = false
            modelPath = null
            projectorPath = null
            sampleImagePath = null
            conversationHistory.clear()
            Log.i(TAG, "Multimodal LLM released")
        }
    }

    /**
     * Prepares the sample image from R.raw.person resource.
     * Saves the bitmap to a temporary file for use with AiliaLLM.
     */
    private fun prepareSampleImageFromResources(context: Context): File? {
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
}
