package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Utility class for downloading model files from Google Cloud Storage.
 */
object ModelDownloader {
    private const val TAG = "ModelDownloader"
    private const val BASE_URL = "https://storage.googleapis.com/ailia-models"

    // LLM Model URLs
    const val GEMMA_2_MODEL_URL = "$BASE_URL/gemma-2-2b-it-Q4_K_M.gguf"
    const val GEMMA_3_MODEL_URL = "$BASE_URL/gemma-3-4b-it-Q4_K_M.gguf"
    const val GEMMA_3_MMPROJ_URL = "$BASE_URL/gemma-3-4b-it-GGUF_mmproj-model-f16.gguf"

    // Sample image for multimodal demo
    const val SAMPLE_IMAGE_URL = "$BASE_URL/misc/sample_image.jpg"

    interface DownloadListener {
        fun onProgress(bytesDownloaded: Long, totalBytes: Long)
        fun onComplete(file: File)
        fun onError(error: String)
    }

    /**
     * Downloads a file from the given URL to the app's cache directory.
     * If the file already exists and has the correct size, it skips the download.
     *
     * @param context The Android context
     * @param url The URL to download from
     * @param fileName The name of the file to save as
     * @param listener Optional listener for download progress
     * @return The downloaded file, or null if download failed
     */
    fun downloadFile(
        context: Context,
        url: String,
        fileName: String,
        listener: DownloadListener? = null
    ): File? {
        val cacheDir = context.cacheDir
        val file = File(cacheDir, fileName)

        // Check if file already exists
        if (file.exists()) {
            Log.i(TAG, "File already exists: ${file.absolutePath}")
            listener?.onComplete(file)
            return file
        }

        return try {
            Log.i(TAG, "Downloading: $url")

            val urlConnection = URL(url).openConnection() as HttpURLConnection
            urlConnection.connectTimeout = 30000
            urlConnection.readTimeout = 60000
            urlConnection.requestMethod = "GET"
            urlConnection.connect()

            if (urlConnection.responseCode != HttpURLConnection.HTTP_OK) {
                val error = "HTTP error: ${urlConnection.responseCode}"
                Log.e(TAG, error)
                listener?.onError(error)
                return null
            }

            val totalBytes = urlConnection.contentLengthLong
            var bytesDownloaded: Long = 0

            val tempFile = File(cacheDir, "$fileName.tmp")

            urlConnection.inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        bytesDownloaded += bytesRead
                        listener?.onProgress(bytesDownloaded, totalBytes)
                    }
                }
            }

            // Rename temp file to final file
            tempFile.renameTo(file)

            Log.i(TAG, "Download complete: ${file.absolutePath}")
            listener?.onComplete(file)
            file

        } catch (e: Exception) {
            val error = "Download failed: ${e.message}"
            Log.e(TAG, error, e)
            listener?.onError(error)
            null
        }
    }

    /**
     * Downloads the Gemma 2 LLM model for text inference.
     */
    fun downloadGemma2Model(context: Context, listener: DownloadListener? = null): File? {
        return downloadFile(context, GEMMA_2_MODEL_URL, "gemma-2-2b-it-Q4_K_M.gguf", listener)
    }

    /**
     * Downloads the Gemma 3 LLM model for multimodal inference.
     */
    fun downloadGemma3Model(context: Context, listener: DownloadListener? = null): File? {
        return downloadFile(context, GEMMA_3_MODEL_URL, "gemma-3-4b-it-Q4_K_M.gguf", listener)
    }

    /**
     * Downloads the Gemma 3 multimodal projector.
     */
    fun downloadGemma3Projector(context: Context, listener: DownloadListener? = null): File? {
        return downloadFile(context, GEMMA_3_MMPROJ_URL, "gemma-3-4b-it-GGUF_mmproj-model-f16.gguf", listener)
    }

    /**
     * Downloads the sample image for multimodal demo.
     */
    fun downloadSampleImage(context: Context, listener: DownloadListener? = null): File? {
        return downloadFile(context, SAMPLE_IMAGE_URL, "sample_image.jpg", listener)
    }

    /**
     * Checks if the Gemma 2 model is already downloaded.
     */
    fun isGemma2ModelDownloaded(context: Context): Boolean {
        return File(context.cacheDir, "gemma-2-2b-it-Q4_K_M.gguf").exists()
    }

    /**
     * Checks if the Gemma 3 model is already downloaded.
     */
    fun isGemma3ModelDownloaded(context: Context): Boolean {
        return File(context.cacheDir, "gemma-3-4b-it-Q4_K_M.gguf").exists()
    }

    /**
     * Checks if the Gemma 3 projector is already downloaded.
     */
    fun isGemma3ProjectorDownloaded(context: Context): Boolean {
        return File(context.cacheDir, "gemma-3-4b-it-GGUF_mmproj-model-f16.gguf").exists()
    }

    /**
     * Checks if the sample image is already downloaded.
     */
    fun isSampleImageDownloaded(context: Context): Boolean {
        return File(context.cacheDir, "sample_image.jpg").exists()
    }

    /**
     * Gets the path to the Gemma 2 model file.
     */
    fun getGemma2ModelPath(context: Context): String {
        return File(context.cacheDir, "gemma-2-2b-it-Q4_K_M.gguf").absolutePath
    }

    /**
     * Gets the path to the Gemma 3 model file.
     */
    fun getGemma3ModelPath(context: Context): String {
        return File(context.cacheDir, "gemma-3-4b-it-Q4_K_M.gguf").absolutePath
    }

    /**
     * Gets the path to the Gemma 3 projector file.
     */
    fun getGemma3ProjectorPath(context: Context): String {
        return File(context.cacheDir, "gemma-3-4b-it-GGUF_mmproj-model-f16.gguf").absolutePath
    }

    /**
     * Gets the path to the sample image file.
     */
    fun getSampleImagePath(context: Context): String {
        return File(context.cacheDir, "sample_image.jpg").absolutePath
    }
}
