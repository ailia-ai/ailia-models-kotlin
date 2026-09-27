package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.Locale
import java.util.Properties

/**
 * A model file and the information required to validate it.
 *
 * [expectedSize] and [sha256] are optional because not all hosted models expose
 * immutable release metadata. Files downloaded by this class still receive a
 * sidecar containing the server length and computed SHA-256, so subsequent
 * launches can detect local corruption without accessing the network.
 */
data class ModelFileSpec(
    val url: String,
    val fileName: String,
    val expectedSize: Long? = null,
    val sha256: String? = null,
)

/** Robust, shared downloader used by every sample in this application. */
object ModelDownloader {
    private const val TAG = "ModelDownloader"
    private const val BASE_URL = "https://storage.googleapis.com/ailia-models"
    private const val BUFFER_SIZE = 64 * 1024

    /** 進捗通知の間隔。数GBのモデルで通知がログ/UIを圧迫しないように間引く。 */
    private const val PROGRESS_STEP_BYTES = 4L * 1024 * 1024

    /** 変換済みQNNモデル(.qnn)の配置先。ailia LLMのバージョンごとにディレクトリが分かれる。 */
    private const val QNN_LLM_PATH = "gemma/qnn/v1.5.0"

    const val GEMMA_2_MODEL_URL = "$BASE_URL/gemma/gemma-2-2b-it-Q4_K_M.gguf"
    const val SAMPLE_IMAGE_URL = "$BASE_URL/misc/sample_image.jpg"

    interface DownloadListener {
        fun onProgress(bytesDownloaded: Long, totalBytes: Long)
        fun onComplete(file: File)
        fun onError(error: String)
    }

    /** Durable app-specific storage. Android does not evict it as a cache entry. */
    fun modelDirectory(context: Context): File = (context.getExternalFilesDir(null) ?: context.filesDir).apply {
        check(exists() || mkdirs()) { "Failed to create model directory: $absolutePath" }
    }

    fun downloadLLMModel(
        context: Context,
        fileName: String,
        listener: DownloadListener? = null,
    ): File? {
        val directory = modelDirectory(context)
        migrateLegacyCache(context, directory, fileName)
        return downloadFile(
            directory,
            ModelFileSpec("$BASE_URL/gemma/$fileName", fileName),
            listener,
        )
    }

    /** SoC固有のQNNモデル(.qnn)をダウンロードする。 */
    fun downloadQnnLLMModel(
        context: Context,
        fileName: String,
        listener: DownloadListener? = null,
    ): File? = downloadFile(modelDirectory(context), qnnLLMModelSpec(fileName), listener)

    fun isQnnLLMModelDownloaded(context: Context, fileName: String): Boolean =
        isDownloaded(modelDirectory(context), qnnLLMModelSpec(fileName))

    private fun qnnLLMModelSpec(fileName: String) =
        ModelFileSpec("$BASE_URL/$QNN_LLM_PATH/$fileName", fileName)

    fun downloadFile(
        directory: File,
        spec: ModelFileSpec,
        listener: DownloadListener? = null,
    ): File? {
        val file = File(directory, spec.fileName)
        return try {
            check(directory.exists() || directory.mkdirs()) {
                "Failed to create model directory: ${directory.absolutePath}"
            }
            val parent = file.parentFile ?: directory
            check(parent.exists() || parent.mkdirs()) {
                "Failed to create model subdirectory: ${parent.absolutePath}"
            }

            if (isValid(file, spec)) {
                Log.i(TAG, "Using validated model: ${file.absolutePath}")
                listener?.onComplete(file)
                return file
            }
            deleteInvalidArtifacts(file, keepTempFile = true)

            // 中断された一時ファイルが残っていればRangeリクエストで続きから再開する
            val tempFile = File(directory, "${spec.fileName}.download")
            var resumeOffset = if (tempFile.isFile) tempFile.length() else 0L
            if (spec.expectedSize != null && resumeOffset > spec.expectedSize) {
                check(tempFile.delete()) { "Failed to remove oversized temporary file: ${tempFile.absolutePath}" }
                resumeOffset = 0L
            }

            val connection = (URL(spec.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 120_000
                requestMethod = "GET"
                instanceFollowRedirects = true
                if (resumeOffset > 0) {
                    setRequestProperty("Range", "bytes=$resumeOffset-")
                }
                connect()
            }
            try {
                val resumed = resumeOffset > 0 &&
                    connection.responseCode == HttpURLConnection.HTTP_PARTIAL
                if (connection.responseCode != HttpURLConnection.HTTP_OK && !resumed) {
                    // 再開リクエストが拒否された場合(416など)は一時ファイルを破棄して次回最初から取り直す
                    if (resumeOffset > 0 && tempFile.exists() && !tempFile.delete()) {
                        Log.w(TAG, "Failed to delete ${tempFile.absolutePath}")
                    }
                    error("HTTP ${connection.responseCode} for ${spec.url}")
                }
                if (!resumed) {
                    // サーバーがRangeに対応しない場合などは最初からダウンロードし直す
                    if (tempFile.exists() && !tempFile.delete()) {
                        error("Failed to remove stale temporary file: ${tempFile.absolutePath}")
                    }
                    resumeOffset = 0L
                } else {
                    Log.i(TAG, "Resuming download of ${spec.fileName} from $resumeOffset bytes")
                }
                val responseSize = connection.getHeaderField("Content-Length")?.toLongOrNull()
                    ?.takeIf { it >= 0 }
                val requiredSize = spec.expectedSize ?: responseSize?.plus(resumeOffset)

                var downloaded = resumeOffset
                var lastReported = -1L
                connection.inputStream.use { input ->
                    FileOutputStream(tempFile, resumed).use { output ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            if (count == 0) continue
                            output.write(buffer, 0, count)
                            downloaded += count
                            if (downloaded - lastReported >= PROGRESS_STEP_BYTES) {
                                listener?.onProgress(downloaded, requiredSize ?: -1L)
                                lastReported = downloaded
                            }
                        }
                        output.fd.sync()
                    }
                }
                listener?.onProgress(downloaded, requiredSize ?: -1L)

                check(downloaded > 0) { "Downloaded file is empty: ${spec.fileName}" }
                check(requiredSize == null || downloaded == requiredSize) {
                    "Size mismatch for ${spec.fileName}: expected=$requiredSize actual=$downloaded"
                }
                val actualSha256 = sha256(tempFile)
                check(spec.sha256 == null || actualSha256.equals(spec.sha256, ignoreCase = true)) {
                    "SHA-256 mismatch for ${spec.fileName}"
                }

                moveChecked(tempFile, file)
                writeMetadata(file, spec.url, downloaded, actualSha256)
                check(isValid(file, spec)) { "Downloaded model validation failed: ${spec.fileName}" }
                listener?.onComplete(file)
                file
            } finally {
                connection.disconnect()
            }
        } catch (e: Exception) {
            // 通信断などによる失敗時は一時ファイルを残し、次回のダウンロードで再開する
            deleteInvalidArtifacts(file, keepTempFile = true)
            val error = "Download failed for ${spec.fileName}: ${e.message}"
            Log.e(TAG, error, e)
            listener?.onError(error)
            null
        }
    }

    fun downloadFile(
        directory: File,
        spec: ModelFileSpec,
        listener: ModelDownloadListener?,
    ): File? = downloadFile(directory, spec, object : DownloadListener {
        override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
            listener?.onProgress(spec.fileName, bytesDownloaded, totalBytes)
        }

        override fun onComplete(file: File) = Unit

        override fun onError(error: String) {
            listener?.onError(error)
        }
    })

    fun isDownloaded(directory: File, spec: ModelFileSpec): Boolean = isValid(File(directory, spec.fileName), spec)

    fun isLLMModelDownloaded(context: Context, fileName: String): Boolean = isDownloaded(
        modelDirectory(context),
        ModelFileSpec("$BASE_URL/gemma/$fileName", fileName),
    )

    fun downloadGemma2Model(context: Context, listener: DownloadListener? = null): File? =
        downloadLLMModel(context, "gemma-2-2b-it-Q4_K_M.gguf", listener)

    fun downloadSampleImage(context: Context, listener: DownloadListener? = null): File? = downloadFile(
        modelDirectory(context),
        ModelFileSpec(SAMPLE_IMAGE_URL, "sample_image.jpg"),
        listener,
    )

    fun isGemma2ModelDownloaded(context: Context): Boolean =
        isLLMModelDownloaded(context, "gemma-2-2b-it-Q4_K_M.gguf")

    fun isSampleImageDownloaded(context: Context): Boolean =
        isDownloaded(modelDirectory(context), ModelFileSpec(SAMPLE_IMAGE_URL, "sample_image.jpg"))

    fun getGemma2ModelPath(context: Context): String =
        File(modelDirectory(context), "gemma-2-2b-it-Q4_K_M.gguf").absolutePath

    fun getSampleImagePath(context: Context): String =
        File(modelDirectory(context), "sample_image.jpg").absolutePath

    private fun isValid(file: File, spec: ModelFileSpec): Boolean {
        if (!file.isFile || !file.canRead() || file.length() <= 0) return false
        if (spec.expectedSize != null && file.length() != spec.expectedSize) return false

        val metadataFile = metadataFile(file)
        if (!metadataFile.isFile) {
            // Preserve valid models downloaded by older application versions. New downloads
            // always have metadata and receive full hash validation below.
            return spec.sha256 == null || sha256(file).equals(spec.sha256, ignoreCase = true)
        }
        val metadata = Properties().apply {
            FileInputStream(metadataFile).use(::load)
        }
        val recordedLength = metadata.getProperty("length")?.toLongOrNull() ?: return false
        val recordedSha256 = metadata.getProperty("sha256") ?: return false
        if (metadata.getProperty("url") != spec.url || recordedLength != file.length()) return false
        if (spec.sha256 != null && !recordedSha256.equals(spec.sha256, ignoreCase = true)) return false
        return sha256(file).equals(recordedSha256, ignoreCase = true)
    }

    private fun writeMetadata(file: File, url: String, length: Long, sha256: String) {
        val temp = File(file.parentFile, "${file.name}.metadata.download")
        val properties = Properties().apply {
            setProperty("url", url)
            setProperty("length", length.toString())
            setProperty("sha256", sha256)
        }
        FileOutputStream(temp).use { output ->
            properties.store(output, "ailia model download metadata")
            output.fd.sync()
        }
        moveChecked(temp, metadataFile(file))
    }

    private fun moveChecked(source: File, destination: File) {
        if (destination.exists() && !destination.delete()) {
            error("Failed to replace ${destination.absolutePath}")
        }
        check(source.renameTo(destination)) {
            "Failed to move ${source.absolutePath} to ${destination.absolutePath}"
        }
    }

    private fun deleteInvalidArtifacts(file: File, keepTempFile: Boolean = false) {
        val artifacts = mutableListOf(
            file,
            metadataFile(file),
            File(file.parentFile, "${file.name}.metadata.download"),
        )
        if (!keepTempFile) {
            artifacts.add(File(file.parentFile, "${file.name}.download"))
        }
        artifacts
            .filter(File::exists)
            .forEach { artifact ->
                if (!artifact.delete()) Log.w(TAG, "Failed to delete ${artifact.absolutePath}")
            }
    }

    private fun metadataFile(file: File): File = File(file.parentFile, "${file.name}.metadata")

    private fun migrateLegacyCache(context: Context, directory: File, fileName: String) {
        val destination = File(directory, fileName)
        val legacy = File(context.cacheDir, fileName)
        if (destination.exists() || !legacy.isFile || legacy.length() <= 0) return
        try {
            if (!legacy.renameTo(destination)) {
                legacy.copyTo(destination, overwrite = false)
                check(legacy.delete()) { "Failed to remove migrated cache file" }
            }
            Log.i(TAG, "Migrated model from cache: ${destination.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "Could not migrate legacy cache model $fileName", e)
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                if (count > 0) digest.update(buffer, 0, count)
            }
        }
        return digest.digest().toHex()
    }

    private fun ByteArray.toHex(): String = joinToString("") {
        String.format(Locale.ROOT, "%02x", it.toInt() and 0xFF)
    }
}
