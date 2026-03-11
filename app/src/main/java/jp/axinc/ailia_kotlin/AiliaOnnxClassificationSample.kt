package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.util.Log
import axip.ailia.*
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

class AiliaOnnxClassificationSample {
    companion object {
        private const val TAG = "AILIA_Main"
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.onnx"
        private const val MODEL_FILE = "mobilenetv2_1.0.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.onnx.prototxt"
        private const val PROTO_FILE = "mobilenetv2_1.0.onnx.prototxt"
    }

    interface DownloadListener {
        fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onComplete()
        fun onError(error: String)
    }

    private var ailia: AiliaModel? = null
    private var classifier: AiliaClassifierModel? = null
    private var isInitialized = false
    private var lastClassificationResult: String = ""
    var modelDir: String = ""

    private fun downloadFile(urlStr: String, fileName: String, listener: DownloadListener? = null): Boolean {
        val dir = modelDir
        val path = "$dir/$fileName"
        val file = File(path)
        if (file.exists()) {
            if (file.canRead()) {
                Log.i(TAG, "Model file already exists and readable: $path (${file.length()} bytes)")
                return true
            } else {
                Log.w(TAG, "Model file exists but not readable, re-downloading: $path")
                file.delete()
            }
        }
        File(path).parentFile?.mkdirs()
        val tmpFile = File("$path.tmp")
        val url = URL(urlStr)
        val connection = url.openConnection() as HttpURLConnection
        connection.connectTimeout = 30000
        connection.readTimeout = 60000
        connection.connect()
        val totalBytes = connection.contentLengthLong
        connection.inputStream.use { input ->
            FileOutputStream(tmpFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesDownloaded: Long = 0
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    bytesDownloaded += bytesRead
                    listener?.onProgress(fileName, bytesDownloaded, totalBytes)
                }
            }
        }
        tmpFile.renameTo(File(path))
        return true
    }

    fun downloadModel(listener: DownloadListener? = null): Boolean {
        try {
            Log.i(TAG, "Starting ONNX classification model download/check...")
            downloadFile(PROTO_URL, PROTO_FILE, listener)
            downloadFile(MODEL_URL, MODEL_FILE, listener)
            listener?.onComplete()
            Log.i(TAG, "ONNX classification model download/check complete")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Model Download Failed: $MODEL_FILE", e)
            listener?.onError(e.message ?: "Download failed")
            return false
        }
    }

    fun initializeClassification(envId: Int): Boolean {
        if (isInitialized) {
            releaseClassification()
        }

        return try {
            val dir = modelDir
            val protoPath = "$dir/$PROTO_FILE"
            val modelPath = "$dir/$MODEL_FILE"

            ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)

            classifier = AiliaClassifierModel(
                ailia!!.handle,
                AiliaNetworkImageFormat.RGB,
                AiliaNetworkImageChannel.FIRST,
                AiliaNetworkImageRange.IMAGENET
            )

            isInitialized = true
            Log.i(TAG, "ONNX Classification initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX classification: ${e.javaClass.name}: ${e.message}")
            releaseClassification()
            false
        }
    }

    fun processClassification(img: ByteArray, w: Int, h: Int): Long {
        if (!isInitialized || classifier == null) {
            Log.e(TAG, "ONNX Classification not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            classifier!!.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 5)
            val endTime = System.nanoTime()

            val count = classifier!!.classCount
            if (count > 0) {
                val topClass = classifier!!.getClass(0)
                val label = if (topClass.category < CocoAndImageNetLabels.IMAGENET_CATEGORY.size) {
                    CocoAndImageNetLabels.IMAGENET_CATEGORY[topClass.category]
                } else {
                    "class${topClass.category}"
                }
                lastClassificationResult = "$label (${String.format("%.2f", topClass.prob)})"
                Log.i(TAG, "class ${topClass.category} $label confidence ${topClass.prob}")
            } else {
                lastClassificationResult = "No classification result"
            }

            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process ONNX classification: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun getLastClassificationResult(): String {
        return lastClassificationResult
    }

    fun releaseClassification() {
        try {
            classifier?.close()
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ONNX classification: ${e.javaClass.name}: ${e.message}")
        } finally {
            classifier = null
            ailia = null
            isInitialized = false
            Log.i(TAG, "ONNX Classification released")
        }
    }
}
