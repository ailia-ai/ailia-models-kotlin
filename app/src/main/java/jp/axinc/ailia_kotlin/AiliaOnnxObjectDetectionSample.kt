package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Environment
import android.util.Log
import axip.ailia.*
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.EnumSet

class AiliaOnnxObjectDetectionSample {
    companion object {
        private const val TAG = "AILIA_Main"
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx"
        private const val MODEL_FILE = "yolox_s.opt.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx.prototxt"
        private const val PROTO_FILE = "yolox_s.opt.onnx.prototxt"
    }

    interface DownloadListener {
        fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onComplete()
        fun onError(error: String)
    }

    private var ailia: AiliaModel? = null
    private var detector: AiliaDetectorModel? = null
    private var isInitialized = false
    private var lastDetectionResults: List<AiliaTrackerSample.DetectionResult> = emptyList()

    private fun modelDirectory(): String {
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath
    }

    private fun downloadFile(urlStr: String, fileName: String, listener: DownloadListener? = null): Boolean {
        val dir = modelDirectory()
        val path = "$dir/$fileName"
        if (File(path).exists()) return true
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
            downloadFile(PROTO_URL, PROTO_FILE, listener)
            downloadFile(MODEL_URL, MODEL_FILE, listener)
            listener?.onComplete()
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Model Download Failed: $MODEL_FILE", e)
            listener?.onError(e.message ?: "Download failed")
            return false
        }
    }

    fun initializeObjectDetection(envId: Int): Boolean {
        if (isInitialized) {
            releaseObjectDetection()
        }

        return try {
            val dir = modelDirectory()
            val protoPath = "$dir/$PROTO_FILE"
            val modelPath = "$dir/$MODEL_FILE"

            ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)

            detector = AiliaDetectorModel(
                ailia!!.handle,
                AiliaNetworkImageFormat.RGB,
                AiliaNetworkImageChannel.FIRST,
                AiliaNetworkImageRange.UNSIGNED_INT8,
                AiliaDetectorAlgorithm.YOLOX,
                CocoAndImageNetLabels.COCO_CATEGORY.size,
                EnumSet.noneOf(AiliaDetectorFlags::class.java)
            )

            isInitialized = true
            Log.i(TAG, "ONNX Object detection initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX object detection: ${e.javaClass.name}: ${e.message}")
            releaseObjectDetection()
            false
        }
    }

    fun processObjectDetection(img: ByteArray, bitmap: Bitmap, canvas: Canvas, paint: Paint, text: Paint, w: Int, h: Int, threshold: Float = 0.25f, iou: Float = 0.45f): Long {
        if (!isInitialized || detector == null) {
            Log.e(TAG, "ONNX Object detection not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            detector!!.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, threshold, iou)
            val endTime = System.nanoTime()

            val count = detector!!.objectCount
            val detectionResults = mutableListOf<AiliaTrackerSample.DetectionResult>()

            for (i in 0 until count) {
                val obj = detector!!.getObject(i)
                canvas.drawRect(
                    obj.x * w, obj.y * h,
                    (obj.x + obj.w) * w, (obj.y + obj.h) * h,
                    paint
                )
                val label = if (obj.category < CocoAndImageNetLabels.COCO_CATEGORY.size) {
                    CocoAndImageNetLabels.COCO_CATEGORY[obj.category]
                } else {
                    "class${obj.category}"
                }
                canvas.drawText("$label ${String.format("%.2f", obj.prob)}", obj.x * w, obj.y * h, text)

                detectionResults.add(
                    AiliaTrackerSample.DetectionResult(
                        category = obj.category,
                        confidence = obj.prob,
                        x = obj.x,
                        y = obj.y,
                        width = obj.w,
                        height = obj.h
                    )
                )

                Log.i(TAG, "x=${obj.x}, y=${obj.y}, w=${obj.w}, h=${obj.h}, class=[${obj.category}, $label], score=${obj.prob}")
            }

            lastDetectionResults = detectionResults
            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process ONNX object detection: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun processObjectDetectionWithoutDrawing(img: ByteArray, w: Int, h: Int, threshold: Float = 0.25f, iou: Float = 0.45f): Long {
        if (!isInitialized || detector == null) {
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            detector!!.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, threshold, iou)
            val endTime = System.nanoTime()

            val count = detector!!.objectCount
            val detectionResults = mutableListOf<AiliaTrackerSample.DetectionResult>()
            for (i in 0 until count) {
                val obj = detector!!.getObject(i)
                detectionResults.add(
                    AiliaTrackerSample.DetectionResult(
                        category = obj.category,
                        confidence = obj.prob,
                        x = obj.x,
                        y = obj.y,
                        width = obj.w,
                        height = obj.h
                    )
                )
            }
            lastDetectionResults = detectionResults
            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process ONNX object detection: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun getDetectionResults(): List<AiliaTrackerSample.DetectionResult> {
        return lastDetectionResults
    }

    fun releaseObjectDetection() {
        try {
            detector?.close()
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ONNX object detection: ${e.javaClass.name}: ${e.message}")
        } finally {
            detector = null
            ailia = null
            isInitialized = false
            Log.i(TAG, "ONNX Object detection released")
        }
    }
}
