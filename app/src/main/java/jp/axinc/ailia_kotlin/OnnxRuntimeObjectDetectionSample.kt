package jp.axinc.ailia_kotlin

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import java.io.File
import kotlin.math.exp
import kotlin.math.pow

/** ONNX Runtime CPU execution path for YOLOX-S benchmark comparisons. */
class OnnxRuntimeObjectDetectionSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"
        private const val MODEL_URL =
            "https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx"
        private const val MODEL_FILE = "yolox_s.opt.onnx"
        private const val INPUT_SIZE = 640
        private const val OUTPUT_ELEMENTS = 5 + 80
    }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var outputName: String? = null
    private var lastDetectionResults: List<DetectionResult> = emptyList()

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            check(
                ModelDownloader.downloadFile(
                    modelDirectory,
                    ModelFileSpec(MODEL_URL, MODEL_FILE),
                    listener,
                ) != null
            )
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime YOLOX model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initializeObjectDetection(): Boolean {
        releaseObjectDetection()
        return try {
            val options = OrtSession.SessionOptions()
            try {
                val newSession = environment.createSession(
                    File(modelDirectory, MODEL_FILE).absolutePath,
                    options,
                )
                session = newSession
                inputName = findTensorName(newSession.inputInfo, INPUT_SIZE, INPUT_SIZE, 3)
                outputName = newSession.outputInfo.entries.firstOrNull { (_, nodeInfo) ->
                    val shape = (nodeInfo.info as? TensorInfo)?.shape ?: return@firstOrNull false
                    shape.size == 3 && shape.last() == OUTPUT_ELEMENTS.toLong()
                }?.key ?: error("YOLOX output tensor was not found")
            } finally {
                options.close()
            }
            lastDetectionResults = emptyList()
            Log.i(TAG, "ONNX Runtime YOLOX initialized")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX Runtime YOLOX", e)
            releaseObjectDetection()
            false
        }
    }

    fun processObjectDetection(
        img: ByteArray,
        canvas: Canvas,
        paint: Paint,
        text: Paint,
        width: Int,
        height: Int,
        threshold: Float = 0.25f,
        iou: Float = 0.45f,
    ): Long {
        val result = detect(img, width, height, threshold, iou) ?: return -1
        drawDetections(result.value, canvas, paint, text, width, height)
        return result.processingTimeMs
    }

    fun processObjectDetectionWithoutDrawing(
        img: ByteArray,
        width: Int,
        height: Int,
        threshold: Float = 0.25f,
        iou: Float = 0.45f,
    ): Long {
        return detect(img, width, height, threshold, iou)?.processingTimeMs ?: -1
    }

    fun detect(
        img: ByteArray,
        width: Int,
        height: Int,
        threshold: Float = 0.25f,
        iou: Float = 0.45f,
    ): ModelInferenceResult<List<DetectionResult>>? {
        val currentSession = session ?: return null
        val currentInputName = inputName ?: return null
        val currentOutputName = outputName ?: return null
        return try {
            val startTime = System.nanoTime()
            val input = resizeRgbaToRgbChw(img, width, height, INPUT_SIZE, INPUT_SIZE)
            val tensor = createFloatTensor(
                environment,
                input,
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
            )
            val rawOutput = try {
                val result = currentSession.run(
                    mapOf(currentInputName to tensor),
                    setOf(currentOutputName),
                )
                try {
                    readFloatTensor(result[0] as OnnxTensor)
                } finally {
                    result.close()
                }
            } finally {
                tensor.close()
            }
            val detections = decodeOrtYoloxOutput(
                rawOutput,
                INPUT_SIZE,
                INPUT_SIZE,
                CocoLabels.CATEGORY.size,
                threshold,
                iou,
            )
            lastDetectionResults = detections
            ModelInferenceResult(detections, (System.nanoTime() - startTime) / 1_000_000)
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime YOLOX inference failed", e)
            null
        }
    }

    fun getDetectionResults(): List<DetectionResult> = lastDetectionResults

    fun releaseObjectDetection() {
        try {
            session?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release ONNX Runtime YOLOX", e)
        } finally {
            session = null
            inputName = null
            outputName = null
            lastDetectionResults = emptyList()
        }
    }

    private fun drawDetections(
        detections: List<DetectionResult>,
        canvas: Canvas,
        paint: Paint,
        text: Paint,
        width: Int,
        height: Int,
    ) {
        for (detection in detections) {
            val color = CategoryColors.forCategory(detection.category)
            val left = detection.x * width
            val top = detection.y * height
            canvas.drawRect(
                left,
                top,
                (detection.x + detection.width) * width,
                (detection.y + detection.height) * height,
                Paint(paint).apply { this.color = color },
            )
            val label = CocoLabels.CATEGORY.getOrElse(detection.category) {
                "class${detection.category}"
            }
            val labelText = "$label ${String.format(java.util.Locale.ROOT, "%.2f", detection.confidence)}"
            canvas.drawRect(
                left,
                top - text.textSize,
                left + text.measureText(labelText),
                top + text.textSize * 0.2f,
                Paint().apply { style = Paint.Style.FILL; this.color = color },
            )
            canvas.drawText(labelText, left, top, Paint(text).apply { this.color = Color.WHITE })
        }
    }
}

private data class OrtYoloxBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
}

/** Decodes the standard YOLOX [1,8400,85] float output and applies class-agnostic NMS. */
internal fun decodeOrtYoloxOutput(
    output: FloatArray,
    inputWidth: Int,
    inputHeight: Int,
    classCount: Int,
    scoreThreshold: Float,
    iouThreshold: Float,
): List<DetectionResult> {
    require(inputWidth > 0 && inputHeight > 0)
    require(classCount > 0)
    val elements = 5 + classCount
    val gridHeights = intArrayOf(inputHeight / 8, inputHeight / 16, inputHeight / 32)
    val gridWidths = intArrayOf(inputWidth / 8, inputWidth / 16, inputWidth / 32)
    val cellCount = gridHeights.indices.sumOf { gridHeights[it] * gridWidths[it] }
    require(output.size >= cellCount * elements) {
        "YOLOX output is too small: ${output.size}"
    }

    val boxes = mutableListOf<OrtYoloxBox>()
    val scores = mutableListOf<Float>()
    val categories = mutableListOf<Int>()
    var cell = 0
    for (scale in gridHeights.indices) {
        val stride = 2f.pow(3 + scale)
        for (y in 0 until gridHeights[scale]) {
            for (x in 0 until gridWidths[scale]) {
                val offset = cell * elements
                var bestClass = 0
                var bestClassScore = Float.NEGATIVE_INFINITY
                for (category in 0 until classCount) {
                    val value = output[offset + 5 + category]
                    if (value > bestClassScore) {
                        bestClassScore = value
                        bestClass = category
                    }
                }
                val score = output[offset + 4] * bestClassScore
                if (score >= scoreThreshold) {
                    val centerX = (output[offset] + x) * stride
                    val centerY = (output[offset + 1] + y) * stride
                    val boxWidth = exp(output[offset + 2]) * stride + 1f
                    val boxHeight = exp(output[offset + 3]) * stride + 1f
                    boxes += OrtYoloxBox(
                        (centerX - boxWidth / 2f) / inputWidth,
                        (centerY - boxHeight / 2f) / inputHeight,
                        (centerX + boxWidth / 2f) / inputWidth,
                        (centerY + boxHeight / 2f) / inputHeight,
                    )
                    scores += score
                    categories += bestClass
                }
                cell++
            }
        }
    }

    val candidates = scores.indices.sortedByDescending { scores[it] }
    val selected = mutableListOf<Int>()
    for (candidate in candidates) {
        if (selected.none { intersectionOverUnion(boxes[candidate], boxes[it]) > iouThreshold }) {
            selected += candidate
        }
    }
    return selected.map { index ->
        val box = boxes[index]
        DetectionResult(
            category = categories[index],
            confidence = scores[index],
            x = box.left,
            y = box.top,
            width = box.width,
            height = box.height,
        )
    }
}

private fun intersectionOverUnion(first: OrtYoloxBox, second: OrtYoloxBox): Float {
    val width = maxOf(0f, minOf(first.right, second.right) - maxOf(first.left, second.left))
    val height = maxOf(0f, minOf(first.bottom, second.bottom) - maxOf(first.top, second.top))
    val intersection = width * height
    val union = first.width * first.height + second.width * second.height - intersection
    return if (union > 0f) intersection / union else 0f
}
