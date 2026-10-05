package jp.axinc.ailia_kotlin

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import java.io.File
import kotlin.math.exp

/**
 * DETR (DEtection TRansformer, ResNet50)による物体検出サンプル。
 * ailia-models(Python版 object_detection/detr)を移植:
 * ImageNet正規化(RGB, CHW)入力、出力logits/boxesをsoftmax+閾値0.7で検出に変換する。
 */
class AiliaDetrSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"
        init { System.loadLibrary("ailia") }
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/detr/detr-r50-e632da11.onnx"
        private const val MODEL_FILE = "detr-r50-e632da11.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/detr/detr-r50-e632da11.onnx.prototxt"
        private const val PROTO_FILE = "detr-r50-e632da11.onnx.prototxt"

        // 入力サイズ(Python版は短辺800。本サンプルは正方形入力のため800x800)
        private const val INPUT_SIZE = 800
        private const val THRESHOLD = 0.7f

        // ImageNet正規化
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // DETRはCOCOの91クラスID(欠番は'N/A')+末尾にno-objectクラスを持つ
        val DETR_CATEGORY = arrayOf(
            "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
            "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
            "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }

    private var ailia: AiliaModel? = null
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var ortImageInputName: String? = null
    private var ortMaskInputName: String? = null
    private var ortLogitsOutputName: String? = null
    private var ortBoxesOutputName: String? = null
    private var isInitialized = false

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting DETR model download/check...")
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(PROTO_URL, PROTO_FILE), listener) != null)
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(MODEL_URL, MODEL_FILE), listener) != null)
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "DETR model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initialize(envId: Int): Boolean {
        if (isInitialized) {
            release()
        }
        return try {
            ailia = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                File(modelDirectory, PROTO_FILE).absolutePath,
                File(modelDirectory, MODEL_FILE).absolutePath,
            )
            try {
                ailia!!.setInputShapeND(intArrayOf(1, 3, INPUT_SIZE, INPUT_SIZE))
                // DETRのONNXは2番目の入力としてパディングマスク(1,H,W)を持つ
                if (ailia!!.inputBlobCount >= 2) {
                    val maskIdx = ailia!!.getBlobIndexByInputIndex(1)
                    ailia!!.setInputBlobShapeND(intArrayOf(1, INPUT_SIZE, INPUT_SIZE), maskIdx)
                }
            } catch (e: Exception) {
                Log.w(TAG, "DETR: input shape setup failed, using model default shape: ${e.message}")
            }
            isInitialized = true
            Log.i(TAG, "DETR initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize DETR: ${e.javaClass.name}: ${e.message}")
            release()
            false
        }
    }

    fun initializeOnnxRuntime(): Boolean {
        release()
        return try {
            val options = OrtSession.SessionOptions()
            try {
                val session = ortEnvironment.createSession(
                    File(modelDirectory, MODEL_FILE).absolutePath,
                    options,
                )
                ortSession = session
                ortImageInputName = session.inputInfo.entries.firstOrNull { (_, info) ->
                    (info.info as? TensorInfo)?.shape?.size == 4
                }?.key ?: error("DETR image input was not found")
                ortMaskInputName = session.inputInfo.entries.firstOrNull { (_, info) ->
                    (info.info as? TensorInfo)?.shape?.size == 3
                }?.key ?: error("DETR padding mask input was not found")
                ortBoxesOutputName = session.outputInfo.entries.firstOrNull { (_, info) ->
                    (info.info as? TensorInfo)?.shape?.lastOrNull() == 4L
                }?.key ?: error("DETR box output was not found")
                ortLogitsOutputName = session.outputInfo.entries.firstOrNull { (_, info) ->
                    (info.info as? TensorInfo)?.shape?.lastOrNull() != 4L
                }?.key ?: error("DETR logits output was not found")
            } finally {
                options.close()
            }
            isInitialized = true
            Log.i(TAG, "DETR initialized with ONNX Runtime CPU")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize DETR with ONNX Runtime", e)
            release()
            false
        }
    }

    /** Compatibility wrapper for the demo UI. Use [detect] when copying inference code. */
    fun processObjectDetection(bitmap: Bitmap, canvas: Canvas, paint: Paint, text: Paint, w: Int, h: Int): Long {
        val result = detect(bitmap) ?: return -1
        drawDetections(result.value, canvas, paint, text, w, h)
        return result.processingTimeMs
    }

    /** Runs preprocessing, DETR inference and postprocessing without rendering. */
    fun detect(bitmap: Bitmap, threshold: Float = THRESHOLD): ModelInferenceResult<List<DetectionResult>>? {
        val model = ailia
        val session = ortSession
        if (!isInitialized || (model == null && session == null)) {
            Log.e(TAG, "DETR not initialized")
            return null
        }

        return try {
            // 前処理: 800x800リサイズ + ImageNet正規化(RGB, CHW)
            val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
            scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            if (scaled !== bitmap) scaled.recycle()
            val plane = INPUT_SIZE * INPUT_SIZE
            val input = FloatArray(3 * plane)
            for (i in 0 until plane) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF) / 255.0f
                val g = ((p shr 8) and 0xFF) / 255.0f
                val b = (p and 0xFF) / 255.0f
                input[i] = (r - MEAN[0]) / STD[0]
                input[plane + i] = (g - MEAN[1]) / STD[1]
                input[2 * plane + i] = (b - MEAN[2]) / STD[2]
            }

            val startTime = System.nanoTime()
            // 出力はlogits(1,100,92)とboxes(1,100,4)。最終次元で判別する
            var logits: FloatArray? = null
            var boxes: FloatArray? = null
            var numQueries = 0
            var numClasses = 0
            if (session != null) {
                val imageTensor = createFloatTensor(
                    ortEnvironment,
                    input,
                    longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                )
                val maskTensor = createBooleanTensor(
                    ortEnvironment,
                    ByteArray(plane),
                    longArrayOf(1, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                )
                try {
                    val result = session.run(
                        mapOf(
                            checkNotNull(ortImageInputName) to imageTensor,
                            checkNotNull(ortMaskInputName) to maskTensor,
                        ),
                        setOf(checkNotNull(ortLogitsOutputName), checkNotNull(ortBoxesOutputName)),
                    )
                    try {
                        for (index in 0 until result.size()) {
                            val tensor = result[index] as OnnxTensor
                            val shape = tensor.info.shape
                            val data = readFloatTensor(tensor)
                            if (shape.last() == 4L) {
                                boxes = data
                            } else {
                                logits = data
                                numQueries = shape[shape.size - 2].toInt()
                                numClasses = shape.last().toInt()
                            }
                        }
                    } finally {
                        result.close()
                    }
                } finally {
                    closeTensors(imageTensor, maskTensor)
                }
            } else {
                val ailiaModel = checkNotNull(model)
                ailiaModel.setInputBlobData(
                    input,
                    input.size * 4,
                    ailiaModel.getBlobIndexByInputIndex(0),
                )
                if (ailiaModel.inputBlobCount >= 2) {
                    val maskInput = FloatArray(plane)
                    ailiaModel.setInputBlobData(
                        maskInput,
                        maskInput.size * 4,
                        ailiaModel.getBlobIndexByInputIndex(1),
                    )
                }
                ailiaModel.update()
                for (outIdx in 0 until ailiaModel.outputBlobCount) {
                    val blobIndex = ailiaModel.getBlobIndexByOutputIndex(outIdx)
                    val shape = ailiaModel.getBlobShapeND(blobIndex)
                    val size = shape.fold(1) { acc, value -> acc * value }
                    val data = FloatArray(size)
                    ailiaModel.getBlobData(data, size * 4, blobIndex)
                    if (shape.last() == 4) {
                        boxes = data
                    } else {
                        logits = data
                        numQueries = shape[shape.size - 2]
                        numClasses = shape.last()
                    }
                }
            }
            val endTime = System.nanoTime()

            if (logits == null || boxes == null) {
                error("DETR: unexpected outputs")
            }
            ModelInferenceResult(
                decodeDetrOutputs(logits, boxes, numQueries, numClasses, threshold),
                (endTime - startTime) / 1_000_000,
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process DETR: ${e.javaClass.name}: ${e.message}", e)
            null
        }
    }

    fun drawDetections(
        detections: List<DetectionResult>,
        canvas: Canvas,
        paint: Paint,
        text: Paint,
        w: Int,
        h: Int,
    ) {
        for (detection in detections) {
            val x1 = detection.x * w
            val y1 = detection.y * h
            val catColor = CategoryColors.forCategory(detection.category, DETR_CATEGORY.size)
            canvas.drawRect(
                x1,
                y1,
                (detection.x + detection.width) * w,
                (detection.y + detection.height) * h,
                Paint(paint).apply { color = catColor },
            )
            val name = DETR_CATEGORY.getOrElse(detection.category) { "class${detection.category}" }
            val labelText = "$name ${String.format(java.util.Locale.ROOT, "%.2f", detection.confidence)}"
            canvas.drawRect(x1, y1 - text.textSize, x1 + text.measureText(labelText), y1 + text.textSize * 0.2f, Paint().apply { style = Paint.Style.FILL; color = catColor })
            canvas.drawText(labelText, x1, y1, Paint(text).apply { color = android.graphics.Color.WHITE })
        }
    }

    fun release() {
        try {
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing DETR: ${e.message}")
        }
        try {
            ortSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ONNX Runtime DETR: ${e.message}")
        } finally {
            ailia = null
            ortSession = null
            ortImageInputName = null
            ortMaskInputName = null
            ortLogitsOutputName = null
            ortBoxesOutputName = null
            isInitialized = false
            Log.i(TAG, "DETR released")
        }
    }
}

/** Pure DETR postprocessing kept outside the Android UI so it can be copied and unit-tested. */
internal fun decodeDetrOutputs(
    logits: FloatArray,
    boxes: FloatArray,
    numQueries: Int,
    numClasses: Int,
    threshold: Float,
): List<DetectionResult> {
    require(numQueries > 0 && numClasses > 1)
    require(logits.size >= numQueries * numClasses)
    require(boxes.size >= numQueries * 4)
    val detections = mutableListOf<DetectionResult>()
    for (query in 0 until numQueries) {
        val offset = query * numClasses
        var maxLogit = Float.NEGATIVE_INFINITY
        for (category in 0 until numClasses) maxLogit = maxOf(maxLogit, logits[offset + category])
        val probabilities = FloatArray(numClasses)
        var sum = 0.0f
        for (category in 0 until numClasses) {
            probabilities[category] = exp((logits[offset + category] - maxLogit).toDouble()).toFloat()
            sum += probabilities[category]
        }
        if (!sum.isFinite() || sum <= 0f) continue
        var label = 0
        var maxProbability = 0.0f
        for (category in 0 until numClasses - 1) {
            val probability = probabilities[category] / sum
            if (probability > maxProbability) {
                maxProbability = probability
                label = category
            }
        }
        if (maxProbability <= threshold) continue

        val cx = boxes[query * 4]
        val cy = boxes[query * 4 + 1]
        val width = boxes[query * 4 + 2]
        val height = boxes[query * 4 + 3]
        val left = (cx - width / 2).coerceIn(0f, 1f)
        val top = (cy - height / 2).coerceIn(0f, 1f)
        val right = (cx + width / 2).coerceIn(0f, 1f)
        val bottom = (cy + height / 2).coerceIn(0f, 1f)
        if (right <= left || bottom <= top) continue
        detections += DetectionResult(label, maxProbability, left, top, right - left, bottom - top)
    }
    return detections
}
