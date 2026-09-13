package jp.axinc.ailia_kotlin

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * ONNX Runtime CPU execution path used to benchmark the same ONNX classifiers as ailia SDK.
 * Model selection and the result type are shared with [AiliaOnnxClassificationSample].
 */
class OnnxRuntimeClassificationSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"
        private const val INPUT_SIZE = 224
    }

    var modelType: OnnxClassificationModelType = OnnxClassificationModelType.MOBILENETV2

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var outputName: String? = null
    private var lastClassificationResult = ""

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            check(
                ModelDownloader.downloadFile(
                    modelDirectory,
                    ModelFileSpec(modelType.weightUrl, modelType.weightFile),
                    listener,
                ) != null
            )
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime classification model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initializeClassification(): Boolean {
        releaseClassification()
        return try {
            val options = OrtSession.SessionOptions()
            try {
                val newSession = environment.createSession(
                    File(modelDirectory, modelType.weightFile).absolutePath,
                    options,
                )
                val newInputName = findTensorName(newSession.inputInfo, INPUT_SIZE, INPUT_SIZE, 3)
                val newOutputName = findOutputName(newSession, ImageNetLabels.CATEGORY.size)
                session = newSession
                inputName = newInputName
                outputName = newOutputName
            } finally {
                options.close()
            }
            lastClassificationResult = ""
            Log.i(TAG, "ONNX Runtime classification (${modelType.displayName}) initialized")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX Runtime classification", e)
            releaseClassification()
            false
        }
    }

    fun processClassification(img: ByteArray, width: Int, height: Int): Long =
        classify(img, width, height)?.processingTimeMs ?: -1

    fun classify(
        img: ByteArray,
        width: Int,
        height: Int,
    ): ModelInferenceResult<List<ClassificationResult>>? {
        val currentSession = session ?: return null
        val currentInputName = inputName ?: return null
        val currentOutputName = outputName ?: return null
        return try {
            val startTime = System.nanoTime()
            val input = preprocessOrtClassificationRgba(img, width, height, modelType)
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
            val results = decodeOrtClassifications(rawOutput, modelType)
            lastClassificationResult = formatClassificationResults(results)
            ModelInferenceResult(results, (System.nanoTime() - startTime) / 1_000_000)
        } catch (e: Exception) {
            lastClassificationResult = "Error: ${e.message ?: e.javaClass.simpleName}"
            Log.e(TAG, "ONNX Runtime classification failed", e)
            null
        }
    }

    fun getLastClassificationResult(): String = lastClassificationResult

    fun releaseClassification() {
        try {
            session?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release ONNX Runtime classification", e)
        } finally {
            session = null
            inputName = null
            outputName = null
        }
    }
}

private fun findOutputName(session: OrtSession, classCount: Int): String {
    return session.outputInfo.entries.firstOrNull { (_, nodeInfo) ->
        val shape = (nodeInfo.info as? TensorInfo)?.shape ?: return@firstOrNull false
        shape.isNotEmpty() && shape.last() == classCount.toLong()
    }?.key ?: error("Classification output tensor was not found")
}

internal fun findTensorName(
    info: Map<String, ai.onnxruntime.NodeInfo>,
    height: Int,
    width: Int,
    channels: Int,
): String {
    return info.entries.firstOrNull { (_, nodeInfo) ->
        val shape = (nodeInfo.info as? TensorInfo)?.shape ?: return@firstOrNull false
        shape.size == 4 &&
            shape[1] == channels.toLong() &&
            shape[2] == height.toLong() &&
            shape[3] == width.toLong()
    }?.key ?: error("Expected image input tensor [N,$channels,$height,$width] was not found")
}

internal fun createFloatTensor(
    environment: OrtEnvironment,
    values: FloatArray,
    shape: LongArray,
): OnnxTensor {
    val byteBuffer = ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    val floatBuffer = byteBuffer.asFloatBuffer()
    floatBuffer.put(values)
    floatBuffer.rewind()
    return OnnxTensor.createTensor(environment, floatBuffer, shape)
}

internal fun readFloatTensor(tensor: OnnxTensor): FloatArray {
    val buffer = tensor.floatBuffer
    buffer.rewind()
    return FloatArray(buffer.remaining()).also { buffer.get(it) }
}

/** Bilinear RGBA-to-RGB/CHW preprocessing matching the ailia classifier ranges. */
internal fun preprocessOrtClassificationRgba(
    img: ByteArray,
    width: Int,
    height: Int,
    modelType: OnnxClassificationModelType,
): FloatArray {
    require(width > 0 && height > 0) { "Image dimensions must be positive" }
    require(img.size >= width * height * 4) { "RGBA image buffer is too small" }
    if (modelType == OnnxClassificationModelType.VIT_B16) {
        return preprocessVitRgba(img, width, height)
    }

    val raw = resizeRgbaToRgbChw(img, width, height, 224, 224)
    val plane = 224 * 224
    if (modelType == OnnxClassificationModelType.MOBILENETV2) {
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (channel in 0 until 3) {
            val offset = channel * plane
            for (i in 0 until plane) {
                raw[offset + i] = (raw[offset + i] / 255f - mean[channel]) / std[channel]
            }
        }
    } else {
        for (i in raw.indices) raw[i] -= 128f
    }
    return raw
}

internal fun resizeRgbaToRgbChw(
    img: ByteArray,
    sourceWidth: Int,
    sourceHeight: Int,
    targetWidth: Int,
    targetHeight: Int,
): FloatArray {
    require(sourceWidth > 0 && sourceHeight > 0)
    require(targetWidth > 0 && targetHeight > 0)
    require(img.size >= sourceWidth * sourceHeight * 4)
    val plane = targetWidth * targetHeight
    val output = FloatArray(3 * plane)
    for (targetY in 0 until targetHeight) {
        val sourceY = ((targetY + 0.5f) * sourceHeight / targetHeight - 0.5f)
            .coerceIn(0f, sourceHeight - 1f)
        val y0 = sourceY.toInt()
        val y1 = minOf(y0 + 1, sourceHeight - 1)
        val yWeight = sourceY - y0
        for (targetX in 0 until targetWidth) {
            val sourceX = ((targetX + 0.5f) * sourceWidth / targetWidth - 0.5f)
                .coerceIn(0f, sourceWidth - 1f)
            val x0 = sourceX.toInt()
            val x1 = minOf(x0 + 1, sourceWidth - 1)
            val xWeight = sourceX - x0
            val targetOffset = targetY * targetWidth + targetX
            for (channel in 0 until 3) {
                fun component(x: Int, y: Int): Float =
                    (img[(y * sourceWidth + x) * 4 + channel].toInt() and 0xFF).toFloat()

                val top = component(x0, y0) * (1f - xWeight) + component(x1, y0) * xWeight
                val bottom = component(x0, y1) * (1f - xWeight) + component(x1, y1) * xWeight
                output[channel * plane + targetOffset] = top * (1f - yWeight) + bottom * yWeight
            }
        }
    }
    return output
}

internal fun decodeOrtClassifications(
    output: FloatArray,
    modelType: OnnxClassificationModelType,
): List<ClassificationResult> {
    require(output.size == ImageNetLabels.CATEGORY.size) {
        "Unexpected classification output count: ${output.size}"
    }
    if (modelType != OnnxClassificationModelType.RESNET50) {
        return decodeVitClassifications(output)
    }
    return output.indices
        .sortedWith(compareByDescending<Int> { output[it] }.thenBy { it })
        .take(CLASSIFICATION_TOP_COUNT)
        .map { category ->
            ClassificationResult(category, ImageNetLabels.CATEGORY[category], output[category])
        }
}
