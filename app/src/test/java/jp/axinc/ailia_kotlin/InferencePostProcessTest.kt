package jp.axinc.ailia_kotlin

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InferencePostProcessTest {
    @Test
    fun detrThresholdUsesBestObjectClassProbability() {
        // The combined object probability is above 0.7, but neither object class is.
        // DETR's threshold must be applied to the best individual class.
        val detections = decodeDetrOutputs(
            logits = floatArrayOf(0.4f, 0.3f, 0.0f),
            boxes = floatArrayOf(0.5f, 0.5f, 0.4f, 0.2f),
            numQueries = 1,
            numClasses = 3,
            threshold = 0.7f,
        )

        assertTrue(detections.isEmpty())
    }

    @Test
    fun detrDecodesConfidentCenterBoxToNormalizedCorners() {
        val detection = decodeDetrOutputs(
            logits = floatArrayOf(0.0f, 5.0f, -2.0f),
            boxes = floatArrayOf(0.5f, 0.5f, 0.4f, 0.2f),
            numQueries = 1,
            numClasses = 3,
            threshold = 0.7f,
        ).single()

        assertEquals(1, detection.category)
        assertTrue(detection.confidence > 0.98f)
        assertEquals(0.3f, detection.x, 0.0001f)
        assertEquals(0.4f, detection.y, 0.0001f)
        assertEquals(0.4f, detection.width, 0.0001f)
        assertEquals(0.2f, detection.height, 0.0001f)
    }

    @Test
    fun quantizedClassificationHandlesSignedInt8() {
        val results = decodeQuantizedClassifications(
            outputShape = intArrayOf(1, 3),
            outputBuffer = byteArrayOf((-128).toByte(), 10, 127),
            signed = true,
            quantScale = 0.01f,
            quantZeroPoint = 0,
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(listOf(2, 1, 0), results.map { it.category })
        assertEquals(listOf("third", "second", "first"), results.map { it.label })
        assertEquals(1.27f, results[0].confidence, 0.0001f)
        assertEquals(0.10f, results[1].confidence, 0.0001f)
        assertEquals(-1.28f, results[2].confidence, 0.0001f)
    }

    @Test
    fun quantizedClassificationHandlesUnsignedInt8() {
        val results = decodeQuantizedClassifications(
            outputShape = intArrayOf(3),
            outputBuffer = byteArrayOf(1, 0xFE.toByte(), 3),
            signed = false,
            quantScale = 0.5f,
            quantZeroPoint = 2,
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(listOf(1, 2, 0), results.map { it.category })
        assertEquals(126.0f, results[0].confidence, 0.0001f)
        assertEquals(0.5f, results[1].confidence, 0.0001f)
        assertEquals(-0.5f, results[2].confidence, 0.0001f)
    }

    @Test
    fun classificationInputQuantizationPreservesSignedRange() {
        val scale = 2f / 255f

        assertEquals((-128).toByte(), quantizeClassificationInput(-1f, scale, -1, signed = true))
        assertEquals((-1).toByte(), quantizeClassificationInput(0f, scale, -1, signed = true))
        assertEquals(126.toByte(), quantizeClassificationInput(1f, scale, -1, signed = true))
    }

    @Test
    fun vitPreprocessingConvertsRgbaToNormalizedRgbChw() {
        val input = preprocessVitRgba(
            img = byteArrayOf(0, 127, 0xFF.toByte(), 0xFF.toByte()),
            width = 1,
            height = 1,
        )
        val planeSize = 224 * 224

        assertEquals(3 * planeSize, input.size)
        assertEquals(-1f, input[0], 0.0001f)
        assertEquals(127f / 127.5f - 1f, input[planeSize], 0.0001f)
        assertEquals(1f, input[2 * planeSize], 0.0001f)
        assertEquals(1f, input.last(), 0.0001f)
    }

    @Test
    fun vitClassificationAppliesSoftmaxToLogits() {
        val results = decodeVitClassifications(
            logits = floatArrayOf(1f, 3f, 2f),
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(listOf(1, 2, 0), results.map { it.category })
        assertEquals(listOf("second", "third", "first"), results.map { it.label })
        assertEquals(0.66524f, results[0].confidence, 0.0001f)
        assertEquals(0.24473f, results[1].confidence, 0.0001f)
        assertEquals(0.09003f, results[2].confidence, 0.0001f)
        assertEquals(
            "1. second (0.67)\n2. third (0.24)\n3. first (0.09)",
            formatClassificationResults(results),
        )
    }

    @Test
    fun onnxRuntimeResnetPreprocessingUsesSignedRgbRange() {
        val input = preprocessOrtClassificationRgba(
            img = byteArrayOf(0, 128.toByte(), 0xFF.toByte(), 0xFF.toByte()),
            width = 1,
            height = 1,
            modelType = OnnxClassificationModelType.RESNET50,
        )
        val plane = 224 * 224

        assertEquals(-128f, input[0], 0.0001f)
        assertEquals(0f, input[plane], 0.0001f)
        assertEquals(127f, input[2 * plane], 0.0001f)
    }

    @Test
    fun onnxRuntimeYoloxDecodesFloatOutput() {
        val classCount = 2
        val cells = 4 * 4 + 2 * 2 + 1
        val output = FloatArray(cells * (5 + classCount))
        output[0] = 0.5f
        output[1] = 0.5f
        output[2] = 0f
        output[3] = 0f
        output[4] = 0.9f
        output[5] = 0.8f
        output[6] = 0.1f

        val detection = decodeOrtYoloxOutput(
            output = output,
            inputWidth = 32,
            inputHeight = 32,
            classCount = classCount,
            scoreThreshold = 0.25f,
            iouThreshold = 0.45f,
        ).single()

        assertEquals(0, detection.category)
        assertEquals(0.72f, detection.confidence, 0.0001f)
        assertEquals(9f / 32f, detection.width, 0.0001f)
        assertEquals(9f / 32f, detection.height, 0.0001f)
    }

    @Test
    fun onnxRuntimePoseConnectsHeatmapPeaksWithPafs() {
        val width = 5
        val height = 5
        val plane = width * height
        val heatmaps = FloatArray(19 * plane)
        fun setPeak(type: Int, x: Int, y: Int) {
            heatmaps[type * plane + y * width + x] = 0.9f
        }
        setPeak(type = 1, x = 1, y = 2) // neck
        setPeak(type = 2, x = 2, y = 2) // right shoulder
        setPeak(type = 3, x = 3, y = 2) // right elbow

        val pafs = FloatArray(38 * plane)
        for (offset in 0 until plane) {
            pafs[12 * plane + offset] = 1f // neck -> right shoulder
            pafs[14 * plane + offset] = 1f // right shoulder -> right elbow
        }

        val pose = decodeOrtLightweightPose(heatmaps, pafs, width, height).single()

        assertEquals(3, pose.points.count { it != null })
        assertEquals(0.25f, pose.points[1]!!.x, 0.0001f)
        assertEquals(0.50f, pose.points[1]!!.y, 0.0001f)
        assertEquals(0.75f, pose.points[3]!!.x, 0.0001f)
    }

    @Test
    fun downloadProgressDisplaysMegabytes() {
        assertEquals(
            "Downloading model.onnx 1.0 / 2.0 MB",
            formatDownloadProgress("model.onnx", 1024L * 1024L, 2L * 1024L * 1024L),
        )
        assertEquals(
            "Downloading model.onnx 1.5 MB",
            formatDownloadProgress("model.onnx", 1536L * 1024L, -1L),
        )
        assertEquals(
            "Downloading 0.0 MB",
            formatDownloadProgress(null, 0L, -1L),
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun segmentationMaskRejectsMismatchedDimensions() {
        SegmentationMask(2, 2, FloatArray(3))
    }
}
