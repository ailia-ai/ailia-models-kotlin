package jp.axinc.ailia_kotlin

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import java.io.File

/**
 * U-2-Netによる背景除去サンプル(ailia-models-flutterのu2net.dartを移植)。
 * 320x320にリサイズしてImageNet正規化した画像を入力し、
 * 出力マスクを黒の半透明オーバーレイとして元画像に重ねる。
 */
class AiliaU2NetSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"
        init { System.loadLibrary("ailia") }
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/u2net/u2net_opset11.onnx"
        private const val MODEL_FILE = "u2net_opset11.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/u2net/u2net_opset11.onnx.prototxt"
        private const val PROTO_FILE = "u2net_opset11.onnx.prototxt"
        private const val IMAGE_SIZE = 320

        // ImageNet正規化(Flutter版imageToAiliaTensorと同じ)
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private var ailia: AiliaModel? = null
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var ortInputName: String? = null
    private var ortOutputName: String? = null
    private var isInitialized = false

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting U2Net model download/check...")
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(PROTO_URL, PROTO_FILE), listener) != null)
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(MODEL_URL, MODEL_FILE), listener) != null)
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "U2Net model download failed", e)
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
            isInitialized = true
            Log.i(TAG, "U2Net initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize U2Net: ${e.javaClass.name}: ${e.message}")
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
                ortInputName = session.inputNames.first()
                ortOutputName = session.outputNames.first()
            } finally {
                options.close()
            }
            isInitialized = true
            Log.i(TAG, "U2Net initialized with ONNX Runtime CPU")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize U2Net with ONNX Runtime", e)
            release()
            false
        }
    }

    /**
     * 背景除去を実行し、背景部分を黒の半透明オーバーレイとしてcanvasに描画する。
     * canvasには元画像が描画済みであること。
     */
    fun process(bitmap: Bitmap, canvas: Canvas, w: Int, h: Int): Long {
        val result = predictMask(bitmap) ?: return -1
        drawMask(result.value, canvas, w, h)
        return result.processingTimeMs
    }

    /** Returns the foreground mask without rendering it into an Android Canvas. */
    fun predictMask(bitmap: Bitmap): ModelInferenceResult<SegmentationMask>? {
        val model = ailia
        val session = ortSession
        if (!isInitialized || (model == null && session == null)) {
            Log.e(TAG, "U2Net not initialized")
            return null
        }

        return try {
            // 前処理: 320x320リサイズ + ImageNet正規化(RGB, CHW)
            val scaled = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
            scaled.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
            if (scaled !== bitmap) scaled.recycle()
            val plane = IMAGE_SIZE * IMAGE_SIZE
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
            val mask = if (session != null) {
                val tensor = createFloatTensor(
                    ortEnvironment,
                    input,
                    longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()),
                )
                try {
                    runFloatTensor(
                        session,
                        mapOf(checkNotNull(ortInputName) to tensor),
                        checkNotNull(ortOutputName),
                    ).copyOf(plane)
                } finally {
                    tensor.close()
                }
            } else {
                checkNotNull(model)
                model.setInputBlobData(input, input.size * 4, model.getBlobIndexByInputIndex(0))
                model.update()
                val outIdx = model.getBlobIndexByOutputIndex(0)
                FloatArray(plane).also {
                    model.getBlobData(it, it.size * 4, outIdx)
                }
            }
            val endTime = System.nanoTime()

            for (i in mask.indices) mask[i] = mask[i].coerceIn(0f, 1f)
            ModelInferenceResult(SegmentationMask(IMAGE_SIZE, IMAGE_SIZE, mask), (endTime - startTime) / 1_000_000)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process U2Net: ${e.javaClass.name}: ${e.message}", e)
            null
        }
    }

    fun drawMask(mask: SegmentationMask, canvas: Canvas, w: Int, h: Int) {
        val overlayPixels = IntArray(mask.values.size)
        for (i in mask.values.indices) {
            overlayPixels[i] = Color.argb(((1.0f - mask.values[i]) * 255).toInt(), 0, 0, 0)
        }
        val overlay = Bitmap.createBitmap(overlayPixels, mask.width, mask.height, Bitmap.Config.ARGB_8888)
        canvas.drawBitmap(overlay, null, Rect(0, 0, w, h), null)
    }

    fun release() {
        try {
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing U2Net: ${e.message}")
        }
        try {
            ortSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ONNX Runtime U2Net: ${e.message}")
        } finally {
            ailia = null
            ortSession = null
            ortInputName = null
            ortOutputName = null
            isInitialized = false
            Log.i(TAG, "U2Net released")
        }
    }
}
