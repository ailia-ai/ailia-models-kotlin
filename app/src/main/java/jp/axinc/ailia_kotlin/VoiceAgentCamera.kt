package jp.axinc.ailia_kotlin

import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.CancellationException

/** One-shot capture with a private metering stream; no visible preview or VLM analysis stream. */
internal class VoiceAgentCamera(private val activity: AppCompatActivity) {
    enum class Facing { BACK, FRONT }

    private val main = Handler(Looper.getMainLooper())
    private val executor = ContextCompat.getMainExecutor(activity)
    private var provider: ProcessCameraProvider? = null
    private var capture: ImageCapture? = null
    private var metering: ImageAnalysis? = null
    private var output: File? = null
    private var completion: ((Result<File>) -> Unit)? = null
    private var generation = 0L
    private var warmupTimeout: Runnable? = null
    private val timeout = Runnable { finish(Result.failure(IllegalStateException("写真撮影がタイムアウトしました"))) }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    @Suppress("DEPRECATION")
    fun takePhoto(facing: Facing = Facing.BACK, onComplete: (Result<File>) -> Unit) {
        cancel()
        val selector = if (facing == Facing.FRONT) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        completion = onComplete
        val token = ++generation
        try {
            val file = File.createTempFile("voice_agent_photo_", ".jpg", activity.cacheDir)
            output = file
            main.postDelayed(timeout, 20_000)
            val future = ProcessCameraProvider.getInstance(activity)
            future.addListener({
                if (token != generation) return@addListener
                try {
                    val cameraProvider = future.get()
                    provider = cameraProvider
                    check(cameraProvider.hasCamera(selector)) {
                        if (facing == Facing.FRONT) "インカメラがありません" else "アウトカメラがありません"
                    }
                    val imageCapture = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .setTargetResolution(Size(640, 480))
                        .setTargetRotation(activity.windowManager.defaultDisplay.rotation)
                        .build()
                    capture = imageCapture
                    var frames = 0
                    var firstFrameAt = 0L
                    var exposureReady = false
                    var whiteBalanceReady = false
                    var shutterStarted = false

                    fun captureWhenReady(force: Boolean = false) {
                        if (token != generation || shutterStarted || frames == 0) return
                        if (!force && (frames < 6 || SystemClock.elapsedRealtime() - firstFrameAt < 300 ||
                                !exposureReady || !whiteBalanceReady)) return
                        shutterStarted = true
                        warmupTimeout?.let { main.removeCallbacks(it) }
                        warmupTimeout = null
                        try {
                            imageCapture.takePicture(ImageCapture.OutputFileOptions.Builder(file).build(), executor,
                                object : ImageCapture.OnImageSavedCallback {
                                    override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                                        if (token == generation) finish(Result.success(file)) else file.delete()
                                    }
                                    override fun onError(error: ImageCaptureException) {
                                        if (token == generation) finish(Result.failure(error)) else file.delete()
                                    }
                                })
                        } catch (error: Exception) {
                            finish(Result.failure(error))
                        }
                    }

                    // ImageCapture alone does not supply repeating frames on all devices.
                    // Keep a private YUV stream running so automatic exposure/white balance can converge.
                    val builder = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    Camera2Interop.Extender(builder).setSessionCaptureCallback(object : CameraCaptureSession.CaptureCallback() {
                        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                            val ae = result.get(CaptureResult.CONTROL_AE_STATE)
                            val awb = result.get(CaptureResult.CONTROL_AWB_STATE)
                            main.post {
                                if (token != generation) return@post
                                exposureReady = ae == null || ae == CaptureResult.CONTROL_AE_STATE_CONVERGED ||
                                    ae == CaptureResult.CONTROL_AE_STATE_LOCKED || ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED
                                whiteBalanceReady = awb == null || awb == CaptureResult.CONTROL_AWB_STATE_CONVERGED ||
                                    awb == CaptureResult.CONTROL_AWB_STATE_LOCKED
                                captureWhenReady()
                            }
                        }
                    })
                    val analysis = builder.build()
                    metering = analysis
                    analysis.setAnalyzer(executor) { image ->
                        try {
                            if (token == generation && !shutterStarted) {
                                if (frames == 0) {
                                    firstFrameAt = SystemClock.elapsedRealtime()
                                    // Some devices never report convergence. After real frames have
                                    // arrived, bound the wait and let quality-mode capture finish 3A.
                                    warmupTimeout = Runnable { captureWhenReady(force = true) }.also { main.postDelayed(it, 2000) }
                                }
                                frames++
                                captureWhenReady()
                            }
                        } finally {
                            image.close()
                        }
                    }
                    cameraProvider.bindToLifecycle(activity, selector, imageCapture, analysis)
                } catch (error: Exception) {
                    finish(Result.failure(error))
                }
            }, executor)
        } catch (error: Exception) {
            finish(Result.failure(error))
        }
    }

    private fun finish(result: Result<File>) {
        generation++
        main.removeCallbacks(timeout)
        warmupTimeout?.let { main.removeCallbacks(it) }
        warmupTimeout = null
        metering?.clearAnalyzer()
        val useCases = listOfNotNull(capture, metering)
        if (useCases.isNotEmpty()) provider?.unbind(*useCases.toTypedArray())
        capture = null
        metering = null
        if (result.isFailure) output?.delete()
        output = null
        val callback = completion
        completion = null
        callback?.invoke(result)
    }

    fun cancel() {
        if (completion != null) finish(Result.failure(CancellationException("撮影を中止しました")))
    }

    fun close() = cancel()
}
