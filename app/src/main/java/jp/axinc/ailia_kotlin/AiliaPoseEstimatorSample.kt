package jp.axinc.ailia_kotlin

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaEnvironment
import axip.ailia.AiliaImageFormat
import axip.ailia.AiliaModel
import axip.ailia.AiliaPoseEstimatorAlgorithm
import axip.ailia.AiliaPoseEstimatorModel
import axip.ailia.AiliaPoseEstimatorObjectPose
import java.io.File

class AiliaPoseEstimatorSample(private val modelDirectory: File) {
    private var ailia: AiliaModel? = null
    private var poseEstimator: AiliaPoseEstimatorModel? = null
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var ortInputName: String? = null
    private var ortHeatmapsOutputName: String? = null
    private var ortPafsOutputName: String? = null
    private var lastOnnxPoseCount = 0
    private var isInitialized = false

    companion object {
        init { System.loadLibrary("ailia") }
        private const val REMOTE_PATH =
            "https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/"
        private val PROTO_SPEC = ModelFileSpec(
            url = "${REMOTE_PATH}lightweight-human-pose-estimation.opt.onnx.prototxt",
            fileName = "lightweight-human-pose-estimation.opt.onnx.prototxt",
            expectedSize = 92_863,
            sha256 = "bf87750506e3c57e294a7743623e344b29b7de1db07b0246c4c0e5669fa83b1e",
        )
        private val MODEL_SPEC = ModelFileSpec(
            url = "${REMOTE_PATH}lightweight-human-pose-estimation.opt.onnx",
            fileName = "lightweight-human-pose-estimation.opt.onnx",
            expectedSize = 16_347_537,
            sha256 = "4c4c01ae393d67e23658943c2aa6842d84a334a5406ac1dcffccf13395d4692c",
        )

        // Python版(lightweight-human-pose-estimation.py)と同じ閾値。
        // スコアがこれ以下のキーポイント/ラインは描画しない
        private const val KEYPOINT_THRESHOLD = 0.3f
        private const val ORT_INPUT_WIDTH = 320
        private const val ORT_INPUT_HEIGHT = 240

        // Python版 display_result() と同じキーポイントの接続
        private val LINE_PAIRS = arrayOf(
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_NOSE, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_NOSE),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_NOSE),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EAR_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EAR_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_WRIST_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_WRIST_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_BODY_CENTER, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_BODY_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_BODY_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ANKLE_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ANKLE_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_RIGHT),
        )

        /**
         * Python版の hsv_to_rgb(255*point1/KEYPOINT_CNT, 255, 255) と同じ色。
         * OpenCVのHSV(H:0-179)相当の値から色相(0-360°)に変換する。
         */
        private fun lineColor(point1: Int): Int {
            val hCv = 255f * point1 / AiliaPoseEstimatorObjectPose.KEYPOINT_COUNT
            val hueDeg = (hCv % 180f) * 2f
            return Color.HSVToColor(floatArrayOf(hueDeg, 1f, 1f))
        }
    }

    fun ailia_environment(cache_dir: String) : AiliaEnvironment {
        Ailia.SetTemporaryCachePath(cache_dir)
        val envList = AiliaModel.getEnvironments()
        var selectedEnv = envList[0]
        for (env in envList) {
            Log.i(
                "AILIA_Main",
                "Environment " + env.id + ": ( type: " + env.type + ", name: " + env.name + ")"
            )
            if (env.type == AiliaEnvironment.TYPE_GPU && env.props and AiliaEnvironment.PROPERTY_FP16 == 0) {
                selectedEnv = env
                break
            }
        }
        Log.i(
            "AILIA_Main",
            "Selected environment id: " + selectedEnv.id + " (" + selectedEnv.name + ")"
        )
        return selectedEnv;
    }

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            check(ModelDownloader.downloadFile(modelDirectory, PROTO_SPEC, listener) != null)
            check(ModelDownloader.downloadFile(modelDirectory, MODEL_SPEC, listener) != null)
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Pose estimator model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initializePoseEstimator(envId: Int): Boolean {
        return try {
            if (isInitialized) {
                releasePoseEstimator()
            }

            Log.i("AILIA_Main", "Pose estimator: initializing with envId=$envId")

            ailia = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                File(modelDirectory, PROTO_SPEC.fileName).absolutePath,
                File(modelDirectory, MODEL_SPEC.fileName).absolutePath,
            )
            poseEstimator = AiliaPoseEstimatorModel(ailia!!.handle, AiliaPoseEstimatorAlgorithm.LW_HUMAN_POSE)
            isInitialized = true
            Log.i("AILIA_Main", "Pose estimator initialized successfully")
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to initialize pose estimator: ${e.javaClass.name}: ${e.message}")
            releasePoseEstimator()
            false
        }
    }

    fun initializeOnnxRuntime(): Boolean {
        releasePoseEstimator()
        return try {
            val options = OrtSession.SessionOptions()
            try {
                val session = ortEnvironment.createSession(
                    File(modelDirectory, MODEL_SPEC.fileName).absolutePath,
                    options,
                )
                ortSession = session
                ortInputName = session.inputInfo.entries.firstOrNull { (_, info) ->
                    val shape = (info.info as? TensorInfo)?.shape ?: return@firstOrNull false
                    shape.size == 4 && shape[1] == 3L
                }?.key ?: error("Pose image input was not found")
                ortHeatmapsOutputName = session.outputInfo.entries.filter { (_, info) ->
                    val shape = (info.info as? TensorInfo)?.shape ?: return@filter false
                    shape.size == 4 && shape[1] == 19L
                }.lastOrNull()?.key ?: error("Pose heatmap output was not found")
                ortPafsOutputName = session.outputInfo.entries.filter { (_, info) ->
                    val shape = (info.info as? TensorInfo)?.shape ?: return@filter false
                    shape.size == 4 && shape[1] == 38L
                }.lastOrNull()?.key ?: error("Pose PAF output was not found")
            } finally {
                options.close()
            }
            isInitialized = true
            Log.i("AILIA_Main", "Pose estimator initialized with ONNX Runtime CPU")
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to initialize ONNX Runtime pose estimator", e)
            releasePoseEstimator()
            false
        }
    }

    fun processPoseEstimation(img: ByteArray, canvas: Canvas, paint: Paint, w: Int, h: Int): Long {
        ortSession?.let { session ->
            return processOnnxRuntimePose(session, img, canvas, paint, w, h)
        }
        if (!isInitialized || poseEstimator == null) {
            Log.e("AILIA_Error", "Pose estimator not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            
            poseEstimator!!.compute(img, w * 4, w, h, AiliaImageFormat.RGBA)
            val objCount = poseEstimator!!.objectCount
            Log.i("AILIA_Main", "objCount (human count) = $objCount")

            val linePaint = Paint().apply {
                strokeWidth = 5.0f
                isAntiAlias = true
                strokeCap = Paint.Cap.ROUND
            }

            for (objIdx in 0 until objCount) {
                val pose = poseEstimator!!.getObjectPose(objIdx)
                Log.i("AILIA_Main", "person[$objIdx] total score = " + pose.totalScore)

                // キーポイント間のライン(Python版と同じ接続・HSVカラー・閾値0.3)
                for (pair in LINE_PAIRS) {
                    val p1 = pose.points[pair[0]]
                    val p2 = pose.points[pair[1]]
                    if (p1.score > KEYPOINT_THRESHOLD && p2.score > KEYPOINT_THRESHOLD) {
                        linePaint.color = lineColor(pair[0])
                        canvas.drawLine(
                            (w * p1.x).toFloat(), (h * p1.y).toFloat(),
                            (w * p2.x).toFloat(), (h * p2.y).toFloat(),
                            linePaint
                        )
                    }
                }

                // キーポイント(スコアの低いものは非表示)
                for (i in 0 until AiliaPoseEstimatorObjectPose.KEYPOINT_COUNT) {
                    val p = pose.points[i]
                    if (p.score <= KEYPOINT_THRESHOLD) {
                        continue
                    }
                    canvas.drawCircle(
                        (w * p.x).toFloat(),
                        (h * p.y).toFloat(),
                        8.0f,
                        paint
                    )
                }
            }
            if (objCount == 0) {
                Log.i("AILIA_Main", "No object detected.")
            }

            val endTime = System.nanoTime()
            return (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to process pose estimation: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    private fun processOnnxRuntimePose(
        session: OrtSession,
        img: ByteArray,
        canvas: Canvas,
        paint: Paint,
        width: Int,
        height: Int,
    ): Long {
        return try {
            val startedAt = System.nanoTime()
            val input = resizeRgbaToRgbChw(
                img,
                width,
                height,
                ORT_INPUT_WIDTH,
                ORT_INPUT_HEIGHT,
            )
            for (index in input.indices) input[index] = (input[index] - 128f) / 256f
            val tensor = createFloatTensor(
                ortEnvironment,
                input,
                longArrayOf(1, 3, ORT_INPUT_HEIGHT.toLong(), ORT_INPUT_WIDTH.toLong()),
            )
            val poses = try {
                val result = session.run(
                    mapOf(checkNotNull(ortInputName) to tensor),
                    linkedSetOf(
                        checkNotNull(ortHeatmapsOutputName),
                        checkNotNull(ortPafsOutputName),
                    ),
                )
                try {
                    var heatmaps: FloatArray? = null
                    var pafs: FloatArray? = null
                    var mapWidth = 0
                    var mapHeight = 0
                    for (index in 0 until result.size()) {
                        val output = result[index] as OnnxTensor
                        val shape = output.info.shape
                        mapHeight = shape[2].toInt()
                        mapWidth = shape[3].toInt()
                        if (shape[1] == 19L) {
                            heatmaps = readFloatTensor(output)
                        } else if (shape[1] == 38L) {
                            pafs = readFloatTensor(output)
                        }
                    }
                    decodeOrtLightweightPose(
                        checkNotNull(heatmaps),
                        checkNotNull(pafs),
                        mapWidth,
                        mapHeight,
                    )
                } finally {
                    result.close()
                }
            } finally {
                tensor.close()
            }
            lastOnnxPoseCount = poses.size
            drawOrtLightweightPoses(poses, canvas, paint, width, height)
            (System.nanoTime() - startedAt) / 1_000_000
        } catch (e: Exception) {
            Log.e("AILIA_Error", "ONNX Runtime pose estimation failed", e)
            -1
        }
    }

    fun getLastOnnxPoseCount(): Int = lastOnnxPoseCount

    fun releasePoseEstimator() {
        try {
            poseEstimator?.close()
            ailia?.close()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing pose estimator: ${e.javaClass.name}: ${e.message}")
        }
        try {
            ortSession?.close()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing ONNX Runtime pose estimator", e)
        } finally {
            poseEstimator = null
            ailia = null
            ortSession = null
            ortInputName = null
            ortHeatmapsOutputName = null
            ortPafsOutputName = null
            lastOnnxPoseCount = 0
            isInitialized = false
            Log.i("AILIA_Main", "Pose estimator released")
        }
    }
}
