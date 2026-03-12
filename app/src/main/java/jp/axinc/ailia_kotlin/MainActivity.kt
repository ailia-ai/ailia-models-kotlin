package jp.axinc.ailia_kotlin

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.Options
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import axip.ailia.*
import axip.ailia_tflite.*
import axip.ailia_llm.AiliaLLM
import java.io.*
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageView: ImageView
    private lateinit var cameraPreviewView: PreviewView
    private lateinit var modeRadioGroup: RadioGroup
    private lateinit var algorithmSpinner: Spinner
    private lateinit var envSpinner: Spinner
    private lateinit var envLabel: TextView
    private lateinit var runtimeSpinner: Spinner
    private lateinit var runtimeLabel: TextView
    private lateinit var processingTimeTextView: TextView
    private lateinit var resultScrollView: ScrollView
    private lateinit var classificationResultTextView: TextView
    private lateinit var tokenizerInputEditText: EditText
    private lateinit var tokenizerOutputTextView: TextView
    private lateinit var trackingResultTextView: TextView
    private lateinit var voiceModelSpinner: Spinner
    private lateinit var voiceStatusTextView: TextView
    private lateinit var voiceGenerateButton: Button
    private lateinit var voiceResultTextView: TextView
    private lateinit var llmInputLabel: TextView
    private lateinit var llmInputEditText: EditText
    private lateinit var llmSendButton: Button
    private lateinit var llmOutputLabel: TextView
    private lateinit var llmOutputTextView: TextView
    private lateinit var llmStatusTextView: TextView
    private lateinit var multimodalImageView: ImageView
    private lateinit var speechModelLabel: TextView
    private lateinit var speechModelSpinner: Spinner
    private lateinit var speechModeRadioGroup: RadioGroup
    private lateinit var diarizationCheckBox: CheckBox
    private lateinit var speechRunButton: Button
    private lateinit var micRecordButton: Button

    private var poseEstimatorSample = AiliaPoseEstimatorSample()
    private var objectDetectionSample = AiliaTFLiteObjectDetectionSample()
    private var classificationSample = AiliaTFLiteClassificationSample()
    private var tokenizerSample = AiliaTokenizerSample()
    private var trackerSample = AiliaTrackerSample()
    private var speechSample = AiliaSpeechSample()
    private var voiceSample = AiliaVoiceSample()
    private var llmSample = AiliaLLMSample()
    private var multimodalLLMSample = AiliaMultimodalLLMSample()
    private var onnxObjectDetectionSample = AiliaOnnxObjectDetectionSample()
    private var onnxClassificationSample = AiliaOnnxClassificationSample()

    private var selectedEnvId: Int = 0
    private var selectedRuntime: String = "TFLite"
    private var ailiaEnvironments: List<AiliaEnvironment>? = null
    private var isInitialized = false
    private var currentAlgorithm = AlgorithmType.POSE_ESTIMATION
    private var pendingAlgorithmSwitch: AlgorithmType? = null
    private var pendingModeSwitch: Int? = null
    private var isProcessing = AtomicBoolean(false)
    private var isWaitAlgorithmSwitch = AtomicBoolean(false)
    private var isWaitModeSwitch = AtomicBoolean(false)
    private var isStopCamera = AtomicBoolean(false)
    private var isDownloadingModel = AtomicBoolean(false)

    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var latestCameraBitmap: Bitmap? = null

    private var selectedVoiceModelType: VoiceModelType = VoiceModelType.GPT_SOVITS_V1
    private var selectedSpeechModelType: SpeechModelType = SpeechModelType.WHISPER_TINY
    private var audioRecord: AudioRecord? = null
    private var isRecording = AtomicBoolean(false)

    enum class AlgorithmType {
        POSE_ESTIMATION,
        OBJECT_DETECTION,
        TRACKING,
        TOKENIZE,
        CLASSIFICATION,
        SPEECH_TO_TEXT,
        TEXT_TO_SPEECH,
        LLM,
        MULTIMODAL_LLM,
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

        init {
            System.loadLibrary("ailia")
            System.loadLibrary("ailia_llm")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // cameraExecutorはsetupModeSelection()より先に初期化する必要がある
        // (SpinnerのonItemSelectedでinitializeAilia()が呼ばれるため)
        cameraExecutor = Executors.newSingleThreadExecutor()

        val modelDir = (getExternalFilesDir(null) ?: filesDir).absolutePath
        onnxObjectDetectionSample.modelDir = modelDir
        onnxClassificationSample.modelDir = modelDir
        speechSample.modelDir = modelDir
        voiceSample.modelDir = modelDir

        initializeViews()
        setupModeSelection()
        updateUIVisibility()

        if (allPermissionsGranted()) {
            initializeAilia()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        cameraPreviewView = findViewById(R.id.cameraPreviewView)
        modeRadioGroup = findViewById(R.id.modeRadioGroup)
        algorithmSpinner = findViewById(R.id.algorithmSpinner)
        envSpinner = findViewById(R.id.envSpinner)
        envLabel = findViewById(R.id.envLabel)
        runtimeSpinner = findViewById(R.id.runtimeSpinner)
        runtimeLabel = findViewById(R.id.runtimeLabel)
        processingTimeTextView = findViewById(R.id.processingTimeTextView)
        resultScrollView = findViewById(R.id.resultScrollView)
        classificationResultTextView = findViewById(R.id.classificationResultTextView)
        tokenizerInputEditText = findViewById(R.id.tokenizerInputEditText)
        tokenizerOutputTextView = findViewById(R.id.tokenizerOutputTextView)
        trackingResultTextView = findViewById(R.id.trackingResultTextView)
        voiceModelSpinner = findViewById(R.id.voiceModelSpinner)
        voiceStatusTextView = findViewById(R.id.voiceStatusTextView)
        voiceGenerateButton = findViewById(R.id.voiceGenerateButton)
        voiceResultTextView = findViewById(R.id.voiceResultTextView)
        llmInputLabel = findViewById(R.id.llmInputLabel)
        llmInputEditText = findViewById(R.id.llmInputEditText)
        llmSendButton = findViewById(R.id.llmSendButton)
        llmOutputLabel = findViewById(R.id.llmOutputLabel)
        llmOutputTextView = findViewById(R.id.llmOutputTextView)
        llmStatusTextView = findViewById(R.id.llmStatusTextView)
        multimodalImageView = findViewById(R.id.multimodalImageView)
        speechModelLabel = findViewById(R.id.speechModelLabel)
        speechModelSpinner = findViewById(R.id.speechModelSpinner)
        speechModeRadioGroup = findViewById(R.id.speechModeRadioGroup)
        diarizationCheckBox = findViewById(R.id.diarizationCheckBox)
        speechRunButton = findViewById(R.id.speechRunButton)
        micRecordButton = findViewById(R.id.micRecordButton)
    }

    private fun setupModeSelection() {
        val algorithms = arrayOf(
            "PoseEstimation",
            "ObjectDetection",
            "Tracking",
            "Tokenize",
            "Classification",
            "Speech2Text",
            "Text2Speech",
            "LLM",
            "MultimodalLLM",
        )

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, algorithms)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        algorithmSpinner.adapter = adapter

        algorithmSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val newAlgorithm = AlgorithmType.values()[position]
                updateRuntimeSpinner(newAlgorithm)
                updateEnvSpinner(newAlgorithm)
                if (newAlgorithm != currentAlgorithm) {
                    switchAlgorithm(newAlgorithm)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        modeRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.imageRadioButton -> {
                    switchToImageMode()
                }

                R.id.cameraRadioButton -> {
                    switchToCameraMode()
                }
            }
        }

        switchToImageMode()
    }

    private fun setupOnnxEnvSpinner(useBlas: Boolean) {
        try {
            if (ailiaEnvironments == null) {
                Ailia.SetTemporaryCachePath(cacheDir.absolutePath)
                ailiaEnvironments = AiliaModel.getEnvironments()
            }
            val envNames = ailiaEnvironments!!.map { "${it.name} (id:${it.id})" }.toTypedArray()
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, envNames)
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            envSpinner.adapter = adapter

            var defaultIndex = 0
            if (useBlas) {
                // デフォルトはBLAS (CPU-OpenBlas)
                for ((index, env) in ailiaEnvironments!!.withIndex()) {
                    if (env.name.contains("OpenBlas", ignoreCase = true)) {
                        defaultIndex = index
                        break
                    }
                }
            } else {
                // デフォルトはGPU
                for ((index, env) in ailiaEnvironments!!.withIndex()) {
                    if (env.type == AiliaEnvironment.TYPE_GPU && env.props and AiliaEnvironment.PROPERTY_FP16 == 0) {
                        defaultIndex = index
                        break
                    }
                }
            }
            envSpinner.setSelection(defaultIndex)
            selectedEnvId = ailiaEnvironments!![defaultIndex].id

            envSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    val newEnvId = ailiaEnvironments!![position].id
                    if (newEnvId != selectedEnvId) {
                        selectedEnvId = newEnvId
                        isInitialized = false
                        isDownloadingModel.set(false)
                        val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
                        if (isImageMode) {
                            releaseCurrentAlgorithm()
                            processImageMode()
                        } else {
                            // Camera mode: cameraExecutorでリリース→再初期化
                            cameraExecutor.execute {
                                releaseCurrentAlgorithm()
                            }
                        }
                    }
                }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }

            envLabel.visibility = View.VISIBLE
            envSpinner.visibility = View.VISIBLE
        } catch (e: Exception) {
            Log.e("AILIA_Main", "Failed to get ailia environments: ${e.message}")
            envLabel.visibility = View.GONE
            envSpinner.visibility = View.GONE
        }
    }

    private fun updateEnvSpinner(algorithm: AlgorithmType) {
        when (algorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                setupOnnxEnvSpinner(useBlas = false)
            }

            AlgorithmType.SPEECH_TO_TEXT, AlgorithmType.TEXT_TO_SPEECH -> {
                setupOnnxEnvSpinner(useBlas = true)
            }

            AlgorithmType.OBJECT_DETECTION, AlgorithmType.CLASSIFICATION, AlgorithmType.TRACKING -> {
                if (selectedRuntime == "ONNX") {
                    setupOnnxEnvSpinner(useBlas = true)
                } else {
                    // TFLite: Reference (CPU) と NNAPI を表示
                    val tfliteEnvNames = arrayOf("Reference (CPU)", "NNAPI")
                    val tfliteEnvIds = intArrayOf(AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE, AiliaTFLite.AILIA_TFLITE_ENV_NNAPI)
                    val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, tfliteEnvNames)
                    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                    envSpinner.adapter = adapter

                    // デフォルトは NNAPI
                    envSpinner.setSelection(1)
                    selectedEnvId = AiliaTFLite.AILIA_TFLITE_ENV_NNAPI

                    envSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                            val newEnvId = tfliteEnvIds[position]
                            if (newEnvId != selectedEnvId) {
                                selectedEnvId = newEnvId
                                isInitialized = false
                                isDownloadingModel.set(false)
                                val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
                                if (isImageMode) {
                                    releaseCurrentAlgorithm()
                                    processImageMode()
                                } else {
                                    cameraExecutor.execute {
                                        releaseCurrentAlgorithm()
                                    }
                                }
                            }
                        }
                        override fun onNothingSelected(parent: AdapterView<*>?) {}
                    }

                    envLabel.visibility = View.VISIBLE
                    envSpinner.visibility = View.VISIBLE
                }
            }

            else -> {
                envLabel.visibility = View.GONE
                envSpinner.visibility = View.GONE
            }
        }
    }

    private fun updateRuntimeSpinner(algorithm: AlgorithmType) {
        when (algorithm) {
            AlgorithmType.OBJECT_DETECTION, AlgorithmType.CLASSIFICATION, AlgorithmType.TRACKING -> {
                val runtimes = arrayOf("TFLite", "ONNX")
                val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, runtimes)
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                runtimeSpinner.adapter = adapter

                val defaultIndex = if (selectedRuntime == "ONNX") 1 else 0
                runtimeSpinner.setSelection(defaultIndex)

                runtimeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                        val newRuntime = runtimes[position]
                        if (newRuntime != selectedRuntime) {
                            selectedRuntime = newRuntime
                            updateEnvSpinner(algorithm)
                            isInitialized = false
                            isDownloadingModel.set(false)
                            val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
                            if (isImageMode) {
                                releaseCurrentAlgorithm()
                                processImageMode()
                            } else {
                                cameraExecutor.execute {
                                    releaseCurrentAlgorithm()
                                }
                            }
                        }
                    }
                    override fun onNothingSelected(parent: AdapterView<*>?) {}
                }

                runtimeLabel.visibility = View.VISIBLE
                runtimeSpinner.visibility = View.VISIBLE
            }
            AlgorithmType.POSE_ESTIMATION -> {
                // PoseEstimation は ONNX 固定
                selectedRuntime = "ONNX"
                runtimeLabel.visibility = View.GONE
                runtimeSpinner.visibility = View.GONE
            }
            else -> {
                runtimeLabel.visibility = View.GONE
                runtimeSpinner.visibility = View.GONE
            }
        }
    }

    private fun processAlgorithm(
        img: ByteArray,
        bitmap: Bitmap,
        canvas: Canvas,
        w: Int,
        h: Int
    ): Long {
        val paint = Paint().apply {
            color = Color.WHITE
        }

        val paint2 = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 3f
        }

        val textPaint = Paint().apply {
            color = Color.BLACK
            textSize = 30f
            isAntiAlias = true
        }

        return when (currentAlgorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                poseEstimatorSample.processPoseEstimation(img, canvas, paint, w, h)
            }

            AlgorithmType.OBJECT_DETECTION -> {
                if (selectedRuntime == "ONNX") {
                    onnxObjectDetectionSample.processObjectDetection(
                        img, bitmap, canvas, paint2, textPaint, w, h
                    )
                } else {
                    objectDetectionSample.processObjectDetection(
                        bitmap, canvas, paint2, textPaint, w, h
                    )
                }
            }

            AlgorithmType.CLASSIFICATION -> {
                if (selectedRuntime == "ONNX") {
                    val time = onnxClassificationSample.processClassification(img, w, h)
                    val result = onnxClassificationSample.getLastClassificationResult()
                    runOnUiThread {
                        classificationResultTextView.text = "Classification Result: $result"
                    }
                    time
                } else {
                    val time = classificationSample.processClassification(bitmap)
                    val result = classificationSample.getLastClassificationResult()
                    runOnUiThread {
                        classificationResultTextView.text = "Classification Result: $result"
                    }
                    time
                }
            }

            AlgorithmType.TOKENIZE -> {
                val inputText =
                    tokenizerInputEditText.text.toString().ifEmpty { "Hello world from ailia!" }
                val time = tokenizerSample.processTokenization(inputText)
                val tokens = tokenizerSample.getLastTokenizationResult()
                runOnUiThread {
                    tokenizerOutputTextView.text = "Tokens: $tokens"
                }
                time
            }

            AlgorithmType.TRACKING -> {
                if (selectedRuntime == "ONNX") {
                    val detectionTime = onnxObjectDetectionSample.processObjectDetectionWithoutDrawing(
                        img, w, h, threshold = 0.1f, iou = 1.0f
                    )
                    val detectionResults = onnxObjectDetectionSample.getDetectionResults()
                    val trackingTime = trackerSample.processTrackingWithDetections(
                        canvas, paint2, w, h, detectionResults
                    )
                    val trackingInfo = trackerSample.getLastTrackingResult()
                    runOnUiThread {
                        trackingResultTextView.text = "Tracking Results: $trackingInfo"
                    }
                    detectionTime + trackingTime
                } else {
                    // First run object detection to get detection results without drawing
                    val detectionTime = objectDetectionSample.processObjectDetectionWithoutDrawing(
                        bitmap, w, h, threshold = 0.1f, iou = 1.0f
                    )
                    val detectionResults = objectDetectionSample.getDetectionResults(bitmap)
                    // Then run tracking with the detection results and draw the tracking results
                    val trackingTime = trackerSample.processTrackingWithDetections(
                        canvas, paint2, w, h, detectionResults
                    )
                    val trackingInfo = trackerSample.getLastTrackingResult()
                    runOnUiThread {
                        trackingResultTextView.text = "Tracking Results: $trackingInfo"
                    }
                    detectionTime + trackingTime
                }
            }

            AlgorithmType.SPEECH_TO_TEXT -> {
                val audio: AudioUtil.WavFileData = AudioUtil().loadRawAudio(this.resources.openRawResource(R.raw.demo))
                val startTime = System.nanoTime()
                val text: String =
                    speechSample.process(audio.audioData, audio.channels, audio.sampleRate)
                val endTime = System.nanoTime()
                runOnUiThread {
                    classificationResultTextView.text = "Speech Result:\n$text"
                }
                (endTime - startTime) / 1000000
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                // Voice is handled asynchronously via the generate button
                0
            }

            AlgorithmType.LLM, AlgorithmType.MULTIMODAL_LLM -> {
                // LLM modes are handled asynchronously via the send button
                0
            }
        }
    }

    private fun updateUIVisibility() {
        val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
        val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton

        when (currentAlgorithm) {
            AlgorithmType.TOKENIZE -> {
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.VISIBLE
                tokenizerOutputTextView.visibility = View.VISIBLE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.VISIBLE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmOutputTextView.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.GONE
                voiceModelSpinner.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechModelLabel.visibility = View.GONE
                speechModelSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
            }

            AlgorithmType.CLASSIFICATION -> {
                modeRadioGroup.visibility = View.VISIBLE
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.GONE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmOutputTextView.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.GONE
                voiceModelSpinner.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechModelLabel.visibility = View.GONE
                speechModelSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
            }

            AlgorithmType.TRACKING -> {
                modeRadioGroup.visibility = View.VISIBLE
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.GONE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmOutputTextView.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.GONE
                voiceModelSpinner.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechModelLabel.visibility = View.GONE
                speechModelSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
            }

            AlgorithmType.SPEECH_TO_TEXT -> {
                val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.VISIBLE
                classificationResultTextView.text = "Speech Result: --"
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmOutputTextView.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.GONE
                voiceModelSpinner.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                // Speech-specific UI
                speechModelLabel.visibility = View.VISIBLE
                speechModelSpinner.visibility = View.VISIBLE
                speechModeRadioGroup.visibility = View.VISIBLE
                diarizationCheckBox.visibility = View.VISIBLE
                speechRunButton.visibility = if (isMicMode) View.GONE else View.VISIBLE
                micRecordButton.visibility = if (isMicMode) View.VISIBLE else View.GONE
            }
            AlgorithmType.LLM -> {
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.VISIBLE
                llmInputEditText.visibility = View.VISIBLE
                llmSendButton.visibility = View.VISIBLE
                llmOutputLabel.visibility = View.VISIBLE
                llmOutputTextView.visibility = View.VISIBLE
                llmStatusTextView.visibility = View.VISIBLE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.GONE
                voiceModelSpinner.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechModelLabel.visibility = View.GONE
                speechModelSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                // モード切り替え時にリセット
                llmInputEditText.setText("Hello!")
                llmOutputTextView.text = ""
                llmStatusTextView.text = "Status: Initializing..."
                llmSendButton.isEnabled = false
            }
            AlgorithmType.MULTIMODAL_LLM -> {
                modeRadioGroup.visibility = View.VISIBLE
                imageView.visibility = View.GONE
                if (isImageMode) {
                    cameraPreviewView.visibility = View.GONE
                } else {
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.VISIBLE
                llmInputLabel.visibility = View.VISIBLE
                llmInputEditText.visibility = View.VISIBLE
                llmSendButton.visibility = View.VISIBLE
                llmOutputLabel.visibility = View.VISIBLE
                llmOutputTextView.visibility = View.VISIBLE
                llmStatusTextView.visibility = View.VISIBLE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.GONE
                voiceModelSpinner.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechModelLabel.visibility = View.GONE
                speechModelSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                // モード切り替え時にリセット
                llmInputEditText.setText("What is in this image?")
                llmOutputTextView.text = ""
                llmStatusTextView.text = "Status: Initializing..."
                llmSendButton.isEnabled = false
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmOutputTextView.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.VISIBLE
                voiceModelSpinner.visibility = View.VISIBLE
                voiceStatusTextView.visibility = View.VISIBLE
                voiceGenerateButton.visibility = View.VISIBLE
                voiceResultTextView.visibility = View.VISIBLE
                speechModelLabel.visibility = View.GONE
                speechModelSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                voiceGenerateButton.isEnabled = false
                voiceResultTextView.text = ""
                voiceStatusTextView.text = "Status: Initializing..."
            }

            else -> {
                modeRadioGroup.visibility = View.VISIBLE
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.GONE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmOutputTextView.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                findViewById<TextView>(R.id.voiceModelLabel).visibility = View.GONE
                voiceModelSpinner.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechModelLabel.visibility = View.GONE
                speechModelSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
            }
        }
    }

    private fun switchAlgorithm(newAlgorithm: AlgorithmType) {
        if (isProcessing.get()) {
            Log.i(
                "AILIA_Main",
                "Processing active, queuing algorithm switch to ${newAlgorithm.name}"
            )
            pendingAlgorithmSwitch = newAlgorithm
            return
        }

        executeAlgorithmSwitch(newAlgorithm)
    }

    private fun executeAlgorithmSwitch(newAlgorithm: AlgorithmType) {
        releaseCurrentAlgorithm()
        currentAlgorithm = newAlgorithm
        isInitialized = false
        isDownloadingModel.set(false)
        // アルゴリズム切り替え時にProcessing Timeをリセット
        processingTimeTextView.text = "Processing Time: -- ms"
        updateUIVisibility()

        if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
            processImageMode()
        }
    }

    private fun releaseCurrentAlgorithm() {
        try {
            stopMicRecording()
            poseEstimatorSample.releasePoseEstimator()
            objectDetectionSample.releaseObjectDetection()
            classificationSample.releaseClassification()
            onnxObjectDetectionSample.releaseObjectDetection()
            onnxClassificationSample.releaseClassification()
            tokenizerSample.releaseTokenizer()
            trackerSample.releaseTracker()
            speechSample.releaseSpeech()
            voiceSample.releaseVoice()
            llmSample.release()
            multimodalLLMSample.release()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing algorithms: ${e.message}")
        }
    }

    private fun switchToImageMode() {
        if (isProcessing.get()) {
            Log.i("AILIA_Main", "Processing active, queuing mode switch to Image")
            pendingModeSwitch = R.id.imageRadioButton
            return
        }

        executeModeSwitch(R.id.imageRadioButton)
    }

    private fun switchToCameraMode() {
        if (isProcessing.get()) {
            Log.i("AILIA_Main", "Processing active, queuing mode switch to Camera")
            pendingModeSwitch = R.id.cameraRadioButton
            return
        }

        executeModeSwitch(R.id.cameraRadioButton)
    }

    private fun executeModeSwitch(modeId: Int) {
        when (modeId) {
            R.id.imageRadioButton -> {
                updateUIVisibility()
                stopCamera()
                processImageMode()
            }

            R.id.cameraRadioButton -> {
                if (allPermissionsGranted()) {
                    updateUIVisibility()
                    imageView.setImageBitmap(null)
                    startCamera()
                } else {
                    Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                    modeRadioGroup.check(R.id.imageRadioButton)
                }
            }
        }
    }

    private fun initializeAilia() {
        try {
            when (currentAlgorithm) {
                AlgorithmType.POSE_ESTIMATION -> {
                    val proto: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_proto)
                    val model: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_weight)
                    isInitialized =
                        poseEstimatorSample.initializePoseEstimator(selectedEnvId, proto, model)
                }

                AlgorithmType.OBJECT_DETECTION -> {
                    if (selectedRuntime == "ONNX") {
                        if (isDownloadingModel.get()) return
                        isDownloadingModel.set(true)
                        runOnUiThread {
                            processingTimeTextView.text = "Downloading ONNX model..."
                        }
                        Log.i("AILIA_Main", "ONNX ObjDet: submitting download task to cameraExecutor")
                        cameraExecutor.execute {
                            Log.i("AILIA_Main", "ONNX ObjDet: cameraExecutor task started")
                            try {
                                val downloaded = onnxObjectDetectionSample.downloadModel(object : AiliaOnnxObjectDetectionSample.DownloadListener {
                                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                        runOnUiThread {
                                            processingTimeTextView.text = "Downloading $fileName... $percent%"
                                        }
                                    }
                                    override fun onComplete() {}
                                    override fun onError(error: String) {
                                        runOnUiThread {
                                            processingTimeTextView.text = "Download error: $error"
                                        }
                                    }
                                })
                                Log.i("AILIA_Main", "ONNX ObjDet: download result=$downloaded")
                                if (downloaded) {
                                    Log.i("AILIA_Main", "ONNX ObjDet: initializing with envId=$selectedEnvId")
                                    val success = onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                                    Log.i("AILIA_Main", "ONNX ObjDet: initialization result=$success")
                                    isInitialized = success
                                    isDownloadingModel.set(false)
                                    runOnUiThread {
                                        if (success) {
                                            processingTimeTextView.text = "ONNX model ready"
                                            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                                                processImageMode()
                                            }
                                        } else {
                                            processingTimeTextView.text = "Failed to initialize ONNX model"
                                        }
                                    }
                                } else {
                                    isDownloadingModel.set(false)
                                }
                            } catch (e: Exception) {
                                Log.e("AILIA_Main", "ONNX ObjDet: exception in cameraExecutor", e)
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    processingTimeTextView.text = "Error: ${e.message}"
                                }
                            }
                        }
                        return
                    } else {
                        //val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_tiny)
                        val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_s)
                        isInitialized = objectDetectionSample.initializeObjectDetection(
                            yoloxModel,
                            env = selectedEnvId
                        )
                    }
                }

                AlgorithmType.CLASSIFICATION -> {
                    if (selectedRuntime == "ONNX") {
                        if (isDownloadingModel.get()) return
                        isDownloadingModel.set(true)
                        runOnUiThread {
                            processingTimeTextView.text = "Downloading ONNX model..."
                        }
                        Log.i("AILIA_Main", "ONNX Classification: submitting download task to cameraExecutor")
                        cameraExecutor.execute {
                            Log.i("AILIA_Main", "ONNX Classification: cameraExecutor task started")
                            try {
                                val downloaded = onnxClassificationSample.downloadModel(object : AiliaOnnxClassificationSample.DownloadListener {
                                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                        runOnUiThread {
                                            processingTimeTextView.text = "Downloading $fileName... $percent%"
                                        }
                                    }
                                    override fun onComplete() {}
                                    override fun onError(error: String) {
                                        runOnUiThread {
                                            processingTimeTextView.text = "Download error: $error"
                                        }
                                    }
                                })
                                Log.i("AILIA_Main", "ONNX Classification: download result=$downloaded")
                                if (downloaded) {
                                    Log.i("AILIA_Main", "ONNX Classification: initializing with envId=$selectedEnvId")
                                    val success = onnxClassificationSample.initializeClassification(selectedEnvId)
                                    Log.i("AILIA_Main", "ONNX Classification: initialization result=$success")
                                    isInitialized = success
                                    isDownloadingModel.set(false)
                                    runOnUiThread {
                                        if (success) {
                                            processingTimeTextView.text = "ONNX model ready"
                                            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                                                processImageMode()
                                            }
                                        } else {
                                            processingTimeTextView.text = "Failed to initialize ONNX model"
                                        }
                                    }
                                } else {
                                    isDownloadingModel.set(false)
                                }
                            } catch (e: Exception) {
                                Log.e("AILIA_Main", "ONNX Classification: exception in cameraExecutor", e)
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    processingTimeTextView.text = "Error: ${e.message}"
                                }
                            }
                        }
                        return
                    } else {
                        val classificationModel: ByteArray? = loadRawFile(R.raw.mobilenetv2)
                        isInitialized = classificationSample.initializeClassification(
                            classificationModel,
                            env = selectedEnvId
                        )
                    }
                }

                AlgorithmType.TOKENIZE -> {
                    isInitialized = tokenizerSample.initializeTokenizer()
                }

                AlgorithmType.TRACKING -> {
                    if (selectedRuntime == "ONNX") {
                        if (isDownloadingModel.get()) return
                        isDownloadingModel.set(true)
                        runOnUiThread {
                            processingTimeTextView.text = "Downloading ONNX model..."
                        }
                        Log.i("AILIA_Main", "ONNX Tracking: submitting download task to cameraExecutor")
                        cameraExecutor.execute {
                            Log.i("AILIA_Main", "ONNX Tracking: cameraExecutor task started")
                            try {
                                val downloaded = onnxObjectDetectionSample.downloadModel(object : AiliaOnnxObjectDetectionSample.DownloadListener {
                                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                        runOnUiThread {
                                            processingTimeTextView.text = "Downloading $fileName... $percent%"
                                        }
                                    }
                                    override fun onComplete() {}
                                    override fun onError(error: String) {
                                        runOnUiThread {
                                            processingTimeTextView.text = "Download error: $error"
                                        }
                                    }
                                })
                                Log.i("AILIA_Main", "ONNX Tracking: download result=$downloaded")
                                if (downloaded) {
                                    Log.i("AILIA_Main", "ONNX Tracking: initializing with envId=$selectedEnvId")
                                    val detectorSuccess = onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                                    val trackerSuccess = if (detectorSuccess) trackerSample.initializeTracker() else false
                                    Log.i("AILIA_Main", "ONNX Tracking: detector=$detectorSuccess, tracker=$trackerSuccess")
                                    isInitialized = trackerSuccess
                                    isDownloadingModel.set(false)
                                    runOnUiThread {
                                        if (trackerSuccess) {
                                            processingTimeTextView.text = "ONNX model ready"
                                            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                                                processImageMode()
                                            }
                                        } else {
                                            processingTimeTextView.text = "Failed to initialize ONNX tracking"
                                        }
                                    }
                                } else {
                                    isDownloadingModel.set(false)
                                }
                            } catch (e: Exception) {
                                Log.e("AILIA_Main", "ONNX Tracking: exception in cameraExecutor", e)
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    processingTimeTextView.text = "Error: ${e.message}"
                                }
                            }
                        }
                        return
                    } else {
                        //val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_tiny)
                        val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_s)
                        if (objectDetectionSample.initializeObjectDetection(
                                yoloxModel,
                                env = selectedEnvId
                            )
                        ) {
                            isInitialized = trackerSample.initializeTracker()
                        }
                    }
                }

                AlgorithmType.SPEECH_TO_TEXT -> {
                    if (isDownloadingModel.get()) return
                    isDownloadingModel.set(true)
                    val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
                    runOnUiThread {
                        processingTimeTextView.text = "Downloading speech model (${selectedSpeechModelType.displayName})..."
                    }
                    Log.i("AILIA_Main", "Speech: submitting download task to cameraExecutor, model=${selectedSpeechModelType.displayName}, micMode=$isMicMode")
                    cameraExecutor.execute {
                        Log.i("AILIA_Main", "Speech: cameraExecutor task started")
                        try {
                            val downloaded = speechSample.downloadModel(selectedSpeechModelType, object : AiliaSpeechSample.DownloadListener {
                                override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                    val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                    runOnUiThread {
                                        processingTimeTextView.text = "Downloading $fileName... $percent%"
                                    }
                                }
                                override fun onComplete() {}
                                override fun onError(error: String) {
                                    runOnUiThread {
                                        processingTimeTextView.text = "Download error: $error"
                                    }
                                }
                            })
                            Log.i("AILIA_Main", "Speech: download result=$downloaded")
                            if (downloaded) {
                                Log.i("AILIA_Main", "Speech: initializing with envId=$selectedEnvId, liveMode=$isMicMode")
                                val success = speechSample.initializeSpeech(selectedEnvId, liveMode = isMicMode)
                                Log.i("AILIA_Main", "Speech: initialization result=$success")
                                isInitialized = success
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    if (success) {
                                        processingTimeTextView.text = "${selectedSpeechModelType.displayName} ready"
                                        if (!isMicMode) {
                                            processImageMode()
                                        }
                                    } else {
                                        processingTimeTextView.text = "Failed to initialize speech model"
                                    }
                                }
                            } else {
                                isDownloadingModel.set(false)
                            }
                        } catch (e: Exception) {
                            Log.e("AILIA_Main", "Speech: exception in cameraExecutor", e)
                            isDownloadingModel.set(false)
                            runOnUiThread {
                                processingTimeTextView.text = "Error: ${e.message}"
                            }
                        }
                    }
                    return
                }

                AlgorithmType.TEXT_TO_SPEECH -> {
                    runOnUiThread {
                        voiceStatusTextView.text = "Status: Downloading model..."
                        voiceGenerateButton.isEnabled = false
                    }
                    voiceSample.modelType = selectedVoiceModelType
                    cameraExecutor.execute {
                        val success = voiceSample.initializeVoice(envId = selectedEnvId, listener = object : AiliaVoiceSample.DownloadListener {
                            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                runOnUiThread {
                                    voiceStatusTextView.text = "Status: Downloading $fileName... $percent%"
                                }
                            }
                            override fun onComplete() {
                                Log.i("AILIA_Main", "Voice model download complete")
                            }
                            override fun onError(error: String) {
                                runOnUiThread {
                                    voiceStatusTextView.text = "Status: Error - $error"
                                }
                            }
                        })
                        runOnUiThread {
                            isInitialized = success
                            if (success) {
                                voiceStatusTextView.text = "Status: Ready"
                                voiceGenerateButton.isEnabled = true
                                setupVoiceGenerateButton()
                            } else {
                                voiceStatusTextView.text = "Status: Initialization failed"
                            }
                        }
                    }
                    return // Don't wait for async initialization
                }

                AlgorithmType.LLM -> {
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Downloading model..."
                        llmSendButton.isEnabled = false
                        algorithmSpinner.isEnabled = false
                    }
                    cameraExecutor.execute {
                        val success = llmSample.initialize(this@MainActivity, object : ModelDownloader.DownloadListener {
                            override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                                val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                runOnUiThread {
                                    llmStatusTextView.text = "Status: Downloading model... $percent%"
                                }
                            }
                            override fun onComplete(file: java.io.File) {
                                Log.i("AILIA_Main", "Model download complete")
                            }
                            override fun onError(error: String) {
                                runOnUiThread {
                                    llmStatusTextView.text = "Status: Download error - $error"
                                }
                            }
                        })
                        runOnUiThread {
                            isInitialized = success
                            algorithmSpinner.isEnabled = true
                            if (success) {
                                llmStatusTextView.text = "Status: Ready"
                                llmSendButton.isEnabled = true
                                setupLLMSendButton()
                            } else {
                                llmStatusTextView.text = "Status: Initialization failed"
                            }
                        }
                    }
                    return // Don't wait for async initialization
                }
                AlgorithmType.MULTIMODAL_LLM -> {
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Downloading model..."
                        llmSendButton.isEnabled = false
                        algorithmSpinner.isEnabled = false
                    }
                    cameraExecutor.execute {
                        val success = multimodalLLMSample.initialize(this@MainActivity, object : AiliaMultimodalLLMSample.MultimodalLLMListener {
                            override fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                runOnUiThread {
                                    llmStatusTextView.text = "Status: Downloading $fileName... $percent%"
                                }
                            }
                            override fun onStatus(status: String) {}
                            override fun onToken(token: String) {}
                            override fun onComplete(fullResponse: String) {}
                            override fun onError(error: String) {
                                runOnUiThread {
                                    llmStatusTextView.text = "Status: Error - $error"
                                }
                            }
                        })
                        Log.i("AILIA_Main", "MultimodalLLM: init done on cameraExecutor, success=$success, posting to UI thread")
                        runOnUiThread {
                            Log.i("AILIA_Main", "MultimodalLLM: runOnUiThread callback, success=$success, currentAlgorithm=$currentAlgorithm")
                            isInitialized = success
                            algorithmSpinner.isEnabled = true
                            if (success) {
                                llmStatusTextView.text = "Status: Ready"
                                llmSendButton.isEnabled = true
                                setupMultimodalLLMSendButton()
                                // Load the sample image
                                loadSampleImageForMultimodal()
                                Log.i("AILIA_Main", "MultimodalLLM: UI updated to Ready")
                            } else {
                                llmStatusTextView.text = "Status: Initialization failed"
                                Log.i("AILIA_Main", "MultimodalLLM: initialization failed")
                            }
                        }
                    }
                    return // Don't wait for async initialization
                }
            }

            if (isInitialized) {
                Log.i("AILIA_Main", "Algorithm ${currentAlgorithm.name} initialized successfully")
            } else {
                Log.e("AILIA_Error", "Failed to initialize algorithm ${currentAlgorithm.name}")
            }
        } catch (e: Exception) {
            Log.e(
                "AILIA_Error",
                "Error initializing algorithm ${currentAlgorithm.name}: ${e.message}"
            )
        }
    }

    private fun setupLLMSendButton() {
        llmSendButton.setOnClickListener {
            val userInput = llmInputEditText.text.toString().trim()
            if (userInput.isEmpty()) {
                llmStatusTextView.text = "Status: Please enter a message"
                return@setOnClickListener
            }

            llmSendButton.isEnabled = false
            algorithmSpinner.isEnabled = false
            llmStatusTextView.text = "Status: Generating..."
            llmOutputTextView.text = ""

            cameraExecutor.execute {
                val processingTime = llmSample.chat(userInput, object : AiliaLLMSample.LLMListener {
                    override fun onToken(token: String) {
                        runOnUiThread {
                            llmOutputTextView.append(token)
                        }
                    }
                    override fun onComplete(fullResponse: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: Complete"
                        }
                    }
                    override fun onError(error: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: Error - $error"
                        }
                    }
                })
                runOnUiThread {
                    llmSendButton.isEnabled = true
                    algorithmSpinner.isEnabled = true
                    if (processingTime > 0) {
                        processingTimeTextView.text = "Processing Time: ${processingTime}ms (LLM)"
                    }
                }
            }
        }
    }

    private fun setupMultimodalLLMSendButton() {
        llmSendButton.setOnClickListener {
            val userInput = llmInputEditText.text.toString().trim()
            if (userInput.isEmpty()) {
                llmStatusTextView.text = "Status: Please enter a question about the image"
                return@setOnClickListener
            }

            llmSendButton.isEnabled = false
            algorithmSpinner.isEnabled = false
            modeRadioGroup.isEnabled = false
            for (i in 0 until modeRadioGroup.childCount) {
                modeRadioGroup.getChildAt(i).isEnabled = false
            }
            llmStatusTextView.text = "Status: Generating..."
            llmOutputTextView.text = ""

            val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton
            val imagePath = if (isCameraMode && latestCameraBitmap != null) {
                val tmpFile = File(cacheDir, "camera_frame.png")
                FileOutputStream(tmpFile).use { latestCameraBitmap!!.compress(Bitmap.CompressFormat.PNG, 100, it) }
                tmpFile.absolutePath
            } else {
                null
            }

            Log.i("AILIA_Main", "MultimodalLLM Send: submitting chatWithImage to cameraExecutor, imagePath=$imagePath, userInput='$userInput'")
            cameraExecutor.execute {
                Log.i("AILIA_Main", "MultimodalLLM Send: cameraExecutor task started, calling chatWithImage...")
                val processingTime = multimodalLLMSample.chatWithImage(imagePath, userInput, object : AiliaMultimodalLLMSample.MultimodalLLMListener {
                    override fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {}
                    override fun onStatus(status: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: $status"
                        }
                    }
                    override fun onToken(token: String) {
                        runOnUiThread {
                            llmOutputTextView.append(token)
                        }
                    }
                    override fun onComplete(fullResponse: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: Complete"
                        }
                    }
                    override fun onError(error: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: Error - $error"
                        }
                    }
                })
                runOnUiThread {
                    llmSendButton.isEnabled = true
                    algorithmSpinner.isEnabled = true
                    modeRadioGroup.isEnabled = true
                    for (i in 0 until modeRadioGroup.childCount) {
                        modeRadioGroup.getChildAt(i).isEnabled = true
                    }
                    if (processingTime > 0) {
                        processingTimeTextView.text = "Processing Time: ${processingTime}ms (MultimodalLLM)"
                    }
                }
            }
        }
    }

    private fun loadSampleImageForMultimodal() {
        val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
        if (!isImageMode) return  // Camera mode: don't load sample image
        val imagePath = multimodalLLMSample.getSampleImagePath()
        if (imagePath != null) {
            val bitmap = android.graphics.BitmapFactory.decodeFile(imagePath)
            if (bitmap != null) {
                multimodalImageView.setImageBitmap(bitmap)
            }
        }
    }

    private fun setupVoiceModelSpinner() {
        val voiceModels = arrayOf("V1", "V2", "V3", "V2-Pro")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, voiceModels)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        voiceModelSpinner.adapter = adapter

        voiceModelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newType = when (position) {
                    0 -> VoiceModelType.GPT_SOVITS_V1
                    1 -> VoiceModelType.GPT_SOVITS_V2
                    2 -> VoiceModelType.GPT_SOVITS_V3
                    3 -> VoiceModelType.GPT_SOVITS_V2_PRO
                    else -> VoiceModelType.GPT_SOVITS_V1
                }
                if (newType != selectedVoiceModelType) {
                    selectedVoiceModelType = newType
                    // モデル切り替え時に再初期化
                    if (currentAlgorithm == AlgorithmType.TEXT_TO_SPEECH) {
                        voiceSample.releaseVoice()
                        isInitialized = false
                        voiceGenerateButton.isEnabled = false
                        voiceResultTextView.text = ""
                        initializeAilia()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setupVoiceGenerateButton() {
        voiceGenerateButton.setOnClickListener {
            voiceGenerateButton.isEnabled = false
            voiceStatusTextView.text = "Status: Generating..."
            voiceResultTextView.text = ""

            cameraExecutor.execute {
                val refAudio: AudioUtil.WavFileData = AudioUtil().loadRawAudio(this.resources.openRawResource(R.raw.reference_audio_girl))
                val text: String
                val textLang: String
                if (selectedVoiceModelType == VoiceModelType.GPT_SOVITS_V1) {
                    text = "Hello world. We will introduce ailia AI voice."
                    textLang = "en"
                } else {
                    text = "こんにちは。今日はいい天気ですね。"
                    textLang = "ja"
                }
                val inferenceTime = voiceSample.textToSpeech(
                    refAudio.audioData,
                    refAudio.channels,
                    refAudio.sampleRate,
                    "水をマレーシアから買わなくてはならない。",
                    "ja",
                    text,
                    textLang,
                )
                runOnUiThread {
                    voiceGenerateButton.isEnabled = true
                    voiceStatusTextView.text = "Status: Complete"
                    voiceResultTextView.text = "${selectedVoiceModelType.name} Generated"
                    if (inferenceTime > 0) {
                        processingTimeTextView.text = "Processing Time: ${inferenceTime}ms (Voice)"
                    }
                }
            }
        }
    }

    private fun setupSpeechModelSpinner() {
        val speechModels = SpeechModelType.values().map { it.displayName }.toTypedArray()
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, speechModels)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        speechModelSpinner.adapter = adapter

        // Set current selection
        val currentIndex = SpeechModelType.values().indexOf(selectedSpeechModelType)
        if (currentIndex >= 0) {
            speechModelSpinner.setSelection(currentIndex)
        }

        speechModelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newType = SpeechModelType.values()[position]
                if (newType != selectedSpeechModelType) {
                    selectedSpeechModelType = newType
                    // Stop recording if active
                    stopMicRecording()
                    // Re-download and re-initialize with new model
                    if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
                        speechSample.releaseSpeech()
                        isInitialized = false
                        isDownloadingModel.set(false)
                        classificationResultTextView.text = "Speech Result: --"
                        initializeAilia()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setupSpeechModeRadioGroup() {
        speechModeRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.wavRadioButton -> {
                    speechRunButton.visibility = View.VISIBLE
                    micRecordButton.visibility = View.GONE
                    stopMicRecording()
                    // Re-initialize in non-live mode
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    classificationResultTextView.text = "Speech Result: --"
                    initializeAilia()
                }
                R.id.micRadioButton -> {
                    speechRunButton.visibility = View.GONE
                    micRecordButton.visibility = View.VISIBLE
                    // Re-initialize in live mode
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    classificationResultTextView.text = "Speech Result: (tap Record)"
                    initializeAilia()
                }
            }
        }
    }

    private fun setupDiarizationCheckBox() {
        diarizationCheckBox.setOnCheckedChangeListener { _, isChecked ->
            speechSample.diarizationEnabled = isChecked
            // Re-initialize to apply diarization setting
            stopMicRecording()
            speechSample.releaseSpeech()
            isInitialized = false
            isDownloadingModel.set(false)
            classificationResultTextView.text = "Speech Result: --"
            initializeAilia()
        }
    }

    private fun setupSpeechRunButton() {
        speechRunButton.setOnClickListener {
            if (!isInitialized) {
                classificationResultTextView.text = "Speech model not ready"
                return@setOnClickListener
            }
            if (isProcessing.get()) {
                return@setOnClickListener
            }
            classificationResultTextView.text = "Speech Result: Processing..."
            cameraExecutor.execute {
                try {
                    val audio: AudioUtil.WavFileData = AudioUtil().loadRawAudio(this.resources.openRawResource(R.raw.demo))
                    val startTime = System.nanoTime()
                    val text: String = speechSample.process(audio.audioData, audio.channels, audio.sampleRate)
                    val endTime = System.nanoTime()
                    val timeMs = (endTime - startTime) / 1000000
                    runOnUiThread {
                        classificationResultTextView.text = "Speech Result:\n$text"
                        processingTimeTextView.text = "Processing Time: $timeMs ms"
                    }
                } catch (e: Exception) {
                    Log.e("AILIA_Main", "Speech run error", e)
                    runOnUiThread {
                        classificationResultTextView.text = "Speech Result: Error - ${e.message}"
                    }
                }
            }
        }
    }

    private fun setupMicRecordButton() {
        micRecordButton.setOnClickListener {
            if (isRecording.get()) {
                stopMicRecording()
            } else {
                startMicRecording()
            }
        }
    }

    private fun startMicRecording() {
        if (!isInitialized) {
            classificationResultTextView.text = "Speech model not ready"
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_CODE_PERMISSIONS)
            return
        }

        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize * 2
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                classificationResultTextView.text = "Failed to initialize AudioRecord"
                audioRecord = null
                return
            }

            audioRecord?.startRecording()
            isRecording.set(true)
            micRecordButton.text = "Stop"
            classificationResultTextView.text = "Recording..."

            // Start reading audio data on background thread
            cameraExecutor.execute {
                val buffer = ShortArray(bufferSize)
                val accumulatedText = StringBuilder()

                while (isRecording.get()) {
                    val readResult = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                    if (readResult > 0) {
                        // Convert Short to Float [-1.0, 1.0]
                        val floatBuffer = FloatArray(readResult)
                        for (i in 0 until readResult) {
                            floatBuffer[i] = buffer[i].toFloat() / Short.MAX_VALUE.toFloat()
                        }

                        val text = speechSample.pushLiveAudio(floatBuffer, 1, sampleRate)
                        if (text.isNotEmpty()) {
                            accumulatedText.clear()
                            accumulatedText.append(text)
                            runOnUiThread {
                                classificationResultTextView.text = "Speech Result (live):\n$text"
                            }
                        }
                    }
                }

                // Finalize when recording stops
                val finalText = speechSample.finalizeLiveAudio()
                runOnUiThread {
                    if (finalText.isNotEmpty()) {
                        classificationResultTextView.text = "Speech Result:\n$finalText"
                    } else if (accumulatedText.isNotEmpty()) {
                        classificationResultTextView.text = "Speech Result:\n$accumulatedText"
                    } else {
                        classificationResultTextView.text = "Speech Result: (no speech detected)"
                    }
                }
            }
        } catch (e: SecurityException) {
            Log.e("AILIA_Main", "SecurityException starting mic recording: ${e.message}")
            classificationResultTextView.text = "Microphone permission denied"
        } catch (e: Exception) {
            Log.e("AILIA_Main", "Error starting mic recording: ${e.message}")
            classificationResultTextView.text = "Error starting recording: ${e.message}"
        }
    }

    private fun stopMicRecording() {
        if (isRecording.get()) {
            isRecording.set(false)
            try {
                audioRecord?.stop()
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Error stopping AudioRecord: ${e.message}")
            }
            try {
                audioRecord?.release()
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Error releasing AudioRecord: ${e.message}")
            }
            audioRecord = null
            runOnUiThread {
                micRecordButton.text = "Record"
            }
        }
    }

    private fun processImageMode() {
        // Speech to Text uses speech model spinner and Wav/Mic mode
        if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
            if (!isInitialized) {
                setupSpeechModelSpinner()
                setupSpeechModeRadioGroup()
                setupDiarizationCheckBox()
                setupSpeechRunButton()
                setupMicRecordButton()
                initializeAilia()
            }
            return
        }

        // 非同期モデルダウンロードが必要なモード
        if (selectedRuntime == "ONNX" && (currentAlgorithm == AlgorithmType.OBJECT_DETECTION ||
                    currentAlgorithm == AlgorithmType.CLASSIFICATION ||
                    currentAlgorithm == AlgorithmType.TRACKING)) {
            if (!isInitialized) {
                initializeAilia()
                return
            }
        }

        // TEXT_TO_SPEECHは非同期初期化のため、このメソッドでは処理しない
        if (currentAlgorithm == AlgorithmType.TEXT_TO_SPEECH) {
            if (!isInitialized) {
                setupVoiceModelSpinner()
                initializeAilia()
            }
            return
        }

        // LLMは非同期初期化のため、このメソッドでは処理しない
        if (currentAlgorithm == AlgorithmType.LLM) {
            if (!isInitialized) {
                initializeAilia()
            }
            return
        }

        // MultimodalLLMはperson画像を表示してから初期化
        if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
            val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
            if (isImageMode) {
                val options = BitmapFactory.Options()
                options.inScaled = false
                val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)
                multimodalImageView.setImageBitmap(personBmp)
            }

            if (!isInitialized) {
                initializeAilia()
            }
            return
        }

        if (isProcessing.get()) {
            return
        }

        isProcessing.set(true)

        try {
            if (!isInitialized) {
                initializeAilia()
            }

            if (!isInitialized) {
                runOnUiThread {
                    processingTimeTextView.text = "Failed to initialize ${currentAlgorithm.name}"
                }
                return
            }

            val options = Options()
            options.inScaled = false
            val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)

            val img = ImageUtil().loadRawImage(personBmp)
            val w = personBmp.width
            val h = personBmp.height

            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(img))

            val canvas = Canvas(bitmap)
            val paint = Paint().apply {
                color = Color.WHITE
            }

            val paint2 = Paint().apply {
                style = Paint.Style.STROKE
                color = Color.RED
                strokeWidth = 5f
            }

            val textPaint = Paint().apply {
                color = Color.BLACK
                textSize = 50f
                isAntiAlias = true
            }

            val processingTime = processAlgorithm(img, personBmp, canvas, w, h)

            runOnUiThread {
                if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                    imageView.setImageBitmap(bitmap)
                }
                var timeText = "Processing Time: ${processingTime}ms (${currentAlgorithm.name})"
                when (currentAlgorithm) {
                    AlgorithmType.CLASSIFICATION -> {
                        val result = if (selectedRuntime == "ONNX") onnxClassificationSample.getLastClassificationResult() else classificationSample.getLastClassificationResult()
                        timeText += "\n$result"
                    }
                    AlgorithmType.TRACKING -> {
                        timeText += "\n${trackerSample.getLastTrackingResult()}"
                    }
                    else -> {}
                }
                processingTimeTextView.text = timeText
            }

        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error in image mode: ${e.message}")
            runOnUiThread {
                processingTimeTextView.text = "Processing Error: ${e.message}"
            }
        } finally {
            isProcessing.set(false)

            pendingAlgorithmSwitch?.let { pendingAlgorithm ->
                pendingAlgorithmSwitch = null
                executeAlgorithmSwitch(pendingAlgorithm)
            }

            pendingModeSwitch?.let { pendingMode ->
                pendingModeSwitch = null
                executeModeSwitch(pendingMode)
            }
        }
    }

    private fun startCamera() {
        isStopCamera.set(false)

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(cameraPreviewView.surfaceProvider)
            }

            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, CameraFrameAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("AILIA_Error", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        isStopCamera.set(true)
        try {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
            cameraProviderFuture.addListener({
                val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
                camera = null
                imageAnalyzer = null
            }, ContextCompat.getMainExecutor(this))
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error stopping camera: ${e.message}")
        }
    }

    fun cropToSquare(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // 正方形のサイズは、元のBitmapの幅と高さのうち小さい方に合わせます
        val newSize = if (width < height) width else height

        // 中央を基準にクロップするための開始XとYを計算します
        val startX = (width - newSize) / 2
        val startY = (height - newSize) / 2

        // Bitmapをクロップして正方形の新しいBitmapを作成します
        return Bitmap.createBitmap(bitmap, startX, startY, newSize, newSize)
    }

    private inner class CameraFrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            if (!isInitialized) {
                initializeAilia()
            }
            if (isInitialized) {
                processCameraFrame(image)
            }
            image.close()
        }

        private fun processCameraFrame(image: ImageProxy) {
            if (isProcessing.get()) {
                return
            }
            if (isWaitAlgorithmSwitch.get()) {
                return
            }
            if (isWaitModeSwitch.get()) {
                return
            }
            if (isStopCamera.get()) {
                return
            }

            isProcessing.set(true)

            try {
                var camera_bitmap = ImageUtil().imageProxyToBitmap(image)
                camera_bitmap = cropToSquare(camera_bitmap)

                val img = ImageUtil().loadRawImage(camera_bitmap)
                val w = camera_bitmap.width
                val h = camera_bitmap.height

                Log.i("AILIA_Main", "${w} ${h}")

                val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(bitmap)
                canvas.drawBitmap(camera_bitmap, 0f, 0f, null)

                val processingTime = processAlgorithm(img, bitmap, canvas, w, h)

                runOnUiThread {
                    if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
                        latestCameraBitmap = camera_bitmap
                        multimodalImageView.setImageBitmap(camera_bitmap)
                    } else if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                        imageView.setImageBitmap(bitmap)
                    }

                    val fps = if (processingTime > 0) 1000 / processingTime else 0
                    var timeText = "Processing Time: ${processingTime}ms (${currentAlgorithm.name}) - FPS: $fps"
                    when (currentAlgorithm) {
                        AlgorithmType.CLASSIFICATION -> {
                            val result = if (selectedRuntime == "ONNX") onnxClassificationSample.getLastClassificationResult() else classificationSample.getLastClassificationResult()
                            timeText += "\n$result"
                        }
                        AlgorithmType.TRACKING -> {
                            timeText += "\n${trackerSample.getLastTrackingResult()}"
                        }
                        else -> {}
                    }
                    processingTimeTextView.text = timeText
                }

            } catch (e: Exception) {
                Log.e("AILIA_Error", "Error processing camera frame: ${e.message}")
            } finally {
                isProcessing.set(false)

                pendingAlgorithmSwitch?.let { pendingAlgorithm ->
                    pendingAlgorithmSwitch = null
                    isWaitAlgorithmSwitch.set(true)
                    runOnUiThread {
                        executeAlgorithmSwitch(pendingAlgorithm)
                        isWaitAlgorithmSwitch.set(false)
                    }
                }

                pendingModeSwitch?.let { pendingMode ->
                    pendingModeSwitch = null
                    isWaitModeSwitch.set(true)
                    runOnUiThread {
                        executeModeSwitch(pendingMode)
                        isWaitModeSwitch.set(false)
                    }
                }
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initializeAilia()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT)
                    .show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopMicRecording()
        releaseCurrentAlgorithm()
        cameraExecutor.shutdown()
    }

    @Throws(IOException::class)
    fun inputStreamToByteArray(`in`: InputStream): ByteArray? {
        val bout = ByteArrayOutputStream()
        BufferedOutputStream(bout).use { out ->
            val buf = ByteArray(128)
            var n = 0
            while (`in`.read(buf).also { n = it } > 0) {
                out.write(buf, 0, n)
            }
        }
        return bout.toByteArray()
    }

    @Throws(IOException::class)
    fun loadRawFile(resourceId: Int): ByteArray? {
        val resources = this.resources
        resources.openRawResource(resourceId).use { `in` -> return inputStreamToByteArray(`in`) }
    }
}
