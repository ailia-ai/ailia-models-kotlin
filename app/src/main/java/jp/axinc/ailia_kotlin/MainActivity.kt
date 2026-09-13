package jp.axinc.ailia_kotlin

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.Options
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import axip.ailia.*
import axip.ailia_tflite.*
import axip.ailia_llm.AiliaLLM
import java.io.*
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageView: ImageView
    private lateinit var cameraPreviewView: PreviewView
    private lateinit var modeRadioGroup: RadioGroup
    private lateinit var algorithmSpinner: Spinner
    private lateinit var envSpinner: Spinner
    private lateinit var processingTimeTextView: TextView
    private lateinit var modelDownloadProgressTextView: TextView
    private lateinit var modelDownloadProgressBar: ProgressBar
    private lateinit var resultScrollView: FrameLayout
    private lateinit var classificationResultTextView: TextView
    private lateinit var tokenizerInputLabel: TextView
    private lateinit var tokenizerInputEditText: EditText
    private lateinit var tokenizerOutputLabel: TextView
    private lateinit var tokenizerOutputTextView: TextView
    private lateinit var trackingResultTextView: TextView
    private lateinit var voiceInputEditText: EditText
    private lateinit var voiceEnvSpinner: Spinner
    private lateinit var speechEnvSpinner: Spinner
    private lateinit var voiceReplayButton: Button
    private lateinit var voiceStatusTextView: TextView
    private lateinit var voiceGenerateButton: Button
    private lateinit var voiceResultTextView: TextView
    private lateinit var llmInputLabel: TextView
    private lateinit var llmInputEditText: EditText
    private lateinit var llmSendButton: Button
    private lateinit var llmOutputLabel: TextView
    private lateinit var llmChatContainer: LinearLayout
    private lateinit var llmStatusTextView: TextView
    private lateinit var multimodalImageView: ImageView
    private lateinit var speechLanguageLabel: TextView
    private lateinit var speechLanguageSpinner: Spinner
    private lateinit var speechModeRadioGroup: RadioGroup
    private lateinit var diarizationCheckBox: CheckBox
    private lateinit var liveModeCheckBox: CheckBox
    private lateinit var cameraSpinner: Spinner
    private lateinit var speechRunButton: Button
    private lateinit var micRecordButton: Button
    private lateinit var waveformView: WaveformView
    private lateinit var waveformInfoTextView: TextView
    private lateinit var voiceWaveformView: WaveformView
    private lateinit var modelSpinner: Spinner
    private lateinit var transcriptTextView: TextView
    private lateinit var rootScrollView: ScrollView
    private lateinit var visionRunButton: Button
    private lateinit var speakerProfileLabel: TextView
    private lateinit var speakerProfileSpinner: Spinner
    private lateinit var speakerInputModeRadioGroup: RadioGroup
    private lateinit var speakerVerifyButton: Button
    private lateinit var speakerInputWavNameTextView: TextView
    private lateinit var speakerRecordedActionButton: Button
    private lateinit var speakerRegisterButton: Button
    private lateinit var speakerReferencePlayButton: Button
    private lateinit var speakerWaveformView: WaveformView
    private lateinit var speakerInputPlayButton: Button
    private lateinit var speakerStatusTextView: TextView
    private lateinit var speakerResultTextView: TextView
    private lateinit var voiceFilterInputWaveformLabel: TextView
    private lateinit var voiceFilterPlayInputButton: Button
    private lateinit var voiceFilterOutputWaveformLabel: TextView
    private lateinit var voiceFilterOutputWaveformView: WaveformView
    private lateinit var voiceFilterPlayOutputButton: Button

    // 画像系/Tokenizeアルゴリズムは、Runボタンが押されるまで
    // モデルのダウンロードと実行を行わない
    private var runRequested = false

    // 議事録風トランスクリプト([mm:ss - mm:ss] text の蓄積)
    private val speechTranscript = mutableListOf<String>()
    private var speechIntermediateText: String? = null

    private val modelDirectory by lazy { ModelDownloader.modelDirectory(this) }
    private val poseEstimatorSample by lazy { AiliaPoseEstimatorSample(modelDirectory) }
    private val objectDetectionSample by lazy {
        AiliaTFLiteObjectDetectionSample(modelDirectory)
    }
    private val classificationSample by lazy { AiliaTFLiteClassificationSample(modelDirectory) }
    private val miniLMv2Sample by lazy { AiliaMiniLMv2Sample(modelDirectory) }
    private val trackerSample = AiliaTrackerSample()
    private val speechSample by lazy { AiliaSpeechSample(modelDirectory) }
    private val voiceSample by lazy { AiliaVoiceSample(modelDirectory) }
    private val llmSample = AiliaLLMSample()
    private val multimodalLLMSample = AiliaMultimodalLLMSample()
    private val onnxObjectDetectionSample by lazy { AiliaOnnxObjectDetectionSample(modelDirectory) }
    private val onnxClassificationSample by lazy { AiliaOnnxClassificationSample(modelDirectory) }
    private val onnxRuntimeObjectDetectionSample by lazy {
        OnnxRuntimeObjectDetectionSample(modelDirectory)
    }
    private val onnxRuntimeClassificationSample by lazy {
        OnnxRuntimeClassificationSample(modelDirectory)
    }
    private val u2netSample by lazy { AiliaU2NetSample(modelDirectory) }
    private val detrSample by lazy { AiliaDetrSample(modelDirectory) }
    private val weSpeakerSample by lazy { AiliaWeSpeakerSample(modelDirectory) }
    private val speakerProfileStore by lazy { SpeakerProfileStore(this) }
    private val voiceFilterSample by lazy { AiliaVoiceFilterSample(modelDirectory) }
    private val voiceFilterProfileStore by lazy {
        SpeakerProfileStore(this, "voice_filter_profiles")
    }

    // ObjectDetectionのONNXモデルとしてDETRを使うかどうか(falseならYOLOX)
    private var useDetr = false

    private var selectedEnvId: Int = 0
    private var selectedRuntime: String = if (ONNX_RUNTIME_ONLY) "ONNX" else "TFLite"
    private var useOnnxRuntime = false
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
    private val operationGeneration = AtomicLong(0)
    private val visionDisplayGeneration = AtomicLong(0)
    private val activityDestroyed = AtomicBoolean(false)

    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var latestCameraBitmap: Bitmap? = null

    // カメラ選択(リストボックスで切り替え)
    private data class CameraChoice(val name: String, val selector: CameraSelector)
    private val cameraChoices = listOf(
        CameraChoice("Back Camera", CameraSelector.DEFAULT_BACK_CAMERA),
        CameraChoice("Front Camera", CameraSelector.DEFAULT_FRONT_CAMERA),
    )
    private var selectedCameraIndex = 0

    // 音声認識処理用の低優先度Executor。カメラ/マイク読み出しと分離して
    // transcribe中もUI(波形表示等)がカクつかないようにする。
    private lateinit var speechExecutor: ExecutorService

    private var selectedVoiceModelType: VoiceModelType = VoiceModelType.GPT_SOVITS_V1
    private var selectedSpeechModelType: SpeechModelType = SpeechModelType.SENSEVOICE_SMALL
    private var selectedSpeechLanguage: String = "ja"
    private var selectedLLMModelType: LLMModelType = LLMModelType.GEMMA_4_E2B
    private var isUpdatingSpeechOptionChecks = false

    private data class SpeakerReference(
        val name: String,
        val embedding: FloatArray? = null,
        val presetRawResource: Int? = null,
        val profileId: String? = null,
    )

    private enum class SpeakerRecordingPurpose { VERIFY, ENROLL }
    private enum class SpeakerPlayback { REFERENCE, INPUT }
    private enum class VoiceFilterRecordingPurpose { FILTER, ENROLL }
    private enum class VoiceFilterPlayback { REFERENCE, BEFORE, AFTER }
    private enum class PendingAudioAction {
        SPEECH,
        SPEAKER_VERIFY,
        SPEAKER_ENROLL,
        VOICE_FILTER,
        VOICE_FILTER_ENROLL,
    }

    private var speakerReferences = emptyList<SpeakerReference>()
    private val presetSpeakerEmbeddings = mutableMapOf<Int, FloatArray>()
    private var speakerRecordingPurpose: SpeakerRecordingPurpose? = null
    private var lastSpeakerMicAudio: FloatArray? = null
    private var lastSpeakerMicSampleRate = SpeakerVerificationAudio.SAMPLE_RATE
    private var speakerPlayback: SpeakerPlayback? = null
    private var voiceFilterReferences = emptyList<SpeakerReference>()
    private val presetVoiceFilterEmbeddings = mutableMapOf<Int, FloatArray>()
    private var voiceFilterRecordingPurpose: VoiceFilterRecordingPurpose? = null
    private var lastVoiceFilterMicAudio: FloatArray? = null
    private var lastVoiceFilterMicSampleRate = VoiceFilterAudio.SAMPLE_RATE
    private var latestVoiceFilterResult: VoiceFilterResult? = null
    private var voiceFilterPlayback: VoiceFilterPlayback? = null
    private var pendingAudioAction: PendingAudioAction? = null

    // マイク録音のREC経過時間表示用
    private val recTimerHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var recStartMs: Long = 0
    private val recTimerRunnable = object : Runnable {
        override fun run() {
            if (speechSample.isMicRecording) {
                val elapsedSec = (android.os.SystemClock.elapsedRealtime() - recStartMs) / 1000
                waveformInfoTextView.text = String.format(Locale.ROOT, "● REC %02d:%02d", elapsedSec / 60, elapsedSec % 60)
                recTimerHandler.postDelayed(this, 500)
            }
        }
    }

    enum class AlgorithmType {
        POSE_ESTIMATION,
        OBJECT_DETECTION,
        TRACKING,
        TOKENIZE,
        CLASSIFICATION,
        BACKGROUND_REMOVAL,
        SPEECH_TO_TEXT,
        TEXT_TO_SPEECH,
        LLM,
        MULTIMODAL_LLM,
        SPEAKER_VERIFICATION,
        VOICE_FILTER,
    }

    companion object {
        init {
            System.loadLibrary("ailia")
        }

        private const val REQUEST_CODE_CAMERA_PERMISSION = 10
        private const val REQUEST_CODE_AUDIO_PERMISSION = 11
        private const val ONNX_RUNTIME_ENV_ID = Int.MIN_VALUE

        /**
         * ORTビルド: ONNX Runtimeを選択できないアルゴリズム/モデルをスピナーで選択不可にする。
         */
        private const val ONNX_RUNTIME_ONLY = true

        /** ONNX Runtimeで実行できないアルゴリズム(ailia Speech / Voice / LLM専用) */
        private val ALGORITHMS_WITHOUT_ONNX_RUNTIME = setOf(
            AlgorithmType.SPEECH_TO_TEXT,
            AlgorithmType.TEXT_TO_SPEECH,
            AlgorithmType.LLM,
            AlgorithmType.MULTIMODAL_LLM,
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Android 15 (targetSdk 35) 以降はエッジツーエッジが強制されるため、
        // システムバー/ディスプレイカットアウトのインセットを上下左右のpaddingとして適用する。
        // ActionBar が消費した後の残りのインセット(ステータスバー分は消費済み)を
        // コンテンツのルートに反映することで、ナビゲーションバー/ノッチとの被りを防ぐ。
        val rootLayout = findViewById<View>(R.id.rootLayout)
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        // cameraExecutorはsetupModeSelection()より先に初期化する必要がある
        // (SpinnerのonItemSelectedでinitializeAilia()が呼ばれるため)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 音声認識はバックグラウンド優先度の専用スレッドで実行し、
        // 推論中でもUIスレッドがCPUを確保できるようにする
        speechExecutor = Executors.newSingleThreadExecutor { r ->
            Thread {
                android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
                r.run()
            }
        }

        initializeViews()
        adjustContentSizeForScreen()
        setupModeSelection()
        updateUIVisibility()

        // Camera and microphone permissions are requested only when their feature is selected.
    }

    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        cameraPreviewView = findViewById(R.id.cameraPreviewView)
        modeRadioGroup = findViewById(R.id.modeRadioGroup)
        algorithmSpinner = findViewById(R.id.algorithmSpinner)
        envSpinner = findViewById(R.id.envSpinner)
        processingTimeTextView = findViewById(R.id.processingTimeTextView)
        modelDownloadProgressTextView = findViewById(R.id.modelDownloadProgressTextView)
        modelDownloadProgressBar = findViewById(R.id.modelDownloadProgressBar)
        resultScrollView = findViewById(R.id.resultScrollView)
        classificationResultTextView = findViewById(R.id.classificationResultTextView)
        tokenizerInputLabel = findViewById(R.id.tokenizerInputLabel)
        tokenizerInputEditText = findViewById(R.id.tokenizerInputEditText)
        tokenizerOutputLabel = findViewById(R.id.tokenizerOutputLabel)
        tokenizerOutputTextView = findViewById(R.id.tokenizerOutputTextView)
        trackingResultTextView = findViewById(R.id.trackingResultTextView)
        voiceInputEditText = findViewById(R.id.voiceInputEditText)
        voiceEnvSpinner = findViewById(R.id.voiceEnvSpinner)
        speechEnvSpinner = findViewById(R.id.speechEnvSpinner)
        voiceReplayButton = findViewById(R.id.voiceReplayButton)
        voiceStatusTextView = findViewById(R.id.voiceStatusTextView)
        voiceGenerateButton = findViewById(R.id.voiceGenerateButton)
        voiceResultTextView = findViewById(R.id.voiceResultTextView)
        llmInputLabel = findViewById(R.id.llmInputLabel)
        llmInputEditText = findViewById(R.id.llmInputEditText)
        llmSendButton = findViewById(R.id.llmSendButton)
        llmOutputLabel = findViewById(R.id.llmOutputLabel)
        llmChatContainer = findViewById(R.id.llmChatContainer)
        llmStatusTextView = findViewById(R.id.llmStatusTextView)
        multimodalImageView = findViewById(R.id.multimodalImageView)
        speechLanguageLabel = findViewById(R.id.speechLanguageLabel)
        speechLanguageSpinner = findViewById(R.id.speechLanguageSpinner)
        speechModeRadioGroup = findViewById(R.id.speechModeRadioGroup)
        diarizationCheckBox = findViewById(R.id.diarizationCheckBox)
        liveModeCheckBox = findViewById(R.id.liveModeCheckBox)
        cameraSpinner = findViewById(R.id.cameraSpinner)
        transcriptTextView = findViewById(R.id.transcriptTextView)
        speechRunButton = findViewById(R.id.speechRunButton)
        micRecordButton = findViewById(R.id.micRecordButton)
        waveformView = findViewById(R.id.waveformView)
        waveformInfoTextView = findViewById(R.id.waveformInfoTextView)
        voiceWaveformView = findViewById(R.id.voiceWaveformView)
        modelSpinner = findViewById(R.id.modelSpinner)
        rootScrollView = findViewById(R.id.rootScrollView)
        visionRunButton = findViewById(R.id.visionRunButton)
        speakerProfileLabel = findViewById(R.id.speakerProfileLabel)
        speakerProfileSpinner = findViewById(R.id.speakerProfileSpinner)
        speakerInputModeRadioGroup = findViewById(R.id.speakerInputModeRadioGroup)
        speakerVerifyButton = findViewById(R.id.speakerVerifyButton)
        speakerInputWavNameTextView = findViewById(R.id.speakerInputWavNameTextView)
        speakerRecordedActionButton = findViewById(R.id.speakerRecordedActionButton)
        speakerRegisterButton = findViewById(R.id.speakerRegisterButton)
        speakerReferencePlayButton = findViewById(R.id.speakerReferencePlayButton)
        speakerWaveformView = findViewById(R.id.speakerWaveformView)
        speakerInputPlayButton = findViewById(R.id.speakerInputPlayButton)
        speakerStatusTextView = findViewById(R.id.speakerStatusTextView)
        speakerResultTextView = findViewById(R.id.speakerResultTextView)
        voiceFilterInputWaveformLabel = findViewById(R.id.voiceFilterInputWaveformLabel)
        voiceFilterPlayInputButton = findViewById(R.id.voiceFilterPlayInputButton)
        voiceFilterOutputWaveformLabel = findViewById(R.id.voiceFilterOutputWaveformLabel)
        voiceFilterOutputWaveformView = findViewById(R.id.voiceFilterOutputWaveformView)
        voiceFilterPlayOutputButton = findViewById(R.id.voiceFilterPlayOutputButton)
    }

    /**
     * 項目ごとに選択可否を指定できるスピナー用アダプタ。
     * 選択不可の項目はドロップダウンでグレー表示になり、タップしても選択されない。
     */
    private class SelectableArrayAdapter(
        context: Context,
        itemLayout: Int,
        private val items: Array<String>,
        private val enabledFlags: BooleanArray,
    ) : ArrayAdapter<String>(context, itemLayout, items) {
        init {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }

        override fun areAllItemsEnabled(): Boolean = enabledFlags.all { it }

        override fun isEnabled(position: Int): Boolean = enabledFlags.getOrElse(position) { true }

        override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = super.getDropDownView(position, convertView, parent)
            val enabled = isEnabled(position)
            view.isEnabled = enabled
            view.alpha = if (enabled) 1.0f else 0.4f
            (view as? TextView)?.text = if (enabled) items[position] else "${items[position]} (ORT N/A)"
            return view
        }
    }

    private fun isAlgorithmSelectable(algorithm: AlgorithmType): Boolean =
        !ONNX_RUNTIME_ONLY || algorithm !in ALGORITHMS_WITHOUT_ONNX_RUNTIME

    /** TFLiteモデル項目はONNX Runtimeで実行できないためORTビルドでは選択不可 */
    private fun isModelItemSelectable(label: String): Boolean =
        !ONNX_RUNTIME_ONLY || !label.contains("(TFLite)")

    private fun setupModeSelection() {
        val algorithms = arrayOf(
            "PoseEstimation",
            "ObjectDetection",
            "Tracking",
            "Tokenize",
            "Classification",
            "BackgroundRemoval",
            "Speech2Text",
            "Text2Speech",
            "LLM",
            "MultimodalLLM",
            "SpeakerVerification",
            "VoiceFilter",
        )

        // 全体メニュー風の見た目にするため専用のitemレイアウト(白太字・中央寄せ)を使う
        // ORTビルドではONNX Runtime非対応のアルゴリズムを選択不可にする
        val algorithmEnabled = BooleanArray(algorithms.size) { index ->
            isAlgorithmSelectable(AlgorithmType.values()[index])
        }
        val adapter = SelectableArrayAdapter(this, R.layout.spinner_item_menu, algorithms, algorithmEnabled)
        algorithmSpinner.adapter = adapter

        algorithmSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val newAlgorithm = AlgorithmType.values()[position]
                updateEnvSpinner(newAlgorithm)
                updateModelSpinner(newAlgorithm)
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

        // 初回起動時、最初のモデル初期化より前にデフォルトのenv(GPU等)を確定させる。
        // これを行わないと selectedEnvId=0 のまま初期化され、後から遅延実行される
        // algorithmSpinner の onItemSelected が envスピナーの表示だけをデフォルト値に
        // 更新するため「表示env≠実際に使われるenv」の不一致が起きる。
        updateEnvSpinner(currentAlgorithm)
        updateModelSpinner()
        setupCameraSpinner()
        setupVisionRunButton()

        switchToImageMode()
    }

    /**
     * 画像/カメラの表示領域を画面サイズに合わせて調整する。
     * 横画面でも高さが画面内に収まるよう、正方形の一辺を
     * 「画面幅(マージン込み)」と「画面高さの60%」の小さい方にする。
     * 画面回転時はActivityが再生成されるためonCreateで再計算される。
     */
    private fun adjustContentSizeForScreen() {
        val dm = resources.displayMetrics
        val horizontalMargin = (32 * dm.density).toInt()
        val side = minOf(dm.widthPixels - horizontalMargin, (dm.heightPixels * 0.6f).toInt())
        for (view in listOf<View>(imageView, cameraPreviewView)) {
            val lp = view.layoutParams
            lp.width = side
            lp.height = side
            view.layoutParams = lp
        }
    }

    /** Runボタンが必要な(押すまで実行しない)アルゴリズムかどうか */
    private fun needsVisionRunButton(algorithm: AlgorithmType = currentAlgorithm): Boolean {
        return when (algorithm) {
            AlgorithmType.POSE_ESTIMATION,
            AlgorithmType.OBJECT_DETECTION,
            AlgorithmType.TRACKING,
            AlgorithmType.CLASSIFICATION,
            AlgorithmType.BACKGROUND_REMOVAL,
            AlgorithmType.TOKENIZE -> true
            else -> false
        }
    }

    /** CameraFrameAnalyzerで画像推論を行うアルゴリズムかどうか。 */
    private fun usesVisionCameraInference(algorithm: AlgorithmType): Boolean {
        return when (algorithm) {
            AlgorithmType.POSE_ESTIMATION,
            AlgorithmType.OBJECT_DETECTION,
            AlgorithmType.TRACKING,
            AlgorithmType.CLASSIFICATION,
            AlgorithmType.BACKGROUND_REMOVAL -> true
            else -> false
        }
    }

    /** Run状態をリセットしてボタン表記をRunに戻す */
    private fun resetRunState() {
        runRequested = false
        visionRunButton.text = "Run"
    }

    /** モデル変更前の推論結果を消し、現在の入力表示へ戻す。 */
    private fun resetVisionDisplayForModelChange() {
        // 切り替え前に開始したカメラ推論のUI反映も無効化する。
        visionDisplayGeneration.incrementAndGet()
        latestCameraBitmap = null
        processingTimeTextView.text = "Processing Time: -- ms"
        // Speech2Textでは同じTextViewをSpeech Resultとして使うため表記を切り替える。
        classificationResultTextView.text = if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
            "Speech Result: --"
        } else {
            "Classification Result: --"
        }
        trackingResultTextView.text = "Tracking Results: --"

        if (modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton) {
            // ImageViewはPreviewViewより手前にあるため、前モデルの最終フレームを消す。
            imageView.setImageBitmap(null)
            imageView.invalidate()
        } else if (currentAlgorithm != AlgorithmType.TOKENIZE) {
            val options = Options().apply { inScaled = false }
            imageView.setImageBitmap(
                BitmapFactory.decodeResource(resources, R.raw.person, options)
            )
        }
    }

    private fun setupVisionRunButton() {
        visionRunButton.setOnClickListener {
            if (isDownloadingModel.get()) {
                return@setOnClickListener
            }
            val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton &&
                currentAlgorithm != AlgorithmType.TOKENIZE
            if (isCameraMode) {
                if (runRequested) {
                    // Stop: フレーム処理を停止する(プレビューは継続)
                    resetRunState()
                    processingTimeTextView.text = "Stopped. Press Run to restart"
                } else {
                    runRequested = true
                    visionRunButton.text = "Stop"
                    // CameraFrameAnalyzerが次フレームから処理を開始する
                }
            } else {
                runRequested = true
                processImageMode()
            }
        }
    }

    private fun setupCameraSpinner() {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, cameraChoices.map { it.name }.toTypedArray())
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        cameraSpinner.adapter = adapter
        cameraSpinner.setSelection(selectedCameraIndex)
        cameraSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position != selectedCameraIndex) {
                    selectedCameraIndex = position
                    // カメラモード中なら選択したカメラで再バインドする
                    if (modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton) {
                        startCamera()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private data class BackendEnvironment(
        val displayName: String,
        val envId: Int,
        val isOnnxRuntime: Boolean = false,
    )

    private fun setupOnnxEnvSpinner(
        useBlas: Boolean,
        target: Spinner = envSpinner,
        allowOnnxRuntime: Boolean = false,
    ) {
        try {
            if (ailiaEnvironments == null) {
                Ailia.SetTemporaryCachePath(cacheDir.absolutePath)
                ailiaEnvironments = AiliaModel.getEnvironments()
            }
            val environments = ailiaEnvironments!!.map {
                BackendEnvironment(it.toBackendDisplayName(), it.id)
            }.toMutableList()
            if (allowOnnxRuntime) {
                val onnxRuntimeEnvironment = BackendEnvironment(
                    displayName = "ONNX Runtime (CPU)",
                    envId = ONNX_RUNTIME_ENV_ID,
                    isOnnxRuntime = true,
                )
                // ORTビルドではONNX Runtimeを先頭に置きデフォルトにする。ailiaは後ろに並べる
                if (ONNX_RUNTIME_ONLY) {
                    environments.add(0, onnxRuntimeEnvironment)
                } else {
                    environments += onnxRuntimeEnvironment
                }
            } else {
                useOnnxRuntime = false
            }
            val adapter = ArrayAdapter(
                this,
                android.R.layout.simple_spinner_item,
                environments.map { it.displayName }.toTypedArray(),
            )
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            target.adapter = adapter

            var defaultIndex = 0
            if (allowOnnxRuntime && (useOnnxRuntime || ONNX_RUNTIME_ONLY)) {
                // ORTビルド、またはユーザーがONNX Runtimeを選択済みならONNX Runtimeをデフォルトにする
                defaultIndex = environments.indexOfFirst { it.isOnnxRuntime }.coerceAtLeast(0)
            } else if (useBlas) {
                // デフォルトはBLAS (CPU-OpenBlas)
                defaultIndex = environments.indexOfFirst { candidate ->
                    !candidate.isOnnxRuntime &&
                        ailiaEnvironments!!.any { it.id == candidate.envId && it.name.contains("OpenBlas", ignoreCase = true) }
                }.coerceAtLeast(0)
            } else {
                // デフォルトはGPU
                defaultIndex = environments.indexOfFirst { candidate ->
                    !candidate.isOnnxRuntime &&
                        ailiaEnvironments!!.any {
                            it.id == candidate.envId &&
                                it.type == AiliaEnvironment.TYPE_GPU &&
                                it.props and AiliaEnvironment.PROPERTY_FP16 == 0
                        }
                }.coerceAtLeast(0)
            }
            target.setSelection(defaultIndex)
            selectedEnvId = environments[defaultIndex].envId
            useOnnxRuntime = environments[defaultIndex].isOnnxRuntime

            target.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    val environment = environments[position]
                    if (
                        environment.envId != selectedEnvId ||
                        environment.isOnnxRuntime != useOnnxRuntime
                    ) {
                        selectedEnvId = environment.envId
                        useOnnxRuntime = environment.isOnnxRuntime
                        isInitialized = false
                        isDownloadingModel.set(false)
                        resetRunState()
                        resetVisionDisplayForModelChange()
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

            target.visibility = View.VISIBLE
        } catch (e: Exception) {
            Log.e("AILIA_Main", "Failed to get ailia environments: ${e.message}")
            target.visibility = View.GONE
        }
    }

    private fun updateEnvSpinner(algorithm: AlgorithmType) {
        when (algorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                setupOnnxEnvSpinner(useBlas = false, allowOnnxRuntime = true)
            }

            AlgorithmType.BACKGROUND_REMOVAL -> {
                setupOnnxEnvSpinner(useBlas = false, allowOnnxRuntime = true)
            }

            AlgorithmType.SPEAKER_VERIFICATION,
            AlgorithmType.VOICE_FILTER,
            AlgorithmType.TOKENIZE -> {
                setupOnnxEnvSpinner(useBlas = true, allowOnnxRuntime = true)
            }

            AlgorithmType.SPEECH_TO_TEXT -> {
                // Speech2Textのバックエンド選択はRun/Recordボタンの右側に表示する
                envSpinner.visibility = View.GONE
                setupOnnxEnvSpinner(useBlas = true, target = speechEnvSpinner)
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                // TTSのバックエンド選択はGenerateボタンの右側に表示する
                envSpinner.visibility = View.GONE
                setupOnnxEnvSpinner(useBlas = true, target = voiceEnvSpinner)
            }

            AlgorithmType.OBJECT_DETECTION, AlgorithmType.CLASSIFICATION, AlgorithmType.TRACKING -> {
                if (selectedRuntime == "ONNX") {
                    setupOnnxEnvSpinner(
                        useBlas = true,
                        allowOnnxRuntime = true,
                    )
                } else {
                    // TFLite: Reference (CPU) と NNAPI を表示
                    useOnnxRuntime = false
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
                                resetRunState()
                                resetVisionDisplayForModelChange()
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

                    envSpinner.visibility = View.VISIBLE
                }
            }

            else -> {
                envSpinner.visibility = View.GONE
            }
        }
    }

    /**
     * モデル選択スピナーを現在のアルゴリズムに合わせて更新する。
     * S2T/TTS/LLMのモデル選択や、YOLOX等のTFLite/ONNX(ランタイム)選択もここに統合。
     */
    private fun updateModelSpinner(algorithm: AlgorithmType = currentAlgorithm) {
        var selectedIndex = 0
        val items: Array<String> = when (algorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                selectedRuntime = "ONNX"
                arrayOf("LightweightHumanPose")
            }
            AlgorithmType.OBJECT_DETECTION -> {
                selectedIndex = when {
                    selectedRuntime == "ONNX" && useDetr -> 2
                    selectedRuntime == "ONNX" -> 1
                    else -> 0
                }
                arrayOf("YOLOX-S (TFLite)", "YOLOX-S (ONNX)", "DETR ResNet50 (ONNX)")
            }
            AlgorithmType.TRACKING -> {
                selectedIndex = if (selectedRuntime == "ONNX") 1 else 0
                arrayOf("YOLOX-S + ByteTrack (TFLite)", "YOLOX-S + ByteTrack (ONNX)")
            }
            AlgorithmType.CLASSIFICATION -> {
                selectedIndex = when {
                    selectedRuntime == "TFLite" && classificationSample.modelType == TFLiteClassificationModelType.RESNET50 -> 1
                    selectedRuntime == "TFLite" -> 0
                    onnxClassificationSample.modelType == OnnxClassificationModelType.VIT_B16 -> 4
                    onnxClassificationSample.modelType == OnnxClassificationModelType.RESNET50 -> 3
                    else -> 2
                }
                arrayOf(
                    "MobileNetV2 (TFLite)",
                    "ResNet50 (TFLite)",
                    "MobileNetV2 (ONNX)",
                    "ResNet50 (ONNX)",
                    "ViT-B/16 (ONNX)",
                )
            }
            AlgorithmType.TOKENIZE -> arrayOf("Multilingual MiniLMv2 (L12)")
            AlgorithmType.BACKGROUND_REMOVAL -> {
                selectedRuntime = "ONNX"
                arrayOf("U-2-Net")
            }
            AlgorithmType.SPEECH_TO_TEXT -> {
                selectedIndex = SpeechModelType.values().indexOf(selectedSpeechModelType)
                SpeechModelType.values().map { it.displayName }.toTypedArray()
            }
            AlgorithmType.TEXT_TO_SPEECH -> {
                selectedIndex = VoiceModelType.values().indexOf(selectedVoiceModelType)
                VoiceModelType.values().map { it.displayName }.toTypedArray()
            }
            AlgorithmType.LLM -> {
                selectedIndex = LLMModelType.values().indexOf(selectedLLMModelType)
                LLMModelType.values().map { "${it.displayName} (Q4_K_M)" }.toTypedArray()
            }
            AlgorithmType.MULTIMODAL_LLM -> arrayOf("Gemma-3 4B IT (Q4_K_M)")
            AlgorithmType.SPEAKER_VERIFICATION -> {
                selectedRuntime = "ONNX"
                arrayOf("WeSpeaker ResNet34 (VoxCeleb) + Silero VAD v6")
            }
            AlgorithmType.VOICE_FILTER -> {
                selectedRuntime = "ONNX"
                arrayOf("VoiceFilter + dynamic d-vector embedder + Silero VAD v6")
            }
        }

        val itemEnabled = BooleanArray(items.size) { isModelItemSelectable(items[it]) }
        if (selectedIndex in items.indices && !itemEnabled[selectedIndex]) {
            selectedIndex = itemEnabled.indexOfFirst { it }.coerceAtLeast(0)
        }
        val adapter = SelectableArrayAdapter(this, android.R.layout.simple_spinner_item, items, itemEnabled)
        modelSpinner.onItemSelectedListener = null
        modelSpinner.adapter = adapter
        if (selectedIndex in items.indices) {
            modelSpinner.setSelection(selectedIndex, false)
        }
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!itemEnabled.getOrElse(position) { true }) return
                onModelSelected(algorithm, position)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    /** モデル選択スピナーの選択変更を各アルゴリズムに反映する */
    private fun onModelSelected(algorithm: AlgorithmType, position: Int) {
        when (algorithm) {
            AlgorithmType.OBJECT_DETECTION -> {
                val newRuntime = if (position >= 1) "ONNX" else "TFLite"
                val newUseDetr = position == 2
                if (newRuntime != selectedRuntime || newUseDetr != useDetr) {
                    selectedRuntime = newRuntime
                    useDetr = newUseDetr
                    if (newRuntime == "TFLite") useOnnxRuntime = false
                    updateEnvSpinner(algorithm)
                    isInitialized = false
                    isDownloadingModel.set(false)
                    resetRunState()
                    resetVisionDisplayForModelChange()
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
            AlgorithmType.TRACKING -> {
                val newRuntime = if (position == 1) "ONNX" else "TFLite"
                if (newRuntime != selectedRuntime) {
                    selectedRuntime = newRuntime
                    if (newRuntime == "TFLite") useOnnxRuntime = false
                    updateEnvSpinner(algorithm)
                    isInitialized = false
                    isDownloadingModel.set(false)
                    resetRunState()
                    resetVisionDisplayForModelChange()
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
            AlgorithmType.CLASSIFICATION -> {
                val newRuntime = if (position <= 1) "TFLite" else "ONNX"
                val newTfliteModel = when (position) {
                    1 -> TFLiteClassificationModelType.RESNET50
                    else -> TFLiteClassificationModelType.MOBILENETV2
                }
                val newOnnxModel = when (position) {
                    3 -> OnnxClassificationModelType.RESNET50
                    4 -> OnnxClassificationModelType.VIT_B16
                    else -> OnnxClassificationModelType.MOBILENETV2
                }
                val modelChanged = if (newRuntime == "TFLite") {
                    newTfliteModel != classificationSample.modelType
                } else {
                    newOnnxModel != onnxClassificationSample.modelType
                }
                if (newRuntime != selectedRuntime || modelChanged) {
                    selectedRuntime = newRuntime
                    if (newRuntime == "TFLite") {
                        useOnnxRuntime = false
                        classificationSample.modelType = newTfliteModel
                    } else {
                        onnxClassificationSample.modelType = newOnnxModel
                        onnxRuntimeClassificationSample.modelType = newOnnxModel
                    }
                    updateEnvSpinner(algorithm)
                    isInitialized = false
                    isDownloadingModel.set(false)
                    resetRunState()
                    resetVisionDisplayForModelChange()
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
            AlgorithmType.SPEECH_TO_TEXT -> {
                val newType = SpeechModelType.values()[position]
                if (newType != selectedSpeechModelType) {
                    selectedSpeechModelType = newType
                    stopMicRecording(finalize = false)
                    // モデルダウンロードはRun/Record押下時まで遅延する
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    clearTranscript()
                    classificationResultTextView.text = "Speech Result: --"
                }
            }
            AlgorithmType.TEXT_TO_SPEECH -> {
                val newType = VoiceModelType.values()[position]
                if (newType != selectedVoiceModelType) {
                    selectedVoiceModelType = newType
                    // モデル切り替え時は解放のみ(ダウンロードはGenerate押下時)
                    voiceSample.releaseVoice()
                    isInitialized = false
                    voiceWaveformView.clear()
                    voiceReplayButton.visibility = View.GONE
                    voiceResultTextView.text = ""
                    voiceInputEditText.setText(newType.defaultText)
                    voiceStatusTextView.text = "Status: Press Generate to synthesize"
                    processingTimeTextView.text = "Processing Time: -- ms"
                }
            }
            AlgorithmType.LLM -> {
                val newType = LLMModelType.values()[position]
                if (newType != selectedLLMModelType) {
                    selectedLLMModelType = newType
                    // モデル切り替え時は解放のみ(ダウンロードはSend押下時)
                    llmSample.release()
                    isInitialized = false
                    llmChatContainer.removeAllViews()
                    llmSendButton.isEnabled = true
                    llmStatusTextView.text = "Status: Press Send to run"
                }
            }
            else -> {}
        }
    }

    private fun processAlgorithm(
        img: ByteArray,
        bitmap: Bitmap,
        canvas: Canvas,
        w: Int,
        h: Int,
        tokenizerInput: String? = null,
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
                if (selectedRuntime == "ONNX" && useDetr) {
                    detrSample.processObjectDetection(bitmap, canvas, paint2, textPaint, w, h)
                } else if (selectedRuntime == "ONNX" && useOnnxRuntime) {
                    onnxRuntimeObjectDetectionSample.processObjectDetection(
                        img, canvas, paint2, textPaint, w, h
                    )
                } else if (selectedRuntime == "ONNX") {
                    onnxObjectDetectionSample.processObjectDetection(
                        img, canvas, paint2, textPaint, w, h
                    )
                } else {
                    objectDetectionSample.processObjectDetection(
                        bitmap, canvas, paint2, textPaint, w, h
                    )
                }
            }

            AlgorithmType.CLASSIFICATION -> {
                if (selectedRuntime == "ONNX" && useOnnxRuntime) {
                    val time = onnxRuntimeClassificationSample.processClassification(img, w, h)
                    val result = onnxRuntimeClassificationSample.getLastClassificationResult()
                    runOnUiThreadIfActive {
                        classificationResultTextView.text = "Classification Result:\n$result"
                    }
                    time
                } else if (selectedRuntime == "ONNX") {
                    val time = onnxClassificationSample.processClassification(img, w, h)
                    val result = onnxClassificationSample.getLastClassificationResult()
                    runOnUiThreadIfActive {
                        classificationResultTextView.text = "Classification Result:\n$result"
                    }
                    time
                } else {
                    val time = classificationSample.processClassification(bitmap)
                    val result = classificationSample.getLastClassificationResult()
                    runOnUiThreadIfActive {
                        classificationResultTextView.text = "Classification Result:\n$result"
                    }
                    time
                }
            }

            AlgorithmType.BACKGROUND_REMOVAL -> {
                u2netSample.process(bitmap, canvas, w, h)
            }

            AlgorithmType.TOKENIZE -> {
                val inputText = tokenizerInput.orEmpty().ifEmpty { "今日、新しいiPhoneが発売されました" }
                val labels = listOf("スマートフォン", "エンタメ", "スポーツ", "政治", "科学")
                val time = miniLMv2Sample.predict(inputText, labels)
                val result = miniLMv2Sample.getLastResult()
                runOnUiThreadIfActive {
                    tokenizerOutputTextView.text = "Result:\n$result"
                }
                time
            }

            AlgorithmType.TRACKING -> {
                if (selectedRuntime == "ONNX") {
                    val detectionTime: Long
                    val detectionResults: List<DetectionResult>
                    if (useOnnxRuntime) {
                        detectionTime =
                            onnxRuntimeObjectDetectionSample.processObjectDetectionWithoutDrawing(
                                img, w, h, threshold = 0.1f, iou = 1.0f
                            )
                        detectionResults = onnxRuntimeObjectDetectionSample.getDetectionResults()
                    } else {
                        detectionTime = onnxObjectDetectionSample.processObjectDetectionWithoutDrawing(
                            img, w, h, threshold = 0.1f, iou = 1.0f
                        )
                        detectionResults = onnxObjectDetectionSample.getDetectionResults()
                    }
                    val trackingTime = trackerSample.processTrackingWithDetections(
                        canvas, paint2, w, h, detectionResults
                    )
                    val trackingInfo = trackerSample.getLastTrackingResult()
                    runOnUiThreadIfActive {
                        trackingResultTextView.text = "Tracking Results: $trackingInfo"
                    }
                    detectionTime + trackingTime
                } else {
                    // First run object detection to get detection results without drawing
                    val detectionTime = objectDetectionSample.processObjectDetectionWithoutDrawing(
                        bitmap, threshold = 0.1f, iou = 1.0f
                    )
                    val detectionResults = objectDetectionSample.getDetectionResults()
                    // Then run tracking with the detection results and draw the tracking results
                    val trackingTime = trackerSample.processTrackingWithDetections(
                        canvas, paint2, w, h, detectionResults
                    )
                    val trackingInfo = trackerSample.getLastTrackingResult()
                    runOnUiThreadIfActive {
                        trackingResultTextView.text = "Tracking Results: $trackingInfo"
                    }
                    detectionTime + trackingTime
                }
            }

            AlgorithmType.SPEECH_TO_TEXT -> {
                // Speech is handled asynchronously via the Run/Record buttons
                0
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                // Voice is handled asynchronously via the generate button
                0
            }

            AlgorithmType.LLM, AlgorithmType.MULTIMODAL_LLM -> {
                // LLM modes are handled asynchronously via the send button
                0
            }

            AlgorithmType.SPEAKER_VERIFICATION, AlgorithmType.VOICE_FILTER -> {
                // Speaker audio tools are handled asynchronously via their WAV/Mic controls.
                0
            }
        }
    }

    /** updateUIVisibilityで管理するView。可視性の初期値はすべてGONEにする。 */
    private fun managedAlgorithmViews(): List<View> = listOf(
        modeRadioGroup,
        imageView,
        cameraPreviewView,
        resultScrollView,
        classificationResultTextView,
        tokenizerInputLabel,
        tokenizerInputEditText,
        tokenizerOutputLabel,
        tokenizerOutputTextView,
        trackingResultTextView,
        transcriptTextView,
        multimodalImageView,
        llmInputLabel,
        llmInputEditText,
        llmSendButton,
        llmOutputLabel,
        llmChatContainer,
        llmStatusTextView,
        voiceInputEditText,
        voiceStatusTextView,
        voiceGenerateButton,
        voiceEnvSpinner,
        voiceResultTextView,
        speechLanguageLabel,
        speechLanguageSpinner,
        speechModeRadioGroup,
        diarizationCheckBox,
        liveModeCheckBox,
        speechRunButton,
        micRecordButton,
        speechEnvSpinner,
        waveformView,
        waveformInfoTextView,
        voiceWaveformView,
        voiceReplayButton,
        speakerProfileLabel,
        speakerProfileSpinner,
        speakerInputModeRadioGroup,
        speakerVerifyButton,
        speakerInputWavNameTextView,
        speakerRecordedActionButton,
        speakerRegisterButton,
        speakerReferencePlayButton,
        speakerWaveformView,
        speakerInputPlayButton,
        speakerStatusTextView,
        speakerResultTextView,
        voiceFilterInputWaveformLabel,
        voiceFilterPlayInputButton,
        voiceFilterOutputWaveformLabel,
        voiceFilterOutputWaveformView,
        voiceFilterPlayOutputButton,
    )

    /** アルゴリズムごとに表示するView集合。共通レイアウトは同じ集合を共有する。 */
    private fun visibleViewsByAlgorithm(isCameraMode: Boolean): Map<AlgorithmType, Set<View>> {
        val visionViews = mutableSetOf<View>(modeRadioGroup, imageView)
        if (isCameraMode) {
            visionViews.add(cameraPreviewView)
        }
        val classificationViews = visionViews + setOf<View>(
            resultScrollView,
            classificationResultTextView,
        )

        val tokenizeViews = setOf<View>(
            resultScrollView,
            tokenizerInputLabel,
            tokenizerInputEditText,
            tokenizerOutputLabel,
            tokenizerOutputTextView,
        )

        val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
        val speechViews = mutableSetOf<View>(
            resultScrollView,
            classificationResultTextView,
            transcriptTextView,
            speechModeRadioGroup,
            diarizationCheckBox,
            speechEnvSpinner,
        )
        if (isMicMode) {
            speechViews.addAll(
                listOf(
                    speechLanguageLabel,
                    speechLanguageSpinner,
                    liveModeCheckBox,
                    micRecordButton,
                    waveformView,
                    waveformInfoTextView,
                )
            )
        } else {
            speechViews.add(speechRunButton)
        }

        val llmViews = setOf<View>(
            resultScrollView,
            llmInputLabel,
            llmInputEditText,
            llmSendButton,
            llmOutputLabel,
            llmChatContainer,
            llmStatusTextView,
        )

        val multimodalViews = llmViews.toMutableSet().apply {
            add(modeRadioGroup)
            add(multimodalImageView)
            if (isCameraMode) {
                add(cameraPreviewView)
            }
        }

        val voiceViews = setOf<View>(
            resultScrollView,
            voiceInputEditText,
            voiceStatusTextView,
            voiceGenerateButton,
            voiceEnvSpinner,
            voiceResultTextView,
            voiceWaveformView,
        )

        val speakerAudioViews = setOf<View>(
            resultScrollView,
            speakerProfileLabel,
            speakerProfileSpinner,
            speakerInputModeRadioGroup,
            speakerVerifyButton,
            speakerRegisterButton,
            speakerReferencePlayButton,
            speakerWaveformView,
            speakerStatusTextView,
            speakerResultTextView,
        )

        val speakerVerificationViews = speakerAudioViews + setOf<View>(
            speakerInputWavNameTextView,
            speakerRecordedActionButton,
            speakerInputPlayButton,
        )

        val voiceFilterViews = speakerAudioViews + setOf<View>(
            speakerRecordedActionButton,
            voiceFilterInputWaveformLabel,
            voiceFilterPlayInputButton,
            voiceFilterOutputWaveformLabel,
            voiceFilterOutputWaveformView,
            voiceFilterPlayOutputButton,
        )

        return mapOf(
            AlgorithmType.POSE_ESTIMATION to visionViews,
            AlgorithmType.OBJECT_DETECTION to visionViews,
            AlgorithmType.TRACKING to visionViews,
            AlgorithmType.TOKENIZE to tokenizeViews,
            AlgorithmType.CLASSIFICATION to classificationViews,
            AlgorithmType.BACKGROUND_REMOVAL to visionViews,
            AlgorithmType.SPEECH_TO_TEXT to speechViews,
            AlgorithmType.TEXT_TO_SPEECH to voiceViews,
            AlgorithmType.LLM to llmViews,
            AlgorithmType.MULTIMODAL_LLM to multimodalViews,
            AlgorithmType.SPEAKER_VERIFICATION to speakerVerificationViews,
            AlgorithmType.VOICE_FILTER to voiceFilterViews,
        )
    }

    private fun updateUIVisibility() {
        val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton
        val visibleViews = visibleViewsByAlgorithm(isCameraMode).getValue(currentAlgorithm)
        managedAlgorithmViews().forEach { view ->
            view.visibility = if (view in visibleViews) View.VISIBLE else View.GONE
        }

        // 表示切り替え時に必要なアルゴリズム固有の初期状態を設定する。
        when (currentAlgorithm) {
            AlgorithmType.SPEECH_TO_TEXT -> {
                classificationResultTextView.text = "Speech Result: --"
            }

            AlgorithmType.LLM -> {
                llmInputEditText.setText("Hello!")
                llmChatContainer.removeAllViews()
                llmStatusTextView.text = "Status: Press Send to run"
                llmSendButton.isEnabled = true
            }

            AlgorithmType.MULTIMODAL_LLM -> {
                llmInputEditText.setText("What is in this image?")
                llmChatContainer.removeAllViews()
                llmStatusTextView.text = "Status: Press Send to run"
                llmSendButton.isEnabled = true
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                voiceWaveformView.clear()
                voiceGenerateButton.isEnabled = true
                voiceResultTextView.text = ""
                voiceStatusTextView.text = "Status: Press Generate to synthesize"
            }

            AlgorithmType.SPEAKER_VERIFICATION -> {
                speakerStatusTextView.text = "Status: Ready"
                speakerResultTextView.text = "Distance: --"
                updateSpeakerInputVisibility()
            }

            AlgorithmType.VOICE_FILTER -> {
                speakerStatusTextView.text = "Status: Ready"
                speakerResultTextView.text = "Filter result: --"
                updateVoiceFilterInputVisibility()
            }

            else -> Unit
        }

        imageView.scaleType = if (isCameraMode) {
            ImageView.ScaleType.CENTER_CROP
        } else {
            ImageView.ScaleType.FIT_CENTER
        }

        cameraSpinner.visibility =
            if (modeRadioGroup.visibility == View.VISIBLE && isCameraMode) {
                View.VISIBLE
            } else {
                View.GONE
            }
        visionRunButton.visibility =
            if (needsVisionRunButton()) View.VISIBLE else View.GONE
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
        resetRunState()
        // アルゴリズム切り替え時も、前モデルが描画した画像を残さない。
        resetVisionDisplayForModelChange()
        // アルゴリズム切り替え時に波形表示をリセット
        hideModelDownloadProgress()
        waveformView.clear()
        voiceWaveformView.clear()
        speakerWaveformView.clear()
        voiceFilterOutputWaveformView.clear()
        weSpeakerSample.stopPlayback()
        speakerPlayback = null
        voiceFilterSample.stopPlayback()
        voiceFilterPlayback = null
        latestVoiceFilterResult = null
        waveformInfoTextView.text = ""
        clearTranscript()
        updateModelSpinner()
        updateUIVisibility()

        // STT/TTSはImage/Cameraモードの選択状態に関わらずUIセットアップが必要
        // (processImageModeはImageモード時しか呼ばれないため、ここで登録する)
        when (newAlgorithm) {
            AlgorithmType.SPEECH_TO_TEXT -> {
                setupSpeechLanguageSpinner()
                setupSpeechModeRadioGroup()
                setupDiarizationCheckBox()
                setupLiveModeCheckBox()
                setupSpeechRunButton()
                setupMicRecordButton()
                // transcribeを呼ぶたびに処理時間表示を更新する
                speechSample.transcribeTimeListener = { timeMs ->
                    runOnUiThreadIfActive {
                        if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
                            processingTimeTextView.text = "Processing Time: $timeMs ms"
                        }
                    }
                }
            }
            AlgorithmType.TEXT_TO_SPEECH -> {
                setupVoiceGenerateButton()
                setupVoiceReplayButton()
                voiceInputEditText.setText(selectedVoiceModelType.defaultText)
            }
            AlgorithmType.LLM -> {
                setupLLMSendButton()
            }
            AlgorithmType.MULTIMODAL_LLM -> {
                setupMultimodalLLMSendButton()
            }
            AlgorithmType.SPEAKER_VERIFICATION -> {
                setupSpeakerVerificationControls()
            }
            AlgorithmType.VOICE_FILTER -> {
                setupVoiceFilterControls()
            }
            else -> {}
        }

        if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
            processImageMode()
        }
    }

    private fun releaseCurrentAlgorithm(includeSpeech: Boolean = true) {
        try {
            stopMicRecording(finalize = false)
            poseEstimatorSample.releasePoseEstimator()
            objectDetectionSample.releaseObjectDetection()
            classificationSample.releaseClassification()
            onnxObjectDetectionSample.releaseObjectDetection()
            onnxClassificationSample.releaseClassification()
            onnxRuntimeObjectDetectionSample.releaseObjectDetection()
            onnxRuntimeClassificationSample.releaseClassification()
            u2netSample.release()
            detrSample.release()
            miniLMv2Sample.release()
            trackerSample.releaseTracker()
            if (includeSpeech) speechSample.releaseSpeech()
            if (includeSpeech) weSpeakerSample.release()
            if (includeSpeech) voiceFilterSample.release()
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
        // モード切り替え時はRun押下からやり直す
        resetRunState()
        when (modeId) {
            R.id.imageRadioButton -> {
                updateUIVisibility()
                stopCamera()
                processImageMode()
            }

            R.id.cameraRadioButton -> {
                if (hasPermission(Manifest.permission.CAMERA)) {
                    updateUIVisibility()
                    imageView.setImageBitmap(null)
                    startCamera()
                } else {
                    Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                    ActivityCompat.requestPermissions(
                        this,
                        arrayOf(Manifest.permission.CAMERA),
                        REQUEST_CODE_CAMERA_PERMISSION,
                    )
                    modeRadioGroup.check(R.id.imageRadioButton)
                }
            }
        }
    }

    /**
     * モデルのダウンロード、初期化、進捗・結果表示を共通の非同期フローで実行する。
     * ダウンロード済みの場合も同じ経路で初期化し、成功後にonReadyをUIスレッドで呼ぶ。
     */
    private fun initializeDownloadedModelAsync(
        logName: String,
        download: (ModelDownloadListener) -> Boolean,
        initialize: () -> Boolean,
        onReady: () -> Unit = {
            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                processImageMode()
            }
        }
    ) {
        val operationId = beginModelOperation() ?: return
        runOnUiThreadIfActive {
            if (isCurrentOperation(operationId)) {
                processingTimeTextView.text = "Processing Time: -- ms"
            }
        }
        Log.i("AILIA_Main", "$logName: submitting download/init task")

        cameraExecutor.execute {
            var initializationSucceeded = false
            try {
                val downloaded = download(object : ModelDownloadListener {
                    override fun onProgress(
                        fileName: String,
                        bytesDownloaded: Long,
                        totalBytes: Long
                    ) {
                        Log.d("AILIA_Main", "$logName: downloading $fileName")
                        if (!isCurrentOperation(operationId)) return
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            showModelDownloadProgress(fileName, bytesDownloaded, totalBytes)
                        }
                    }

                    override fun onComplete() = Unit

                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            processingTimeTextView.text = "Download error: $error"
                            hideModelDownloadProgress()
                        }
                    }
                })
                Log.i("AILIA_Main", "$logName: download result=$downloaded")
                if (!downloaded) return@execute

                initializationSucceeded = initialize()
                Log.i("AILIA_Main", "$logName: initialization result=$initializationSucceeded")
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = initializationSucceeded
                    if (initializationSucceeded) {
                        processingTimeTextView.text = "Processing Time: -- ms"
                    } else {
                        processingTimeTextView.text = "Initialization failed"
                    }
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "$logName: download/init error", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    processingTimeTextView.text = "Error: ${e.message}"
                }
            } finally {
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    if (initializationSucceeded) onReady()
                }
            }
        }
    }

    private fun initializeAilia() {
        try {
            when (currentAlgorithm) {
                AlgorithmType.POSE_ESTIMATION -> {
                    initializeDownloadedModelAsync(
                        logName = if (useOnnxRuntime) {
                            "ONNX Runtime Lightweight Human Pose"
                        } else {
                            "Lightweight Human Pose"
                        },
                        download = { poseEstimatorSample.downloadModel(it) },
                        initialize = {
                            if (useOnnxRuntime) {
                                poseEstimatorSample.initializeOnnxRuntime()
                            } else {
                                poseEstimatorSample.initializePoseEstimator(selectedEnvId)
                            }
                        },
                    )
                    return
                }

                AlgorithmType.OBJECT_DETECTION -> {
                    if (selectedRuntime == "ONNX" && useDetr) {
                        initializeDownloadedModelAsync(
                            logName = if (useOnnxRuntime) "ONNX Runtime DETR" else "DETR",
                            download = { detrSample.downloadModel(it) },
                            initialize = {
                                if (useOnnxRuntime) {
                                    detrSample.initializeOnnxRuntime()
                                } else {
                                    detrSample.initialize(selectedEnvId)
                                }
                            }
                        )
                        return
                    } else if (selectedRuntime == "ONNX" && useOnnxRuntime) {
                        initializeDownloadedModelAsync(
                            logName = "ONNX Runtime YOLOX-S",
                            download = { onnxRuntimeObjectDetectionSample.downloadModel(it) },
                            initialize = {
                                onnxRuntimeObjectDetectionSample.initializeObjectDetection()
                            }
                        )
                        return
                    } else if (selectedRuntime == "ONNX") {
                        initializeDownloadedModelAsync(
                            logName = "ONNX ObjDet",
                            download = { onnxObjectDetectionSample.downloadModel(it) },
                            initialize = {
                                onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                            }
                        )
                        return
                    } else {
                        initializeDownloadedModelAsync(
                            logName = "TFLite YOLOX-S",
                            download = { objectDetectionSample.downloadModel(it) },
                            initialize = {
                                objectDetectionSample.initializeFromFile(env = selectedEnvId)
                            },
                        )
                        return
                    }
                }

                AlgorithmType.CLASSIFICATION -> {
                    if (selectedRuntime == "ONNX" && useOnnxRuntime) {
                        onnxRuntimeClassificationSample.modelType =
                            onnxClassificationSample.modelType
                        initializeDownloadedModelAsync(
                            logName = "ONNX Runtime Classification",
                            download = { onnxRuntimeClassificationSample.downloadModel(it) },
                            initialize = {
                                onnxRuntimeClassificationSample.initializeClassification()
                            }
                        )
                        return
                    } else if (selectedRuntime == "ONNX") {
                        initializeDownloadedModelAsync(
                            logName = "ONNX Classification",
                            download = { onnxClassificationSample.downloadModel(it) },
                            initialize = {
                                onnxClassificationSample.initializeClassification(selectedEnvId)
                            }
                        )
                        return
                    } else {
                        initializeDownloadedModelAsync(
                            logName = "TFLite ${classificationSample.modelType.displayName}",
                            download = { classificationSample.downloadModel(it) },
                            initialize = {
                                classificationSample.initializeFromFile(env = selectedEnvId)
                            },
                        )
                        return
                    }
                }

                AlgorithmType.TOKENIZE -> {
                    initializeDownloadedModelAsync(
                        logName = if (useOnnxRuntime) "ONNX Runtime MiniLMv2" else "MiniLMv2",
                        download = { miniLMv2Sample.downloadModel(it) },
                        initialize = {
                            if (useOnnxRuntime) {
                                miniLMv2Sample.initializeOnnxRuntime()
                            } else {
                                miniLMv2Sample.initialize(selectedEnvId)
                            }
                        },
                        onReady = { processImageMode() }
                    )
                    return
                }

                AlgorithmType.TRACKING -> {
                    if (selectedRuntime == "ONNX" && useOnnxRuntime) {
                        initializeDownloadedModelAsync(
                            logName = "ONNX Runtime Tracking",
                            download = { onnxRuntimeObjectDetectionSample.downloadModel(it) },
                            initialize = {
                                val detectorSuccess =
                                    onnxRuntimeObjectDetectionSample.initializeObjectDetection()
                                detectorSuccess && trackerSample.initializeTracker()
                            }
                        )
                        return
                    } else if (selectedRuntime == "ONNX") {
                        initializeDownloadedModelAsync(
                            logName = "ONNX Tracking",
                            download = { onnxObjectDetectionSample.downloadModel(it) },
                            initialize = {
                                val detectorSuccess =
                                    onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                                detectorSuccess && trackerSample.initializeTracker()
                            }
                        )
                        return
                    } else {
                        initializeDownloadedModelAsync(
                            logName = "TFLite Tracking",
                            download = { objectDetectionSample.downloadModel(it) },
                            initialize = {
                                val detectorSuccess =
                                    objectDetectionSample.initializeFromFile(env = selectedEnvId)
                                detectorSuccess && trackerSample.initializeTracker()
                            },
                        )
                        return
                    }
                }

                AlgorithmType.BACKGROUND_REMOVAL -> {
                    initializeDownloadedModelAsync(
                        logName = if (useOnnxRuntime) "ONNX Runtime U2Net" else "U2Net",
                        download = { u2netSample.downloadModel(it) },
                        initialize = {
                            if (useOnnxRuntime) {
                                u2netSample.initializeOnnxRuntime()
                            } else {
                                u2netSample.initialize(selectedEnvId)
                            }
                        }
                    )
                    return
                }

                AlgorithmType.SPEECH_TO_TEXT -> {
                    // モデルダウンロードはRun/Record押下時(ensureSpeechReady)まで遅延する
                    return
                }

                AlgorithmType.SPEAKER_VERIFICATION -> {
                    // モデルダウンロードはVerify/Register押下時まで遅延する
                    return
                }

                AlgorithmType.VOICE_FILTER -> {
                    // モデルダウンロードはFilter/Register押下時まで遅延する
                    return
                }

                AlgorithmType.TEXT_TO_SPEECH -> {
                    // モデルダウンロードはGenerate押下時(setupVoiceGenerateButton)まで遅延する
                    return
                }

                AlgorithmType.LLM, AlgorithmType.MULTIMODAL_LLM -> {
                    // モデルダウンロードはSend押下時(initialize*Async)まで遅延する
                    return
                }
            }
        } catch (e: Exception) {
            Log.e(
                "AILIA_Error",
                "Error initializing algorithm ${currentAlgorithm.name}: ${e.message}"
            )
        }
    }

    /** チャット吹き出しを追加する(ユーザー右寄せ/AI左寄せ) */
    private fun addChatBubble(message: String, isUser: Boolean): TextView {
        val bubble = TextView(this)
        bubble.text = message
        bubble.setTextColor(Color.BLACK)
        bubble.textSize = 14f
        bubble.setBackgroundResource(if (isUser) R.drawable.chat_bubble_user else R.drawable.chat_bubble_assistant)
        val pad = (10 * resources.displayMetrics.density).toInt()
        bubble.setPadding(pad, pad, pad, pad)
        bubble.maxWidth = (resources.displayMetrics.widthPixels * 0.75f).toInt()
        val params = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        )
        val margin = (4 * resources.displayMetrics.density).toInt()
        params.setMargins(margin, margin, margin, margin)
        params.gravity = if (isUser) android.view.Gravity.END else android.view.Gravity.START
        llmChatContainer.addView(bubble, params)
        scrollResultToBottom()
        return bubble
    }

    private fun scrollResultToBottom() {
        rootScrollView.post { rootScrollView.fullScroll(View.FOCUS_DOWN) }
    }

    /** モデルダウンロードの進捗を画面下部の共通ProgressBarへ表示する。 */
    private fun showModelDownloadProgress(
        fileName: String? = null,
        bytesDownloaded: Long = 0,
        totalBytes: Long = 0,
    ) {
        modelDownloadProgressTextView.text =
            formatDownloadProgress(fileName, bytesDownloaded, totalBytes)
        modelDownloadProgressTextView.visibility = View.VISIBLE
        modelDownloadProgressBar.visibility = View.VISIBLE
        if (totalBytes > 0) {
            modelDownloadProgressBar.isIndeterminate = false
            modelDownloadProgressBar.progress =
                ((bytesDownloaded * 100 / totalBytes).coerceIn(0, 100)).toInt()
        } else {
            modelDownloadProgressBar.isIndeterminate = true
        }
    }

    private fun hideModelDownloadProgress() {
        modelDownloadProgressTextView.text = ""
        modelDownloadProgressTextView.visibility = View.GONE
        modelDownloadProgressBar.visibility = View.GONE
        modelDownloadProgressBar.isIndeterminate = false
        modelDownloadProgressBar.progress = 0
    }

    /** Prevents model/environment changes while a native model is being initialized or used. */
    private fun beginModelOperation(): Long? {
        if (!isDownloadingModel.compareAndSet(false, true)) return null
        val operationId = operationGeneration.incrementAndGet()
        if (android.os.Looper.myLooper() == android.os.Looper.getMainLooper()) {
            setModelOperationControlsEnabled(false)
        } else {
            runOnUiThreadIfActive {
                if (isCurrentOperation(operationId)) setModelOperationControlsEnabled(false)
            }
        }
        return operationId
    }

    private fun isCurrentOperation(operationId: Long): Boolean =
        !activityDestroyed.get() && operationGeneration.get() == operationId

    private fun finishModelOperation(operationId: Long) {
        if (!isCurrentOperation(operationId)) return
        isDownloadingModel.set(false)
        setModelOperationControlsEnabled(true)
    }

    private fun setModelOperationControlsEnabled(enabled: Boolean) {
        algorithmSpinner.isEnabled = enabled
        modelSpinner.isEnabled = enabled
        envSpinner.isEnabled = enabled
        voiceEnvSpinner.isEnabled = enabled
        speechEnvSpinner.isEnabled = enabled
        modeRadioGroup.isEnabled = enabled
        for (index in 0 until modeRadioGroup.childCount) {
            modeRadioGroup.getChildAt(index).isEnabled = enabled
        }
        speechModeRadioGroup.isEnabled = enabled
        speechLanguageSpinner.isEnabled = enabled
        diarizationCheckBox.isEnabled = enabled
        liveModeCheckBox.isEnabled = enabled && !diarizationCheckBox.isChecked
        visionRunButton.isEnabled = enabled
        speakerProfileSpinner.isEnabled = enabled
        speakerInputModeRadioGroup.isEnabled = enabled
        for (index in 0 until speakerInputModeRadioGroup.childCount) {
            speakerInputModeRadioGroup.getChildAt(index).isEnabled = enabled
        }
        speakerVerifyButton.isEnabled = enabled
        speakerRecordedActionButton.isEnabled = enabled && when (currentAlgorithm) {
            AlgorithmType.SPEAKER_VERIFICATION ->
                speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerMicRadioButton &&
                    lastSpeakerMicAudio != null
            AlgorithmType.VOICE_FILTER ->
                speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerMicRadioButton &&
                    lastVoiceFilterMicAudio != null
            else -> false
        }
        speakerRegisterButton.isEnabled = enabled
        when (currentAlgorithm) {
            AlgorithmType.SPEAKER_VERIFICATION -> updateSpeakerPlaybackButtons(enabled)
            AlgorithmType.VOICE_FILTER -> updateVoiceFilterPlaybackButtons(enabled)
            else -> Unit
        }
    }

    private fun runOnUiThreadIfActive(action: () -> Unit) {
        if (activityDestroyed.get()) return
        runOnUiThread {
            if (!activityDestroyed.get()) action()
        }
    }

    private fun setupLLMSendButton() {
        llmSendButton.setOnClickListener {
            val userInput = llmInputEditText.text.toString().trim()
            if (userInput.isEmpty()) {
                llmStatusTextView.text = "Status: Please enter a message"
                return@setOnClickListener
            }
            performLLMChat(userInput)
        }
    }

    private fun performLLMChat(userInput: String) {
        val operationId = beginModelOperation() ?: return
        val needsInitialization = !isInitialized
        val modelType = selectedLLMModelType
        llmSendButton.isEnabled = false
        processingTimeTextView.text = "Processing Time: -- ms"
        llmStatusTextView.text = if (needsInitialization) "Status: Initializing..." else "Status: Generating..."
        // チャット風表示: 履歴は消さず、ユーザー発言とAI応答の吹き出しを追加する
        addChatBubble(userInput, isUser = true)
        val assistantBubble = addChatBubble("", isUser = false)
        llmInputEditText.setText("")

        cameraExecutor.execute {
            try {
                val initialized = if (needsInitialization) {
                    llmSample.modelType = modelType
                    llmSample.initialize(this@MainActivity, object : ModelDownloader.DownloadListener {
                        override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                            if (!isCurrentOperation(operationId)) return
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                showModelDownloadProgress(
                                    modelType.fileName,
                                    bytesDownloaded,
                                    totalBytes,
                                )
                            }
                        }

                        override fun onComplete(file: File) = Unit

                        override fun onError(error: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: Download error - $error"
                            }
                        }
                    })
                } else {
                    true
                }

                if (!initialized || !isCurrentOperation(operationId)) {
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        isInitialized = false
                        llmStatusTextView.text = "Status: Initialization failed"
                        hideModelDownloadProgress()
                        llmSendButton.isEnabled = true
                        finishModelOperation(operationId)
                    }
                    return@execute
                }

                runOnUiThreadIfActive {
                    if (isCurrentOperation(operationId)) llmStatusTextView.text = "Status: Generating..."
                }
                val processingTime = llmSample.chat(userInput, object : AiliaLLMSample.LLMListener {
                    override fun onToken(token: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            assistantBubble.append(token)
                            scrollResultToBottom()
                        }
                    }

                    override fun onComplete(fullResponse: String) = Unit

                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            llmStatusTextView.text = "Status: Error - $error"
                        }
                    }
                })
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = true
                    llmSendButton.isEnabled = true
                    hideModelDownloadProgress()
                    if (processingTime >= 0) {
                        llmStatusTextView.text = "Status: Complete"
                        processingTimeTextView.text = "Processing Time: ${processingTime}ms"
                    }
                    finishModelOperation(operationId)
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "LLM request failed", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    llmSendButton.isEnabled = true
                    hideModelDownloadProgress()
                    llmStatusTextView.text = "Status: Error - ${e.message}"
                    finishModelOperation(operationId)
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
            performMultimodalChat(userInput)
        }
    }

    private fun performMultimodalChat(userInput: String) {
            val operationId = beginModelOperation() ?: return
            val needsInitialization = !isInitialized
            llmSendButton.isEnabled = false
            processingTimeTextView.text = "Processing Time: -- ms"
            llmStatusTextView.text = if (needsInitialization) "Status: Initializing..." else "Status: Generating..."
            // チャット風表示: 履歴は消さず、ユーザー発言とAI応答の吹き出しを追加する
            addChatBubble(userInput, isUser = true)
            val assistantBubble = addChatBubble("", isUser = false)
            llmInputEditText.setText("")

            val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton
            val cameraFrame = if (isCameraMode) latestCameraBitmap?.copy(Bitmap.Config.ARGB_8888, false) else null

            cameraExecutor.execute {
                try {
                    val imagePath = cameraFrame?.let { bitmap ->
                        try {
                            val file = File(cacheDir, "camera_frame.png")
                            FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                            file.absolutePath
                        } finally {
                            bitmap.recycle()
                        }
                    }
                    val listener = object : AiliaMultimodalLLMSample.MultimodalLLMListener {
                        override fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                            if (!isCurrentOperation(operationId)) return
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                showModelDownloadProgress(fileName, bytesDownloaded, totalBytes)
                            }
                        }

                        override fun onStatus(status: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: $status"
                            }
                        }

                        override fun onToken(token: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                assistantBubble.append(token)
                                scrollResultToBottom()
                            }
                        }

                        override fun onComplete(fullResponse: String) = Unit

                        override fun onError(error: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: Error - $error"
                            }
                        }
                    }

                    val initialized = if (needsInitialization) {
                        multimodalLLMSample.initialize(this@MainActivity, listener)
                    } else {
                        true
                    }
                    if (!initialized || !isCurrentOperation(operationId)) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            isInitialized = false
                            llmStatusTextView.text = "Status: Initialization failed"
                            llmSendButton.isEnabled = true
                            hideModelDownloadProgress()
                            finishModelOperation(operationId)
                        }
                        return@execute
                    }

                    val processingTime = multimodalLLMSample.chatWithImage(imagePath, userInput, listener)
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        isInitialized = true
                        llmSendButton.isEnabled = true
                        hideModelDownloadProgress()
                        if (processingTime >= 0) {
                            llmStatusTextView.text = "Status: Complete"
                            processingTimeTextView.text = "Processing Time: ${processingTime}ms"
                        }
                        if (needsInitialization) loadSampleImageForMultimodal()
                        finishModelOperation(operationId)
                    }
                } catch (e: Exception) {
                    Log.e("AILIA_Main", "Multimodal LLM request failed", e)
                    cameraFrame?.takeUnless(Bitmap::isRecycled)?.recycle()
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        llmSendButton.isEnabled = true
                        hideModelDownloadProgress()
                        llmStatusTextView.text = "Status: Error - ${e.message}"
                        finishModelOperation(operationId)
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

    private fun setupVoiceReplayButton() {
        voiceReplayButton.setOnClickListener {
            val audio = voiceSample.lastAudioData ?: return@setOnClickListener
            voiceSample.playAudio(audio, voiceSample.lastAudioChannels, voiceSample.lastAudioSampleRate)
            voiceWaveformView.startPlayback(audio, voiceSample.lastAudioChannels, voiceSample.lastAudioSampleRate)
        }
    }

    private fun setupVoiceGenerateButton() {
        voiceGenerateButton.setOnClickListener {
            val inputText = voiceInputEditText.text.toString().trim()
            if (inputText.isEmpty()) {
                voiceStatusTextView.text = "Status: Please enter text to speak"
                return@setOnClickListener
            }
            val operationId = beginModelOperation() ?: return@setOnClickListener
            val needsInitialization = !isInitialized
            val modelType = selectedVoiceModelType
            val envId = selectedEnvId
            voiceGenerateButton.isEnabled = false
            voiceResultTextView.text = ""
            voiceWaveformView.clear()
            voiceStatusTextView.text =
                if (needsInitialization) "Status: Initializing..." else "Status: Generating..."
            processingTimeTextView.text = "Processing Time: -- ms"

            cameraExecutor.execute {
                try {
                    // モデルダウンロード+初期化はGenerate押下時に行う
                    val initialized = if (needsInitialization) {
                        voiceSample.modelType = modelType
                        voiceSample.initializeVoice(envId = envId, listener = object : ModelDownloadListener {
                            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                if (!isCurrentOperation(operationId)) return
                                runOnUiThreadIfActive {
                                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                    showModelDownloadProgress(fileName, bytesDownloaded, totalBytes)
                                }
                            }

                            override fun onComplete() = Unit

                            override fun onError(error: String) {
                                runOnUiThreadIfActive {
                                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                    voiceStatusTextView.text = "Status: Error - $error"
                                }
                            }
                        })
                    } else {
                        true
                    }
                    if (!initialized || !isCurrentOperation(operationId)) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            isInitialized = false
                            voiceGenerateButton.isEnabled = true
                            voiceStatusTextView.text = "Status: Initialization failed"
                            hideModelDownloadProgress()
                            finishModelOperation(operationId)
                        }
                        return@execute
                    }
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        voiceStatusTextView.text = "Status: Generating..."
                        hideModelDownloadProgress()
                    }

                    val refAudio = AudioUtil().loadRawAudio(resources.openRawResource(R.raw.reference_audio_girl))
                    // 入力テキストの日英を判定してG2Pの言語を切り替える
                    val textLang = AiliaVoiceSample.detectLanguage(inputText)
                    val inferenceTime = voiceSample.textToSpeech(
                        refAudio.audioData,
                        refAudio.channels,
                        refAudio.sampleRate,
                        "水をマレーシアから買わなくてはならない。",
                        "ja",
                        inputText,
                        textLang,
                    )
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        isInitialized = true
                        voiceGenerateButton.isEnabled = true
                        if (inferenceTime >= 0) {
                            voiceStatusTextView.text = "Status: Complete"
                            voiceResultTextView.text = "${modelType.displayName} Generated"
                            processingTimeTextView.text = "Processing Time: ${inferenceTime}ms"
                            scrollResultToBottom()
                        } else {
                            voiceStatusTextView.text = "Status: Generation failed"
                        }
                        // 合成音声の波形を再生位置に追従して表示する
                        voiceSample.lastAudioData?.let { audio ->
                            voiceWaveformView.startPlayback(audio, voiceSample.lastAudioChannels, voiceSample.lastAudioSampleRate)
                            voiceReplayButton.visibility = View.VISIBLE
                        }
                        finishModelOperation(operationId)
                    }
                } catch (e: Exception) {
                    Log.e("AILIA_Main", "Voice request failed", e)
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        voiceGenerateButton.isEnabled = true
                        hideModelDownloadProgress()
                        voiceStatusTextView.text = "Status: Error - ${e.message}"
                        finishModelOperation(operationId)
                    }
                }
            }
        }
    }

    private fun setupSpeechLanguageSpinner() {
        val languages = arrayOf("ja", "en", "auto")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, languages)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        speechLanguageSpinner.adapter = adapter

        val currentIndex = languages.indexOf(selectedSpeechLanguage)
        if (currentIndex >= 0) {
            speechLanguageSpinner.setSelection(currentIndex)
        }

        speechLanguageSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newLanguage = languages[position]
                if (newLanguage != selectedSpeechLanguage) {
                    selectedSpeechLanguage = newLanguage
                    speechSample.language = newLanguage
                    // 言語変更のため要再初期化(ダウンロードはRun/Record押下時)
                    if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
                        stopMicRecording(finalize = false)
                        speechSample.releaseSpeech()
                        isInitialized = false
                        isDownloadingModel.set(false)
                        clearTranscript()
                        classificationResultTextView.text = "Speech Result: --"
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
                    waveformView.visibility = View.GONE
                    waveformInfoTextView.visibility = View.GONE
                    liveModeCheckBox.visibility = View.GONE
                    speechLanguageLabel.visibility = View.GONE
                    speechLanguageSpinner.visibility = View.GONE
                    stopMicRecording(finalize = false)
                    // liveモード設定が変わるため要再初期化(ダウンロードはRun押下時)
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    clearTranscript()
                    classificationResultTextView.text = "Speech Result: --"
                }
                R.id.micRadioButton -> {
                    speechRunButton.visibility = View.GONE
                    micRecordButton.visibility = View.VISIBLE
                    waveformView.visibility = View.VISIBLE
                    waveformView.clear()
                    waveformInfoTextView.visibility = View.VISIBLE
                    waveformInfoTextView.text = ""
                    liveModeCheckBox.visibility = View.VISIBLE
                    speechLanguageLabel.visibility = View.VISIBLE
                    speechLanguageSpinner.visibility = View.VISIBLE
                    // liveモード設定が変わるため要再初期化(ダウンロードはRecord押下時)
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    clearTranscript()
                    classificationResultTextView.text = "Speech Result: (tap Record)"
                }
            }
        }
    }

    private fun setupDiarizationCheckBox() {
        speechSample.diarizationEnabled = diarizationCheckBox.isChecked
        updateLiveModeAvailability(diarizationCheckBox.isChecked)
        diarizationCheckBox.setOnCheckedChangeListener { _, isChecked ->
            speechSample.diarizationEnabled = isChecked
            updateLiveModeAvailability(isChecked)
            resetSpeechForOptionChange()
        }
    }

    private fun setupLiveModeCheckBox() {
        liveModeCheckBox.setOnCheckedChangeListener { _, _ ->
            if (!isUpdatingSpeechOptionChecks) {
                resetSpeechForOptionChange()
            }
        }
    }

    /** Speaker Diarizationと併用できないLive ModeをOFFにして選択不可にする。 */
    private fun updateLiveModeAvailability(diarizationEnabled: Boolean) {
        isUpdatingSpeechOptionChecks = true
        try {
            if (diarizationEnabled) {
                liveModeCheckBox.isChecked = false
            }
            liveModeCheckBox.isEnabled = !diarizationEnabled
        } finally {
            isUpdatingSpeechOptionChecks = false
        }
    }

    /** 音声認識オプション変更後、次回のRun/Recordで設定を反映する。 */
    private fun resetSpeechForOptionChange() {
        stopMicRecording(finalize = false)
        speechSample.releaseSpeech()
        isInitialized = false
        isDownloadingModel.set(false)
        clearTranscript()
        classificationResultTextView.text = "Speech Result: --"
    }

    /** 議事録風トランスクリプトに行を追記して表示を更新する(UIスレッドで呼ぶこと) */
    private fun appendTranscriptLines(lines: List<String>) {
        if (lines.isEmpty()) return
        speechIntermediateText = null
        speechTranscript.addAll(lines)
        updateTranscriptDisplay()
    }

    /** 確定済み行の末尾にIntermediate結果を一時表示する。次の結果で同じ行を置き換える。 */
    private fun showIntermediateTranscript(text: String) {
        speechIntermediateText = text.trim().takeIf { it.isNotEmpty() }
        updateTranscriptDisplay()
    }

    private fun updateTranscriptDisplay() {
        val displayLines = speechTranscript.toMutableList()
        speechIntermediateText?.let { displayLines.add("$it...") }
        transcriptTextView.text = displayLines.joinToString("\n")
        scrollResultToBottom()
    }

    private fun clearTranscript() {
        speechTranscript.clear()
        speechIntermediateText = null
        transcriptTextView.text = ""
    }

    private fun setupSpeakerVerificationControls() {
        speakerProfileLabel.text = "Reference speaker:"
        speakerRegisterButton.text = "Add Speaker"
        speakerResultTextView.text = "Distance: --"
        speakerProfileSpinner.onItemSelectedListener = null
        refreshSpeakerProfiles()
        speakerProfileSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                stopSpeakerPlayback(status = null)
                updateSpeakerPlaybackButtons()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                stopSpeakerPlayback(status = null)
                updateSpeakerPlaybackButtons()
            }
        }
        speakerInputModeRadioGroup.setOnCheckedChangeListener { _, _ ->
            stopSpeakerPlayback(status = null)
            if (weSpeakerSample.isRecording) {
                weSpeakerSample.cancelRecording()
                resetSpeakerRecordingUi()
            }
            updateSpeakerInputVisibility()
        }
        speakerVerifyButton.setOnClickListener {
            if (weSpeakerSample.isRecording) {
                if (speakerRecordingPurpose == SpeakerRecordingPurpose.VERIFY) {
                    speakerStatusTextView.text = "Status: Finishing recording..."
                    speakerVerifyButton.isEnabled = false
                    weSpeakerSample.stopRecording()
                }
                return@setOnClickListener
            }
            if (isDownloadingModel.get()) return@setOnClickListener
            if (speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerMicRadioButton) {
                ensureWeSpeakerReady { startSpeakerRecording(SpeakerRecordingPurpose.VERIFY) }
            } else {
                ensureWeSpeakerReady { verifySelectedSpeakerWav() }
            }
        }
        speakerRecordedActionButton.setOnClickListener {
            if (weSpeakerSample.isRecording || isDownloadingModel.get()) {
                return@setOnClickListener
            }
            val audio = lastSpeakerMicAudio
            if (audio == null) {
                speakerStatusTextView.text = "Status: Record microphone audio first"
                updateSpeakerInputVisibility()
                return@setOnClickListener
            }
            ensureWeSpeakerReady {
                speakerWaveformView.showAudio(audio)
                performSpeakerVerification(audio, 1, lastSpeakerMicSampleRate)
            }
        }
        speakerRegisterButton.setOnClickListener {
            if (weSpeakerSample.isRecording) {
                if (speakerRecordingPurpose == SpeakerRecordingPurpose.ENROLL) {
                    speakerStatusTextView.text = "Status: Separating speech with Silero VAD v6..."
                    speakerRegisterButton.isEnabled = false
                    weSpeakerSample.stopRecording()
                }
                return@setOnClickListener
            }
            if (isDownloadingModel.get()) return@setOnClickListener
            ensureWeSpeakerReady { startSpeakerRecording(SpeakerRecordingPurpose.ENROLL) }
        }
        speakerReferencePlayButton.setOnClickListener {
            if (speakerPlayback == SpeakerPlayback.REFERENCE) {
                stopSpeakerPlayback("Status: Playback stopped")
            } else {
                startSpeakerPlayback(SpeakerPlayback.REFERENCE)
            }
        }
        speakerInputPlayButton.setOnClickListener {
            if (speakerPlayback == SpeakerPlayback.INPUT) {
                stopSpeakerPlayback("Status: Playback stopped")
            } else {
                startSpeakerPlayback(SpeakerPlayback.INPUT)
            }
        }
        updateSpeakerInputVisibility()
    }

    private fun refreshSpeakerProfiles(selectProfileId: String? = null) {
        val customProfiles = speakerProfileStore.list()
        speakerReferences = listOf(
            SpeakerReference(
                name = "ailia-models 00001_spk1 (preset)",
                embedding = presetSpeakerEmbeddings[R.raw.wespeaker_00001_spk1],
                presetRawResource = R.raw.wespeaker_00001_spk1,
            ),
            SpeakerReference(
                name = "ailia-models 00003_spk2 (preset)",
                embedding = presetSpeakerEmbeddings[R.raw.wespeaker_00003_spk2],
                presetRawResource = R.raw.wespeaker_00003_spk2,
            ),
        ) + customProfiles.map { profile ->
            SpeakerReference(
                name = profile.name,
                embedding = profile.embedding,
                profileId = profile.id,
            )
        }
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            speakerReferences.map { it.name }.toTypedArray(),
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        speakerProfileSpinner.adapter = adapter
        if (selectProfileId != null) {
            val selectedProfile = speakerReferences.indexOfFirst { it.profileId == selectProfileId }
            if (selectedProfile >= 0) speakerProfileSpinner.setSelection(selectedProfile)
        }
    }

    private fun updateSpeakerInputVisibility() {
        if (currentAlgorithm != AlgorithmType.SPEAKER_VERIFICATION) return
        val wavMode = speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerWavRadioButton
        if (!weSpeakerSample.isRecording) {
            speakerVerifyButton.text = if (wavMode) "Verify Wav" else "Record"
        }
        speakerInputWavNameTextView.visibility = if (wavMode) View.VISIBLE else View.GONE
        val canVerifyRecorded =
            !wavMode && lastSpeakerMicAudio != null && !weSpeakerSample.isRecording
        speakerRecordedActionButton.visibility = if (wavMode) View.GONE else View.VISIBLE
        speakerRecordedActionButton.text = "Verify"
        speakerRecordedActionButton.isEnabled = canVerifyRecorded && !isDownloadingModel.get()
        updateSpeakerPlaybackButtons()
    }

    private fun startSpeakerPlayback(playback: SpeakerPlayback) {
        stopSpeakerPlayback(status = null, restoreWaveform = false)
        val reference = speakerReferences.getOrNull(speakerProfileSpinner.selectedItemPosition)
        val audio = try {
            when (playback) {
                SpeakerPlayback.REFERENCE -> loadReferenceAudio(
                    checkNotNull(reference) { "Select a reference speaker" },
                    speakerProfileStore,
                )
                SpeakerPlayback.INPUT -> loadSpeakerInputAudio()
            }
        } catch (e: Exception) {
            speakerStatusTextView.text = "Status: ${e.message}"
            updateSpeakerPlaybackButtons()
            return
        }

        if (playback == SpeakerPlayback.INPUT) {
            speakerWaveformView.startPlayback(audio.audioData, audio.channels, audio.sampleRate)
        }
        speakerPlayback = playback
        updateSpeakerPlaybackButtons()
        speakerStatusTextView.text = when (playback) {
            SpeakerPlayback.REFERENCE -> "Status: Playing reference speaker"
            SpeakerPlayback.INPUT -> if (
                speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerWavRadioButton
            ) {
                "Status: Playing default WAV"
            } else {
                "Status: Playing recorded voice"
            }
        }
        try {
            weSpeakerSample.playAudio(audio.audioData, audio.sampleRate) {
                runOnUiThreadIfActive {
                    if (speakerPlayback == playback) {
                        stopSpeakerPlayback("Status: Playback complete")
                    }
                }
            }
        } catch (e: Exception) {
            stopSpeakerPlayback("Status: Playback error - ${e.message}")
        }
    }

    private fun stopSpeakerPlayback(
        status: String?,
        restoreWaveform: Boolean = true,
    ) {
        weSpeakerSample.stopPlayback()
        speakerWaveformView.stopPlayback()
        speakerPlayback = null
        if (restoreWaveform) {
            runCatching { loadSpeakerInputAudio() }.getOrNull()?.let { audio ->
                speakerWaveformView.showAudio(audio.audioData, audio.channels)
            }
        }
        updateSpeakerPlaybackButtons()
        if (status != null && currentAlgorithm == AlgorithmType.SPEAKER_VERIFICATION) {
            speakerStatusTextView.text = status
        }
    }

    private fun updateSpeakerPlaybackButtons(controlsEnabled: Boolean = !isDownloadingModel.get()) {
        val hasReference =
            speakerReferences.getOrNull(speakerProfileSpinner.selectedItemPosition) != null
        val hasInput =
            speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerWavRadioButton ||
                lastSpeakerMicAudio != null
        when (speakerPlayback) {
            SpeakerPlayback.REFERENCE -> {
                speakerReferencePlayButton.text = "Stop"
                speakerReferencePlayButton.isEnabled = true
                speakerInputPlayButton.text = "Play"
                speakerInputPlayButton.isEnabled = false
            }
            SpeakerPlayback.INPUT -> {
                speakerReferencePlayButton.text = "Play"
                speakerReferencePlayButton.isEnabled = false
                speakerInputPlayButton.text = "Stop"
                speakerInputPlayButton.isEnabled = true
            }
            null -> {
                speakerReferencePlayButton.text = "Play"
                speakerReferencePlayButton.isEnabled = controlsEnabled && hasReference
                speakerInputPlayButton.text = "Play"
                speakerInputPlayButton.isEnabled = controlsEnabled && hasInput
            }
        }
    }

    private fun loadSpeakerInputAudio(): AudioUtil.WavFileData {
        return if (speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerWavRadioButton) {
            AudioUtil().loadRawAudio(resources.openRawResource(R.raw.wespeaker_00024_spk1))
        } else {
            AudioUtil.WavFileData(
                sampleRate = lastSpeakerMicSampleRate,
                channels = 1,
                audioData = checkNotNull(lastSpeakerMicAudio) {
                    "Record microphone audio first"
                },
            )
        }
    }

    private fun ensureWeSpeakerReady(onReady: () -> Unit) {
        if (isInitialized) {
            onReady()
            return
        }
        val operationId = beginModelOperation() ?: return
        val envId = selectedEnvId
        val useOrt = useOnnxRuntime
        speakerStatusTextView.text = "Status: Preparing WeSpeaker and Silero VAD v6..."
        processingTimeTextView.text = "Processing Time: -- ms"
        speechExecutor.execute {
            try {
                val downloaded = weSpeakerSample.downloadModel(object : ModelDownloadListener {
                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                        runOnUiThreadIfActive {
                            if (isCurrentOperation(operationId)) {
                                showModelDownloadProgress(fileName, bytesDownloaded, totalBytes)
                            }
                        }
                    }

                    override fun onComplete() = Unit

                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (isCurrentOperation(operationId)) speakerStatusTextView.text = "Status: $error"
                        }
                    }
                })
                val success = downloaded && if (useOrt) {
                    weSpeakerSample.initializeOnnxRuntime()
                } else {
                    weSpeakerSample.initialize(envId)
                }
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = success
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    if (success) {
                        speakerStatusTextView.text = "Status: Ready"
                        onReady()
                    } else {
                        speakerStatusTextView.text = "Status: Initialization failed"
                    }
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "WeSpeaker download/init failed", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    speakerStatusTextView.text = "Status: Error - ${e.message}"
                }
            }
        }
    }

    private fun verifySelectedSpeakerWav() {
        val audio = try {
            loadSpeakerInputAudio()
        } catch (e: Exception) {
            speakerStatusTextView.text = "Status: WAV error - ${e.message}"
            return
        }
        speakerWaveformView.showAudio(audio.audioData, audio.channels)
        performSpeakerVerification(audio.audioData, audio.channels, audio.sampleRate)
    }

    private fun performSpeakerVerification(audio: FloatArray, channels: Int, sampleRate: Int) {
        stopSpeakerPlayback(status = null)
        val reference = speakerReferences.getOrNull(speakerProfileSpinner.selectedItemPosition)
        if (reference == null) {
            speakerStatusTextView.text = "Status: Select a reference speaker"
            return
        }
        val operationId = beginModelOperation() ?: return
        speakerStatusTextView.text = "Status: Separating speech and comparing..."
        speakerResultTextView.text = "Distance: --"
        speechExecutor.execute {
            try {
                val referenceEmbedding = reference.embedding ?: loadPresetSpeakerEmbedding(reference)
                val result = weSpeakerSample.verify(referenceEmbedding, audio, channels, sampleRate)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    val verdict = if (result.metrics.isMatch) "MATCH" else "DIFFERENT"
                    speakerResultTextView.text = String.format(
                        Locale.ROOT,
                        "Distance: %.4f\nCosine similarity: %.4f\nNormalized similarity: %.1f%%\nSpeech: %.2f s\nResult: %s",
                        result.metrics.cosineDistance,
                        result.metrics.cosineSimilarity,
                        result.metrics.normalizedSimilarity * 100f,
                        result.speechDurationMs / 1000f,
                        verdict,
                    )
                    processingTimeTextView.text = "Processing Time: ${result.processingTimeMs} ms"
                    speakerStatusTextView.text = "Status: Complete"
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Speaker verification failed", e)
                runOnUiThreadIfActive {
                    if (isCurrentOperation(operationId)) {
                        speakerStatusTextView.text = "Status: Error - ${e.message}"
                    }
                }
            } finally {
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    finishModelOperation(operationId)
                    updateSpeakerInputVisibility()
                }
            }
        }
    }

    private fun loadPresetSpeakerEmbedding(reference: SpeakerReference): FloatArray {
        val resource = reference.presetRawResource ?: error("Reference embedding is missing")
        presetSpeakerEmbeddings[resource]?.let { return it }
        val wav = AudioUtil().loadRawAudio(resources.openRawResource(resource))
        val embedding = weSpeakerSample.extractEmbedding(wav.audioData, wav.channels, wav.sampleRate).embedding
        presetSpeakerEmbeddings[resource] = embedding
        return embedding
    }

    private fun startSpeakerRecording(purpose: SpeakerRecordingPurpose) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingAudioAction = when (purpose) {
                SpeakerRecordingPurpose.VERIFY -> PendingAudioAction.SPEAKER_VERIFY
                SpeakerRecordingPurpose.ENROLL -> PendingAudioAction.SPEAKER_ENROLL
            }
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_CODE_AUDIO_PERMISSION,
            )
            return
        }
        pendingAudioAction = null
        speakerRecordingPurpose = purpose
        stopSpeakerPlayback(status = null, restoreWaveform = false)
        if (purpose == SpeakerRecordingPurpose.VERIFY) {
            lastSpeakerMicAudio = null
            speakerResultTextView.text = "Distance: --"
        }
        speakerWaveformView.clear()
        val started = weSpeakerSample.startRecording(object : AiliaWeSpeakerSample.RecordingListener {
            override fun onWaveform(samples: FloatArray, sampleRate: Int) {
                val blocks = WaveformView.peakBlocks(samples, samples.size, sampleRate)
                runOnUiThreadIfActive { speakerWaveformView.push(blocks) }
            }

            override fun onCompleted(audio: FloatArray, sampleRate: Int) {
                when (purpose) {
                    SpeakerRecordingPurpose.VERIFY -> runOnUiThreadIfActive {
                        lastSpeakerMicAudio = audio.copyOf()
                        lastSpeakerMicSampleRate = sampleRate
                        speakerWaveformView.showAudio(audio)
                        resetSpeakerRecordingUi()
                        speakerStatusTextView.text = "Status: Recording complete. Press Verify"
                    }
                    SpeakerRecordingPurpose.ENROLL -> {
                        runOnUiThreadIfActive {
                            val name = "My Voice ${speakerProfileStore.list().size + 1}"
                            resetSpeakerRecordingUi()
                            performSpeakerEnrollment(name, audio, sampleRate)
                        }
                    }
                }
            }

            override fun onError(error: String) {
                runOnUiThreadIfActive {
                    resetSpeakerRecordingUi()
                    speakerStatusTextView.text = "Status: Recording error - $error"
                }
            }
        })
        if (started) {
            // Keep only the active Stop button enabled so a model/environment switch
            // cannot release native objects while AudioRecord is still producing data.
            setModelOperationControlsEnabled(false)
            speakerRecordedActionButton.visibility = View.VISIBLE
            speakerRecordedActionButton.isEnabled = false
            speakerStatusTextView.text = "Status: Recording (press Stop when finished)"
            if (purpose == SpeakerRecordingPurpose.VERIFY) {
                speakerVerifyButton.isEnabled = true
                speakerVerifyButton.text = "Stop"
            } else {
                speakerRegisterButton.isEnabled = true
                speakerRegisterButton.text = "Stop"
            }
        } else {
            resetSpeakerRecordingUi()
        }
    }

    private fun performSpeakerEnrollment(name: String, audio: FloatArray, sampleRate: Int) {
        val operationId = beginModelOperation() ?: return
        speakerStatusTextView.text = "Status: Separating speech and enrolling..."
        speechExecutor.execute {
            try {
                val result = weSpeakerSample.extractEmbedding(audio, 1, sampleRate)
                val profile = speakerProfileStore.add(name, result.embedding, audio, sampleRate)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    refreshSpeakerProfiles(profile.id)
                    speakerStatusTextView.text = "Status: Registered ${profile.name}"
                    speakerResultTextView.text = String.format(
                        Locale.ROOT,
                        "Registered: %s\nSpeech: %.2f s",
                        profile.name,
                        result.speechDurationMs / 1000f,
                    )
                    processingTimeTextView.text = "Processing Time: ${result.processingTimeMs} ms"
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Speaker enrollment failed", e)
                runOnUiThreadIfActive {
                    if (isCurrentOperation(operationId)) {
                        speakerStatusTextView.text = "Status: Error - ${e.message}"
                    }
                }
            } finally {
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    finishModelOperation(operationId)
                    updateSpeakerInputVisibility()
                }
            }
        }
    }

    private fun resetSpeakerRecordingUi() {
        speakerRecordingPurpose = null
        if (!isDownloadingModel.get()) setModelOperationControlsEnabled(true)
        speakerRegisterButton.text = "Add Speaker"
        updateSpeakerInputVisibility()
    }

    private fun setupVoiceFilterControls() {
        speakerProfileLabel.text = "Reference speaker:"
        speakerRegisterButton.text = "Add Speaker"
        refreshVoiceFilterProfiles()
        clearVoiceFilterResult()

        speakerProfileSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                clearVoiceFilterResult()
                if (speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerMicRadioButton) {
                    lastVoiceFilterMicAudio?.let { speakerWaveformView.showAudio(it) }
                }
                val referenceName = voiceFilterReferences.getOrNull(position)?.name
                speakerStatusTextView.text = if (referenceName != null) {
                    "Status: Reference changed to $referenceName"
                } else {
                    "Status: Select a reference speaker"
                }
                updateVoiceFilterInputVisibility()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }

        speakerInputModeRadioGroup.setOnCheckedChangeListener { _, _ ->
            if (voiceFilterSample.isRecording) {
                voiceFilterSample.cancelRecording()
                resetVoiceFilterRecordingUi()
            }
            clearVoiceFilterResult()
            val micMode =
                speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerMicRadioButton
            if (micMode) {
                lastVoiceFilterMicAudio?.let { speakerWaveformView.showAudio(it) }
                speakerStatusTextView.text = if (lastVoiceFilterMicAudio != null) {
                    "Status: Previous Mic audio is ready"
                } else {
                    "Status: Ready to record"
                }
            } else {
                speakerStatusTextView.text = "Status: Ready to filter the default WAV"
            }
            updateVoiceFilterInputVisibility()
        }
        speakerVerifyButton.setOnClickListener {
            if (voiceFilterSample.isRecording) {
                if (voiceFilterRecordingPurpose == VoiceFilterRecordingPurpose.FILTER) {
                    speakerStatusTextView.text = "Status: Finishing recording..."
                    speakerVerifyButton.isEnabled = false
                    voiceFilterSample.stopRecording()
                }
                return@setOnClickListener
            }
            if (isDownloadingModel.get()) return@setOnClickListener
            if (speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerMicRadioButton) {
                ensureVoiceFilterReady { startVoiceFilterRecording(VoiceFilterRecordingPurpose.FILTER) }
            } else {
                ensureVoiceFilterReady { filterSelectedVoiceFilterWav() }
            }
        }
        speakerRecordedActionButton.setOnClickListener {
            if (voiceFilterSample.isRecording || isDownloadingModel.get()) {
                return@setOnClickListener
            }
            val audio = lastVoiceFilterMicAudio
            if (audio == null) {
                speakerStatusTextView.text = "Status: Record microphone audio first"
                updateVoiceFilterInputVisibility()
                return@setOnClickListener
            }
            ensureVoiceFilterReady {
                speakerWaveformView.showAudio(audio)
                performVoiceFiltering(
                    audio,
                    channels = 1,
                    sampleRate = lastVoiceFilterMicSampleRate,
                    applyVad = true,
                )
            }
        }
        speakerRegisterButton.setOnClickListener {
            if (voiceFilterSample.isRecording) {
                if (voiceFilterRecordingPurpose == VoiceFilterRecordingPurpose.ENROLL) {
                    speakerStatusTextView.text = "Status: Creating speaker profile..."
                    speakerRegisterButton.isEnabled = false
                    voiceFilterSample.stopRecording()
                }
                return@setOnClickListener
            }
            if (isDownloadingModel.get()) return@setOnClickListener
            ensureVoiceFilterReady { startVoiceFilterRecording(VoiceFilterRecordingPurpose.ENROLL) }
        }
        speakerReferencePlayButton.setOnClickListener {
            if (voiceFilterPlayback == VoiceFilterPlayback.REFERENCE) {
                stopVoiceFilterPlayback("Status: Playback stopped")
            } else {
                startVoiceFilterPlayback(VoiceFilterPlayback.REFERENCE)
            }
        }
        voiceFilterPlayInputButton.setOnClickListener {
            if (voiceFilterPlayback == VoiceFilterPlayback.BEFORE) {
                stopVoiceFilterPlayback("Status: Playback stopped")
            } else {
                startVoiceFilterPlayback(VoiceFilterPlayback.BEFORE)
            }
        }
        voiceFilterPlayOutputButton.setOnClickListener {
            if (voiceFilterPlayback == VoiceFilterPlayback.AFTER) {
                stopVoiceFilterPlayback("Status: Playback stopped")
            } else {
                startVoiceFilterPlayback(VoiceFilterPlayback.AFTER)
            }
        }
        updateVoiceFilterInputVisibility()
    }

    private fun refreshVoiceFilterProfiles(selectProfileId: String? = null) {
        val customProfiles = voiceFilterProfileStore.list()
        voiceFilterReferences = listOf(
            SpeakerReference(
                name = "ref-voice (preset)",
                embedding = presetVoiceFilterEmbeddings[R.raw.voicefilter_ref_voice],
                presetRawResource = R.raw.voicefilter_ref_voice,
            ),
            SpeakerReference(
                name = "ref-ailia (preset)",
                embedding = presetVoiceFilterEmbeddings[R.raw.voicefilter_ref_ailia],
                presetRawResource = R.raw.voicefilter_ref_ailia,
            ),
        ) + customProfiles.map { profile ->
            SpeakerReference(
                name = profile.name,
                embedding = profile.embedding,
                profileId = profile.id,
            )
        }
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            voiceFilterReferences.map { it.name }.toTypedArray(),
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        speakerProfileSpinner.adapter = adapter
        if (selectProfileId != null) {
            val selectedProfile = voiceFilterReferences.indexOfFirst { it.profileId == selectProfileId }
            if (selectedProfile >= 0) speakerProfileSpinner.setSelection(selectedProfile)
        }
    }

    private fun updateVoiceFilterInputVisibility() {
        if (currentAlgorithm != AlgorithmType.VOICE_FILTER) return
        val wavMode = speakerInputModeRadioGroup.checkedRadioButtonId == R.id.speakerWavRadioButton
        if (!voiceFilterSample.isRecording) {
            speakerVerifyButton.text = if (wavMode) "Filter Wav" else "Record"
        }
        val canFilterRecorded =
            !wavMode && lastVoiceFilterMicAudio != null && !voiceFilterSample.isRecording
        speakerRecordedActionButton.visibility = if (wavMode) View.GONE else View.VISIBLE
        speakerRecordedActionButton.text = "Filter"
        speakerRecordedActionButton.isEnabled = canFilterRecorded && !isDownloadingModel.get()
        updateVoiceFilterPlaybackButtons()
    }

    private fun ensureVoiceFilterReady(onReady: () -> Unit) {
        if (isInitialized) {
            onReady()
            return
        }
        val operationId = beginModelOperation() ?: return
        val envId = selectedEnvId
        val useOrt = useOnnxRuntime
        speakerStatusTextView.text = "Status: Preparing VoiceFilter and Silero VAD v6..."
        processingTimeTextView.text = "Processing Time: -- ms"
        speechExecutor.execute {
            try {
                val downloaded = voiceFilterSample.downloadModel(object : ModelDownloadListener {
                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                        runOnUiThreadIfActive {
                            if (isCurrentOperation(operationId)) {
                                showModelDownloadProgress(fileName, bytesDownloaded, totalBytes)
                            }
                        }
                    }

                    override fun onComplete() = Unit

                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (isCurrentOperation(operationId)) speakerStatusTextView.text = "Status: $error"
                        }
                    }
                })
                val success = downloaded && if (useOrt) {
                    voiceFilterSample.initializeOnnxRuntime()
                } else {
                    voiceFilterSample.initialize(envId)
                }
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = success
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    if (success) {
                        speakerStatusTextView.text = "Status: Ready"
                        onReady()
                    } else {
                        speakerStatusTextView.text =
                            "Status: ${voiceFilterSample.lastErrorMessage ?: "Initialization failed"}"
                    }
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "VoiceFilter download/init failed", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    speakerStatusTextView.text = "Status: Error - ${e.message}"
                }
            }
        }
    }

    private fun filterSelectedVoiceFilterWav() {
        val audio = try {
            AudioUtil().loadRawAudio(resources.openRawResource(R.raw.voicefilter_mixed))
        } catch (e: Exception) {
            speakerStatusTextView.text = "Status: WAV error - ${e.message}"
            return
        }
        performVoiceFiltering(audio.audioData, audio.channels, audio.sampleRate, applyVad = false)
    }

    private fun performVoiceFiltering(
        audio: FloatArray,
        channels: Int,
        sampleRate: Int,
        applyVad: Boolean,
    ) {
        val reference = voiceFilterReferences.getOrNull(speakerProfileSpinner.selectedItemPosition)
        if (reference == null) {
            speakerStatusTextView.text = "Status: Select a target speaker"
            return
        }
        val operationId = beginModelOperation() ?: return
        clearVoiceFilterResult()
        speakerStatusTextView.text = if (applyVad) {
            "Status: Separating speech and filtering for ${reference.name}..."
        } else {
            "Status: Filtering for ${reference.name}..."
        }
        speechExecutor.execute {
            try {
                val referenceEmbedding = reference.embedding ?: loadPresetVoiceFilterEmbedding(reference)
                val result = voiceFilterSample.filter(
                    referenceEmbedding,
                    audio,
                    channels,
                    sampleRate,
                    applyVad,
                )
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    latestVoiceFilterResult = result
                    speakerWaveformView.showAudio(result.inputAudio)
                    voiceFilterOutputWaveformView.showAudio(result.outputAudio)
                    voiceFilterPlayInputButton.isEnabled = true
                    voiceFilterPlayOutputButton.isEnabled = true
                    speakerResultTextView.text = String.format(
                        Locale.ROOT,
                        "Target: %s\nDuration: %.2f s\nBefore/after audio is ready for playback",
                        reference.name,
                        result.outputAudio.size.toFloat() / result.sampleRate,
                    )
                    processingTimeTextView.text = "Processing Time: ${result.processingTimeMs} ms"
                    speakerStatusTextView.text = "Status: Complete"
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Voice filtering failed", e)
                runOnUiThreadIfActive {
                    if (isCurrentOperation(operationId)) {
                        speakerStatusTextView.text = "Status: Error - ${e.message}"
                    }
                }
            } finally {
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    finishModelOperation(operationId)
                    updateVoiceFilterInputVisibility()
                }
            }
        }
    }

    private fun loadPresetVoiceFilterEmbedding(reference: SpeakerReference): FloatArray {
        val resource = reference.presetRawResource ?: error("Target speaker embedding is missing")
        presetVoiceFilterEmbeddings[resource]?.let { return it }
        val wav = AudioUtil().loadRawAudio(resources.openRawResource(resource))
        val embedding = voiceFilterSample.createEmbedding(
            wav.audioData,
            wav.channels,
            wav.sampleRate,
        ).embedding
        presetVoiceFilterEmbeddings[resource] = embedding
        return embedding
    }

    private fun startVoiceFilterRecording(purpose: VoiceFilterRecordingPurpose) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingAudioAction = when (purpose) {
                VoiceFilterRecordingPurpose.FILTER -> PendingAudioAction.VOICE_FILTER
                VoiceFilterRecordingPurpose.ENROLL -> PendingAudioAction.VOICE_FILTER_ENROLL
            }
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_CODE_AUDIO_PERMISSION,
            )
            return
        }
        pendingAudioAction = null
        voiceFilterRecordingPurpose = purpose
        clearVoiceFilterResult()
        if (purpose == VoiceFilterRecordingPurpose.FILTER) {
            lastVoiceFilterMicAudio = null
            speakerWaveformView.clear()
        }
        val started = voiceFilterSample.startRecording(object : AiliaVoiceFilterSample.RecordingListener {
            override fun onWaveform(samples: FloatArray, sampleRate: Int) {
                val blocks = WaveformView.peakBlocks(samples, samples.size, sampleRate)
                runOnUiThreadIfActive { speakerWaveformView.push(blocks) }
            }

            override fun onCompleted(audio: FloatArray, sampleRate: Int) {
                when (purpose) {
                    VoiceFilterRecordingPurpose.FILTER -> runOnUiThreadIfActive {
                        lastVoiceFilterMicAudio = audio.copyOf()
                        lastVoiceFilterMicSampleRate = sampleRate
                        speakerWaveformView.showAudio(audio)
                        resetVoiceFilterRecordingUi()
                        speakerStatusTextView.text = "Status: Recording complete. Press Filter"
                    }
                    VoiceFilterRecordingPurpose.ENROLL -> runOnUiThreadIfActive {
                        val name = "My Voice ${voiceFilterProfileStore.list().size + 1}"
                        resetVoiceFilterRecordingUi()
                        performVoiceFilterEnrollment(name, audio, sampleRate)
                    }
                }
            }

            override fun onError(error: String) {
                runOnUiThreadIfActive {
                    resetVoiceFilterRecordingUi()
                    speakerStatusTextView.text = "Status: Recording error - $error"
                }
            }
        })
        if (started) {
            setModelOperationControlsEnabled(false)
            speakerRecordedActionButton.visibility = View.VISIBLE
            speakerRecordedActionButton.isEnabled = false
            speakerStatusTextView.text = "Status: Recording (press Stop when finished)"
            if (purpose == VoiceFilterRecordingPurpose.FILTER) {
                speakerVerifyButton.isEnabled = true
                speakerVerifyButton.text = "Stop"
            } else {
                speakerRegisterButton.isEnabled = true
                speakerRegisterButton.text = "Stop"
            }
        } else {
            resetVoiceFilterRecordingUi()
        }
    }

    private fun performVoiceFilterEnrollment(name: String, audio: FloatArray, sampleRate: Int) {
        val operationId = beginModelOperation() ?: return
        speakerStatusTextView.text = "Status: Creating VoiceFilter speaker profile..."
        speechExecutor.execute {
            try {
                val result = voiceFilterSample.createEmbedding(audio, 1, sampleRate)
                val profile = voiceFilterProfileStore.add(name, result.embedding, audio, sampleRate)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    refreshVoiceFilterProfiles(profile.id)
                    speakerStatusTextView.text = "Status: Registered ${profile.name}"
                    speakerResultTextView.text = String.format(
                        Locale.ROOT,
                        "Registered: %s\nReference speech: %.2f s",
                        profile.name,
                        result.referenceDurationMs / 1_000f,
                    )
                    processingTimeTextView.text = "Processing Time: ${result.processingTimeMs} ms"
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "VoiceFilter enrollment failed", e)
                runOnUiThreadIfActive {
                    if (isCurrentOperation(operationId)) {
                        speakerStatusTextView.text = "Status: Error - ${e.message}"
                    }
                }
            } finally {
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    finishModelOperation(operationId)
                    updateVoiceFilterInputVisibility()
                }
            }
        }
    }

    private fun clearVoiceFilterResult() {
        stopVoiceFilterPlayback(status = null, restoreWaveforms = false)
        latestVoiceFilterResult = null
        speakerWaveformView.clear()
        voiceFilterOutputWaveformView.clear()
        updateVoiceFilterPlaybackButtons()
        if (currentAlgorithm == AlgorithmType.VOICE_FILTER) {
            speakerResultTextView.text = "Filter result: --"
        }
    }

    private fun startVoiceFilterPlayback(playback: VoiceFilterPlayback) {
        stopVoiceFilterPlayback(status = null, restoreWaveforms = false)
        val audio = try {
            when (playback) {
                VoiceFilterPlayback.REFERENCE -> loadReferenceAudio(
                    checkNotNull(
                        voiceFilterReferences.getOrNull(speakerProfileSpinner.selectedItemPosition)
                    ) { "Select a reference speaker" },
                    voiceFilterProfileStore,
                )
                VoiceFilterPlayback.BEFORE -> {
                    val result = checkNotNull(latestVoiceFilterResult) { "Run VoiceFilter first" }
                    speakerWaveformView.startPlayback(result.inputAudio, 1, result.sampleRate)
                    voiceFilterOutputWaveformView.showAudio(result.outputAudio)
                    AudioUtil.WavFileData(result.sampleRate, 1, result.inputAudio)
                }
                VoiceFilterPlayback.AFTER -> {
                    val result = checkNotNull(latestVoiceFilterResult) { "Run VoiceFilter first" }
                    speakerWaveformView.showAudio(result.inputAudio)
                    voiceFilterOutputWaveformView.startPlayback(result.outputAudio, 1, result.sampleRate)
                    AudioUtil.WavFileData(result.sampleRate, 1, result.outputAudio)
                }
            }
        } catch (e: Exception) {
            speakerStatusTextView.text = "Status: ${e.message}"
            updateVoiceFilterPlaybackButtons()
            return
        }
        voiceFilterPlayback = playback
        updateVoiceFilterPlaybackButtons()
        speakerStatusTextView.text = when (playback) {
            VoiceFilterPlayback.REFERENCE -> "Status: Playing reference speaker"
            VoiceFilterPlayback.BEFORE -> "Status: Playing before filtering"
            VoiceFilterPlayback.AFTER -> "Status: Playing after filtering"
        }
        try {
            voiceFilterSample.playAudio(audio.audioData, audio.sampleRate) {
                runOnUiThreadIfActive {
                    if (voiceFilterPlayback == playback) {
                        stopVoiceFilterPlayback("Status: Playback complete")
                    }
                }
            }
        } catch (e: Exception) {
            stopVoiceFilterPlayback("Status: Playback error - ${e.message}")
        }
    }

    private fun stopVoiceFilterPlayback(
        status: String?,
        restoreWaveforms: Boolean = true,
    ) {
        voiceFilterSample.stopPlayback()
        speakerWaveformView.stopPlayback()
        voiceFilterOutputWaveformView.stopPlayback()
        voiceFilterPlayback = null
        if (restoreWaveforms) {
            latestVoiceFilterResult?.let { result ->
                speakerWaveformView.showAudio(result.inputAudio)
                voiceFilterOutputWaveformView.showAudio(result.outputAudio)
            }
        }
        updateVoiceFilterPlaybackButtons()
        if (status != null && currentAlgorithm == AlgorithmType.VOICE_FILTER) {
            speakerStatusTextView.text = status
        }
    }

    private fun updateVoiceFilterPlaybackButtons(controlsEnabled: Boolean = !isDownloadingModel.get()) {
        val hasResult = latestVoiceFilterResult != null
        val hasReference =
            voiceFilterReferences.getOrNull(speakerProfileSpinner.selectedItemPosition) != null
        when (voiceFilterPlayback) {
            VoiceFilterPlayback.REFERENCE -> {
                speakerReferencePlayButton.text = "Stop"
                speakerReferencePlayButton.isEnabled = true
                voiceFilterPlayInputButton.text = "Play Before"
                voiceFilterPlayInputButton.isEnabled = false
                voiceFilterPlayOutputButton.text = "Play After"
                voiceFilterPlayOutputButton.isEnabled = false
            }
            VoiceFilterPlayback.BEFORE -> {
                speakerReferencePlayButton.text = "Play"
                speakerReferencePlayButton.isEnabled = false
                voiceFilterPlayInputButton.text = "Stop"
                voiceFilterPlayInputButton.isEnabled = true
                voiceFilterPlayOutputButton.text = "Play After"
                voiceFilterPlayOutputButton.isEnabled = false
            }
            VoiceFilterPlayback.AFTER -> {
                speakerReferencePlayButton.text = "Play"
                speakerReferencePlayButton.isEnabled = false
                voiceFilterPlayInputButton.text = "Play Before"
                voiceFilterPlayInputButton.isEnabled = false
                voiceFilterPlayOutputButton.text = "Stop"
                voiceFilterPlayOutputButton.isEnabled = true
            }
            null -> {
                speakerReferencePlayButton.text = "Play"
                speakerReferencePlayButton.isEnabled = controlsEnabled && hasReference
                voiceFilterPlayInputButton.text = "Play Before"
                voiceFilterPlayOutputButton.text = "Play After"
                voiceFilterPlayInputButton.isEnabled = controlsEnabled && hasResult
                voiceFilterPlayOutputButton.isEnabled = controlsEnabled && hasResult
            }
        }
    }

    private fun loadReferenceAudio(
        reference: SpeakerReference,
        profileStore: SpeakerProfileStore,
    ): AudioUtil.WavFileData {
        reference.presetRawResource?.let { resource ->
            return AudioUtil().loadRawAudio(resources.openRawResource(resource))
        }
        val profileId = reference.profileId ?: error("Reference audio is unavailable")
        val sessionAudio = profileStore.loadAudio(profileId)
            ?: error("Record this speaker again to enable playback")
        return AudioUtil.WavFileData(
            sampleRate = sessionAudio.sampleRate,
            channels = 1,
            audioData = sessionAudio.audioData,
        )
    }

    private fun resetVoiceFilterRecordingUi() {
        voiceFilterRecordingPurpose = null
        if (!isDownloadingModel.get()) setModelOperationControlsEnabled(true)
        speakerRegisterButton.text = "Add Speaker"
        updateVoiceFilterInputVisibility()
    }

    /**
     * Speechモデルを必要ならダウンロード+初期化してからonReadyをUIスレッドで呼ぶ。
     * モデルダウンロードはRun/Record押下時に初めて行う。
     */
    private fun ensureSpeechReady(onReady: () -> Unit) {
        if (isInitialized) {
            onReady()
            return
        }
        val operationId = beginModelOperation() ?: return
        val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
        // LIVEフラグはLive Modeチェックボックスで有効化(マイクモード時のみ)
        val liveMode = isMicMode && liveModeCheckBox.isChecked
        val modelType = selectedSpeechModelType
        val envId = selectedEnvId
        val language = if (isMicMode) selectedSpeechLanguage else "auto"
        processingTimeTextView.text = "Processing Time: -- ms"
        speechRunButton.isEnabled = false
        micRecordButton.isEnabled = false
        speechExecutor.execute {
            try {
                val downloaded = speechSample.downloadModel(modelType, object : ModelDownloadListener {
                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                        Log.d("AILIA_Main", "Speech: downloading $fileName")
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            showModelDownloadProgress(fileName, bytesDownloaded, totalBytes)
                        }
                    }
                    override fun onComplete() {}
                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            processingTimeTextView.text = "Download error: $error"
                            hideModelDownloadProgress()
                        }
                    }
                })
                var success = false
                if (downloaded) {
                    // Wavモードは常にauto、Micモードのみ言語選択を適用する
                    Log.i("AILIA_Main", "Speech: initializing with envId=$envId, liveMode=$liveMode, language=$language")
                    speechSample.language = language
                    success = speechSample.initializeSpeech(envId, liveMode = liveMode)
                }
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = success
                    speechRunButton.isEnabled = true
                    micRecordButton.isEnabled = true
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    if (success) {
                        processingTimeTextView.text = "Processing Time: -- ms"
                        onReady()
                    } else {
                        processingTimeTextView.text = "Failed to initialize speech model"
                    }
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Speech: download/init error", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    speechRunButton.isEnabled = true
                    micRecordButton.isEnabled = true
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    processingTimeTextView.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun setupSpeechRunButton() {
        speechRunButton.setOnClickListener {
            if (isProcessing.get() || isDownloadingModel.get()) {
                return@setOnClickListener
            }
            speechRunButton.isEnabled = false
            ensureSpeechReady {
                // 認識結果が表示されるまでRunをグレーアウトする
                speechRunButton.isEnabled = false
                clearTranscript()
                classificationResultTextView.text = "Speech Result: Processing..."
                speechExecutor.execute {
                    try {
                        val audio: AudioUtil.WavFileData = AudioUtil().loadRawAudio(this.resources.openRawResource(R.raw.demo))
                        // 処理時間はtranscribeTimeListenerがtranscribeごとに更新する
                        val lines = speechSample.process(audio.audioData, audio.channels, audio.sampleRate)
                        runOnUiThreadIfActive {
                            appendTranscriptLines(lines)
                            classificationResultTextView.text =
                                if (lines.isEmpty()) "Speech Result: (no speech detected)" else "Speech Result:"
                        }
                    } catch (e: Exception) {
                        Log.e("AILIA_Main", "Speech run error", e)
                        runOnUiThreadIfActive {
                            classificationResultTextView.text = "Speech Result: Error - ${e.message}"
                        }
                    } finally {
                        runOnUiThreadIfActive {
                            speechRunButton.isEnabled = true
                        }
                    }
                }
            }
        }
    }

    private fun setupMicRecordButton() {
        micRecordButton.setOnClickListener {
            if (speechSample.isMicRecording) {
                stopMicRecording()
            } else {
                if (isDownloadingModel.get()) {
                    return@setOnClickListener
                }
                ensureSpeechReady {
                    startMicRecording()
                }
            }
        }
    }

    private fun startMicRecording() {
        if (!isInitialized) {
            classificationResultTextView.text = "Speech model not ready"
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingAudioAction = PendingAudioAction.SPEECH
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_CODE_AUDIO_PERMISSION)
            return
        }
        pendingAudioAction = null

        val started = speechSample.startMicRecording(object : AiliaSpeechSample.MicRecordingListener {
            override fun onWaveform(samples: FloatArray, sampleRate: Int) {
                val blocks = WaveformView.peakBlocks(samples, samples.size, sampleRate)
                runOnUiThreadIfActive {
                    waveformView.push(blocks)
                }
            }

            override fun onIntermediateResult(text: String) {
                runOnUiThreadIfActive {
                    showIntermediateTranscript(text)
                    classificationResultTextView.text = "Recording..."
                }
            }

            override fun onResult(
                lines: List<String>,
                isFinal: Boolean,
                processingTimeMs: Long,
            ) {
                runOnUiThreadIfActive {
                    appendTranscriptLines(lines)
                    // 処理時間はtranscribeTimeListenerがtranscribeごとに更新する
                    if (isFinal) {
                        speechIntermediateText = null
                        updateTranscriptDisplay()
                        classificationResultTextView.text =
                            if (speechTranscript.isEmpty()) "Speech Result: (no speech detected)" else "Speech Result:"
                        recTimerHandler.removeCallbacks(recTimerRunnable)
                        micRecordButton.text = "Record"
                        micRecordButton.isEnabled = true
                        waveformInfoTextView.text = ""
                    }
                }
            }

            override fun onError(error: String) {
                runOnUiThreadIfActive {
                    Log.e("AILIA_Main", "Microphone recording error: $error")
                    classificationResultTextView.text = "Speech Result: Error - $error"
                    if (!speechSample.isMicRecording) {
                        recTimerHandler.removeCallbacks(recTimerRunnable)
                        micRecordButton.text = "Record"
                        waveformInfoTextView.text = ""
                    }
                    micRecordButton.isEnabled = true
                }
            }
        })

        if (started) {
            micRecordButton.text = "Stop"
            classificationResultTextView.text = "Recording..."
            processingTimeTextView.text = "Processing Time: -- ms"
            clearTranscript()

            // 波形表示とREC経過時間の初期化
            waveformView.clear()
            recStartMs = android.os.SystemClock.elapsedRealtime()
            waveformInfoTextView.text = "● REC 00:00"
            recTimerHandler.post(recTimerRunnable)
        }
    }

    private fun stopMicRecording(finalize: Boolean = true) {
        if (speechSample.isMicRecording) {
            speechSample.stopMicRecording(finalize)
            recTimerHandler.removeCallbacks(recTimerRunnable)
            runOnUiThreadIfActive {
                micRecordButton.text = "Record"
                micRecordButton.isEnabled = !finalize
                waveformInfoTextView.text = ""
            }
        }
    }

    private fun processImageMode() {
        if (currentAlgorithm == AlgorithmType.SPEAKER_VERIFICATION ||
            currentAlgorithm == AlgorithmType.VOICE_FILTER) {
            return
        }

        // Speech to Text uses speech model spinner and Wav/Mic mode
        // モデルダウンロードはRun/Record押下時まで遅延する
        if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
            if (!isInitialized) {
                setupSpeechLanguageSpinner()
                setupSpeechModeRadioGroup()
                setupDiarizationCheckBox()
                setupLiveModeCheckBox()
                setupSpeechRunButton()
                setupMicRecordButton()
            }
            return
        }

        // 画像系/TokenizeはRunボタンが押されるまでダウンロード・実行しない
        if (needsVisionRunButton() && !runRequested) {
            // Run押下前でも入力画像は表示しておく(Imageモードのみ。カメラモードはプレビューが見える)
            if (currentAlgorithm != AlgorithmType.TOKENIZE &&
                modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                val options = Options()
                options.inScaled = false
                val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)
                imageView.setImageBitmap(personBmp)
            }
            return
        }

        // 画像系とTokenizerのモデルはすべて非同期でダウンロード・初期化する。
        // UIスレッドから初期化を開始してここで戻り、完了コールバックから推論へ進む。
        // 推論Executor内で初期化を予約すると、予約直後の未初期化状態を失敗と誤判定する。
        if (needsVisionRunButton() && !isInitialized) {
            initializeAilia()
            return
        }

        // TEXT_TO_SPEECHはGenerate押下時にダウンロード+初期化する
        if (currentAlgorithm == AlgorithmType.TEXT_TO_SPEECH) {
            if (!isInitialized) {
                setupVoiceGenerateButton()
            }
            return
        }

        // LLMはSend押下時にダウンロード+初期化する
        if (currentAlgorithm == AlgorithmType.LLM) {
            return
        }

        // MultimodalLLMはperson画像の表示のみ(ダウンロードはSend押下時)
        if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
            val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
            if (isImageMode) {
                val options = BitmapFactory.Options()
                options.inScaled = false
                val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)
                multimodalImageView.setImageBitmap(personBmp)
            }
            return
        }

        if (!isProcessing.compareAndSet(false, true)) return
        val tokenizerInput = tokenizerInputEditText.text.toString()
        if (currentAlgorithm == AlgorithmType.TOKENIZE) {
            // Tokenizerは1回のRunにつき1回だけ実行する。初期化完了後にここへ
            // 到達した時点で要求を消費し、別のUI更新から再実行されないようにする。
            resetRunState()
        }
        setModelOperationControlsEnabled(false)
        cameraExecutor.execute { runImageModeInference(tokenizerInput) }
    }

    /** Runs bundled-model initialization and still-image inference away from the UI thread. */
    private fun runImageModeInference(tokenizerInput: String) {
        try {
            if (!isInitialized) {
                Log.w("AILIA_Main", "Skipping image inference because initialization is pending")
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
            val processingTime = processAlgorithm(img, personBmp, canvas, w, h, tokenizerInput)

            runOnUiThreadIfActive {
                if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                    imageView.setImageBitmap(bitmap)
                }
                var timeText = "Processing Time: ${processingTime}ms"
                when (currentAlgorithm) {
                    AlgorithmType.TRACKING -> {
                        timeText += "\n${trackerSample.getLastTrackingResult()}"
                    }
                    else -> {}
                }
                processingTimeTextView.text = timeText
            }

        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error in image mode: ${e.message}")
            runOnUiThreadIfActive {
                processingTimeTextView.text = "Processing Error: ${e.message}"
            }
        } finally {
            isProcessing.set(false)
            runOnUiThreadIfActive {
                if (!isDownloadingModel.get()) setModelOperationControlsEnabled(true)
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

            val cameraSelector = cameraChoices[selectedCameraIndex].selector

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

    private inner class CameraFrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            // MultimodalLLMはプレビュー表示とフレーム保持のみ行う(推論はSend押下時)
            if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
                processCameraFrame(image)
                image.close()
                return
            }
            // 非画像系は、直前のCamera Modeが選択されたままでも解析しない。
            // TTSなどが表示したProcessing Timeを0ms/FPS:0で上書きするのを防ぐ。
            if (!usesVisionCameraInference(currentAlgorithm)) {
                image.close()
                return
            }
            // 画像系はRunボタンが押されるまでダウンロード・実行しない
            if (needsVisionRunButton() && !runRequested) {
                image.close()
                return
            }
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
            val displayGeneration = visionDisplayGeneration.get()

            try {
                // フロントカメラはPreviewViewと同じ鏡像表示になるよう左右反転する
                val isFrontCamera = cameraChoices[selectedCameraIndex].selector == CameraSelector.DEFAULT_FRONT_CAMERA
                val imageUtil = ImageUtil()
                var camera_bitmap = imageUtil.imageProxyToBitmap(image, mirror = isFrontCamera)
                camera_bitmap = imageUtil.cropToSquare(camera_bitmap)

                val img = ImageUtil().loadRawImage(camera_bitmap)
                val w = camera_bitmap.width
                val h = camera_bitmap.height

                Log.i("AILIA_Main", "${w} ${h}")

                val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(bitmap)
                canvas.drawBitmap(camera_bitmap, 0f, 0f, null)

                val processingTime = processAlgorithm(img, bitmap, canvas, w, h)

                runOnUiThreadIfActive {
                    // モデル/アルゴリズム切り替え前に開始したフレームは表示しない。
                    if (displayGeneration != visionDisplayGeneration.get()) {
                        return@runOnUiThreadIfActive
                    }
                    // Stop押下後に処理中だったフレームの結果で表示を上書きしない
                    if (needsVisionRunButton() && !runRequested) {
                        return@runOnUiThreadIfActive
                    }
                    if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
                        latestCameraBitmap = camera_bitmap
                        multimodalImageView.setImageBitmap(camera_bitmap)
                    } else if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                        imageView.setImageBitmap(bitmap)
                    }

                    val fps = if (processingTime > 0) 1000 / processingTime else 0
                    var timeText = "Processing Time: ${processingTime}ms - FPS: $fps"
                    when (currentAlgorithm) {
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
                    runOnUiThreadIfActive {
                        executeAlgorithmSwitch(pendingAlgorithm)
                        isWaitAlgorithmSwitch.set(false)
                    }
                }

                pendingModeSwitch?.let { pendingMode ->
                    pendingModeSwitch = null
                    isWaitModeSwitch.set(true)
                    runOnUiThreadIfActive {
                        executeModeSwitch(pendingMode)
                        isWaitModeSwitch.set(false)
                    }
                }
            }
        }
    }

    private fun hasPermission(permission: String): Boolean =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
        when (requestCode) {
            REQUEST_CODE_CAMERA_PERMISSION -> if (granted) {
                modeRadioGroup.check(R.id.cameraRadioButton)
            } else {
                Toast.makeText(this, "Camera permission was not granted.", Toast.LENGTH_SHORT).show()
            }
            REQUEST_CODE_AUDIO_PERMISSION -> {
                val action = pendingAudioAction
                pendingAudioAction = null
                if (granted) {
                    when (action) {
                        PendingAudioAction.SPEAKER_VERIFY ->
                            startSpeakerRecording(SpeakerRecordingPurpose.VERIFY)
                        PendingAudioAction.SPEAKER_ENROLL ->
                            startSpeakerRecording(SpeakerRecordingPurpose.ENROLL)
                        PendingAudioAction.VOICE_FILTER ->
                            startVoiceFilterRecording(VoiceFilterRecordingPurpose.FILTER)
                        PendingAudioAction.VOICE_FILTER_ENROLL ->
                            startVoiceFilterRecording(VoiceFilterRecordingPurpose.ENROLL)
                        PendingAudioAction.SPEECH, null -> startMicRecording()
                    }
                } else {
                    Toast.makeText(this, "Microphone permission was not granted.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroy() {
        activityDestroyed.set(true)
        lastSpeakerMicAudio = null
        lastVoiceFilterMicAudio = null
        operationGeneration.incrementAndGet()
        isDownloadingModel.set(false)
        llmSample.cancelGeneration()
        multimodalLLMSample.cancelGeneration()
        recTimerHandler.removeCallbacksAndMessages(null)
        stopMicRecording(finalize = false)
        stopCamera()

        // Native objects are released after already queued work on the same executor.
        // This avoids destroying a model while its inference call is still running.
        speechExecutor.execute {
            speechSample.releaseSpeech()
            weSpeakerSample.release()
            voiceFilterSample.release()
        }
        speechExecutor.shutdown()
        cameraExecutor.execute { releaseCurrentAlgorithm(includeSpeech = false) }
        cameraExecutor.shutdown()
        super.onDestroy()
    }

}

internal fun formatDownloadProgress(
    fileName: String?,
    bytesDownloaded: Long,
    totalBytes: Long,
): String {
    val bytesPerMegabyte = 1024.0 * 1024.0
    val downloadedMb = bytesDownloaded.coerceAtLeast(0) / bytesPerMegabyte
    val sizeText = if (totalBytes > 0) {
        val totalMb = totalBytes / bytesPerMegabyte
        String.format(Locale.ROOT, "%.1f / %.1f MB", downloadedMb, totalMb)
    } else {
        String.format(Locale.ROOT, "%.1f MB", downloadedMb)
    }
    return listOfNotNull("Downloading", fileName?.takeIf { it.isNotBlank() }, sizeText)
        .joinToString(" ")
}
