package jp.axinc.ailia_kotlin

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Instrumented tests for Speech model selection feature.
 * Tests download, initialization, and inference for each SpeechModelType.
 */
@RunWith(AndroidJUnit4::class)
class SpeechInstrumentedTest {
    companion object {
        private const val TAG = "SpeechTest"
    }

    private lateinit var speechSample: AiliaSpeechSample
    private lateinit var modelDir: String

    @Before
    fun setup() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        modelDir = context.filesDir.absolutePath
        speechSample = AiliaSpeechSample()
        speechSample.modelDir = modelDir
    }

    private fun loadDemoAudio(): AudioUtil.WavFileData {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val resId = context.resources.getIdentifier("demo", "raw", context.packageName)
        assertTrue("Demo audio resource should exist", resId != 0)
        return AudioUtil().loadRawAudio(context.resources.openRawResource(resId))
    }

    private fun testModelDownloadAndInit(modelType: SpeechModelType) {
        Log.i(TAG, "=== Testing ${modelType.displayName} download and init ===")

        // Download
        val downloaded = speechSample.downloadModel(modelType, object : AiliaSpeechSample.DownloadListener {
            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                Log.i(TAG, "Downloading $fileName: $percent%")
            }
            override fun onComplete() {
                Log.i(TAG, "${modelType.displayName} download complete")
            }
            override fun onError(error: String) {
                Log.e(TAG, "${modelType.displayName} download error: $error")
            }
        })
        assertTrue("${modelType.displayName} download should succeed", downloaded)

        // Verify encoder file exists
        val encoderFile = File("$modelDir/${modelType.encoderFileName}")
        assertTrue("Encoder file should exist: ${encoderFile.absolutePath}", encoderFile.exists())
        assertTrue("Encoder file should be readable", encoderFile.canRead())
        assertTrue("Encoder file should have non-zero size", encoderFile.length() > 0)

        // Verify decoder file exists (if needed)
        if (modelType.needsDecoder) {
            val decoderFile = File("$modelDir/${modelType.decoderFileName}")
            assertTrue("Decoder file should exist: ${decoderFile.absolutePath}", decoderFile.exists())
            assertTrue("Decoder file should be readable", decoderFile.canRead())
            assertTrue("Decoder file should have non-zero size", decoderFile.length() > 0)
        }

        // Initialize
        val initialized = speechSample.initializeSpeech()
        assertTrue("${modelType.displayName} initialization should succeed", initialized)

        Log.i(TAG, "${modelType.displayName} download and init OK")
    }

    private fun testModelInference(modelType: SpeechModelType) {
        Log.i(TAG, "=== Testing ${modelType.displayName} inference ===")

        // Download and initialize
        val downloaded = speechSample.downloadModel(modelType)
        assertTrue("Download should succeed", downloaded)
        val initialized = speechSample.initializeSpeech()
        assertTrue("Init should succeed", initialized)

        // Load demo audio
        val audio = loadDemoAudio()
        Log.i(TAG, "Audio: samples=${audio.audioData.size}, channels=${audio.channels}, sampleRate=${audio.sampleRate}")

        // Process
        val startTime = System.nanoTime()
        val result = speechSample.process(audio.audioData, audio.channels, audio.sampleRate)
        val elapsed = (System.nanoTime() - startTime) / 1000000
        Log.i(TAG, "${modelType.displayName} inference completed in ${elapsed}ms")
        Log.i(TAG, "Result: '$result'")

        assertTrue("${modelType.displayName} result should not be empty", result.isNotEmpty())

        // Cleanup
        speechSample.releaseSpeech()
    }

    @Test
    fun testWhisperTiny_downloadAndInit() {
        testModelDownloadAndInit(SpeechModelType.WHISPER_TINY)
        speechSample.releaseSpeech()
    }

    @Test
    fun testWhisperTiny_inference() {
        testModelInference(SpeechModelType.WHISPER_TINY)
    }

    @Test
    fun testWhisperBase_downloadAndInit() {
        testModelDownloadAndInit(SpeechModelType.WHISPER_BASE)
        speechSample.releaseSpeech()
    }

    @Test
    fun testWhisperBase_inference() {
        testModelInference(SpeechModelType.WHISPER_BASE)
    }

    @Test
    fun testWhisperSmall_downloadAndInit() {
        testModelDownloadAndInit(SpeechModelType.WHISPER_SMALL)
        speechSample.releaseSpeech()
    }

    @Test
    fun testWhisperSmall_inference() {
        testModelInference(SpeechModelType.WHISPER_SMALL)
    }

    @Test
    fun testSenseVoiceSmall_downloadAndInit() {
        testModelDownloadAndInit(SpeechModelType.SENSEVOICE_SMALL)
        speechSample.releaseSpeech()
    }

    @Test
    fun testSenseVoiceSmall_inference() {
        testModelInference(SpeechModelType.SENSEVOICE_SMALL)
    }

    @Test
    fun testModelSwitch_whisperTinyToBase() {
        Log.i(TAG, "=== Testing model switch: Whisper Tiny -> Whisper Base ===")

        // Initialize with Whisper Tiny
        val downloaded1 = speechSample.downloadModel(SpeechModelType.WHISPER_TINY)
        assertTrue("Tiny download should succeed", downloaded1)
        val init1 = speechSample.initializeSpeech()
        assertTrue("Tiny init should succeed", init1)

        val audio = loadDemoAudio()
        val result1 = speechSample.process(audio.audioData, audio.channels, audio.sampleRate)
        assertTrue("Tiny result should not be empty", result1.isNotEmpty())
        Log.i(TAG, "Tiny result: '$result1'")

        // Switch to Whisper Base
        speechSample.releaseSpeech()
        val downloaded2 = speechSample.downloadModel(SpeechModelType.WHISPER_BASE)
        assertTrue("Base download should succeed", downloaded2)
        val init2 = speechSample.initializeSpeech()
        assertTrue("Base init should succeed", init2)

        val result2 = speechSample.process(audio.audioData, audio.channels, audio.sampleRate)
        assertTrue("Base result should not be empty", result2.isNotEmpty())
        Log.i(TAG, "Base result: '$result2'")

        speechSample.releaseSpeech()
    }

    @Test
    fun testLiveMode_whisperTiny() {
        Log.i(TAG, "=== Testing live mode: Whisper Tiny ===")

        val downloaded = speechSample.downloadModel(SpeechModelType.WHISPER_TINY)
        assertTrue("Download should succeed", downloaded)
        val initialized = speechSample.initializeSpeech(liveMode = true)
        assertTrue("Live mode init should succeed", initialized)

        // Push some audio chunks (simulate live input)
        val audio = loadDemoAudio()
        val chunkSize = audio.sampleRate // 1 second chunks
        var offset = 0
        var lastResult = ""

        while (offset < audio.audioData.size) {
            val end = minOf(offset + chunkSize, audio.audioData.size)
            val chunk = audio.audioData.copyOfRange(offset, end)
            val result = speechSample.pushLiveAudio(chunk, audio.channels, audio.sampleRate)
            if (result.isNotEmpty()) {
                lastResult = result
                Log.i(TAG, "Live intermediate result: '$result'")
            }
            offset = end
        }

        // Finalize
        val finalResult = speechSample.finalizeLiveAudio()
        Log.i(TAG, "Live final result: '$finalResult'")

        // At least one of the results should be non-empty
        assertTrue(
            "Live mode should produce some result",
            lastResult.isNotEmpty() || finalResult.isNotEmpty()
        )

        speechSample.releaseSpeech()
    }
}
