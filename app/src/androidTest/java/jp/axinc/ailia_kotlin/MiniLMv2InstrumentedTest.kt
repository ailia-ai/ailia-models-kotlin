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
 * Instrumented tests for MiniLMv2 zero-shot classification.
 * Tests download, initialization, and inference with ailia Tokenizer + ailia SDK.
 */
@RunWith(AndroidJUnit4::class)
class MiniLMv2InstrumentedTest {
    companion object {
        private const val TAG = "MiniLMv2Test"

        init {
            System.loadLibrary("ailia")
        }
    }

    private lateinit var miniLMv2Sample: AiliaMiniLMv2Sample
    private lateinit var modelDir: String

    @Before
    fun setup() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        modelDir = context.filesDir.absolutePath
        miniLMv2Sample = AiliaMiniLMv2Sample()
        miniLMv2Sample.modelDir = modelDir
    }

    @Test
    fun testDownloadModel() {
        Log.i(TAG, "=== Testing MiniLMv2 model download ===")

        val downloaded = miniLMv2Sample.downloadModel(object : AiliaMiniLMv2Sample.DownloadListener {
            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                Log.i(TAG, "Downloading $fileName: $percent%")
            }
            override fun onComplete() {
                Log.i(TAG, "MiniLMv2 model download complete")
            }
            override fun onError(error: String) {
                Log.e(TAG, "MiniLMv2 model download error: $error")
            }
        })
        assertTrue("MiniLMv2 model download should succeed", downloaded)

        // Verify model files exist
        val onnxFile = File("$modelDir/minilm_l12.onnx")
        assertTrue("ONNX model file should exist", onnxFile.exists())
        assertTrue("ONNX model file should be readable", onnxFile.canRead())
        assertTrue("ONNX model file should have non-zero size", onnxFile.length() > 0)

        val protoFile = File("$modelDir/minilm_l12.onnx.prototxt")
        assertTrue("Proto file should exist", protoFile.exists())
        assertTrue("Proto file should be readable", protoFile.canRead())
        assertTrue("Proto file should have non-zero size", protoFile.length() > 0)

        val tokenizerFile = File("$modelDir/sentencepiece.bpe.model")
        assertTrue("Tokenizer file should exist", tokenizerFile.exists())
        assertTrue("Tokenizer file should be readable", tokenizerFile.canRead())
        assertTrue("Tokenizer file should have non-zero size", tokenizerFile.length() > 0)

        Log.i(TAG, "MiniLMv2 model download OK")
    }

    @Test
    fun testInitialize() {
        Log.i(TAG, "=== Testing MiniLMv2 initialization ===")

        val downloaded = miniLMv2Sample.downloadModel()
        assertTrue("Download should succeed", downloaded)

        val initialized = miniLMv2Sample.initialize(0)
        assertTrue("MiniLMv2 initialization should succeed", initialized)

        miniLMv2Sample.release()
        Log.i(TAG, "MiniLMv2 initialization OK")
    }

    @Test
    fun testPredictJapanese() {
        Log.i(TAG, "=== Testing MiniLMv2 Japanese zero-shot classification ===")

        val downloaded = miniLMv2Sample.downloadModel()
        assertTrue("Download should succeed", downloaded)
        val initialized = miniLMv2Sample.initialize(0)
        assertTrue("Init should succeed", initialized)

        val sentence = "今日、新しいiPhoneが発売されました"
        val labels = listOf("スマートフォン", "エンタメ", "スポーツ", "政治", "科学")
        val startTime = System.nanoTime()
        val processingTime = miniLMv2Sample.predict(sentence, labels)
        val elapsed = (System.nanoTime() - startTime) / 1000000
        Log.i(TAG, "Prediction completed in ${elapsed}ms (returned ${processingTime}ms)")

        assertTrue("Processing time should be positive", processingTime > 0)

        val result = miniLMv2Sample.getLastResult()
        Log.i(TAG, "Result:\n$result")
        assertTrue("Result should not be empty", result.isNotEmpty())

        // The top label should be "スマートフォン" for an iPhone-related sentence
        val lines = result.split("\n")
        assertTrue("Result should have entries for all labels", lines.size == labels.size)
        val topLabel = lines[0].split(":")[0].trim()
        Log.i(TAG, "Top label: $topLabel")
        assertEquals("Top label should be スマートフォン", "スマートフォン", topLabel)

        miniLMv2Sample.release()
        Log.i(TAG, "MiniLMv2 Japanese prediction OK")
    }

    @Test
    fun testPredictEnglish() {
        Log.i(TAG, "=== Testing MiniLMv2 English zero-shot classification ===")

        val downloaded = miniLMv2Sample.downloadModel()
        assertTrue("Download should succeed", downloaded)
        val initialized = miniLMv2Sample.initialize(0)
        assertTrue("Init should succeed", initialized)

        val sentence = "The player scored a goal in the last minute of the match."
        val labels = listOf("sports", "politics", "technology", "entertainment", "science")
        val processingTime = miniLMv2Sample.predict(sentence, labels)
        Log.i(TAG, "Prediction completed in ${processingTime}ms")

        assertTrue("Processing time should be positive", processingTime > 0)

        val result = miniLMv2Sample.getLastResult()
        Log.i(TAG, "Result:\n$result")
        assertTrue("Result should not be empty", result.isNotEmpty())

        // The top label should be "sports" for a goal-scoring sentence
        val topLabel = result.split("\n")[0].split(":")[0].trim()
        Log.i(TAG, "Top label: $topLabel")
        assertEquals("Top label should be sports", "sports", topLabel)

        miniLMv2Sample.release()
        Log.i(TAG, "MiniLMv2 English prediction OK")
    }

    @Test
    fun testPredictCustomTemplate() {
        Log.i(TAG, "=== Testing MiniLMv2 with custom hypothesis template ===")

        val downloaded = miniLMv2Sample.downloadModel()
        assertTrue("Download should succeed", downloaded)
        val initialized = miniLMv2Sample.initialize(0)
        assertTrue("Init should succeed", initialized)

        val sentence = "I love this movie, it was amazing!"
        val labels = listOf("positive", "negative", "neutral")
        val processingTime = miniLMv2Sample.predict(
            sentence, labels,
            hypothesisTemplate = "The sentiment of this text is {}."
        )
        Log.i(TAG, "Prediction completed in ${processingTime}ms")

        assertTrue("Processing time should be positive", processingTime > 0)

        val result = miniLMv2Sample.getLastResult()
        Log.i(TAG, "Result:\n$result")
        assertTrue("Result should not be empty", result.isNotEmpty())

        // The top label should be "positive" for a positive sentiment sentence
        val topLabel = result.split("\n")[0].split(":")[0].trim()
        Log.i(TAG, "Top label: $topLabel")
        assertEquals("Top label should be positive", "positive", topLabel)

        miniLMv2Sample.release()
        Log.i(TAG, "MiniLMv2 custom template prediction OK")
    }

    @Test
    fun testMultiplePredictions() {
        Log.i(TAG, "=== Testing MiniLMv2 multiple sequential predictions ===")

        val downloaded = miniLMv2Sample.downloadModel()
        assertTrue("Download should succeed", downloaded)
        val initialized = miniLMv2Sample.initialize(0)
        assertTrue("Init should succeed", initialized)

        // First prediction
        val time1 = miniLMv2Sample.predict(
            "東京オリンピックで日本がメダルを獲得",
            listOf("スポーツ", "政治", "経済")
        )
        assertTrue("First prediction time should be positive", time1 > 0)
        val result1 = miniLMv2Sample.getLastResult()
        Log.i(TAG, "Prediction 1:\n$result1")
        assertTrue("First result should not be empty", result1.isNotEmpty())

        // Second prediction
        val time2 = miniLMv2Sample.predict(
            "新しい量子コンピュータが開発された",
            listOf("テクノロジー", "スポーツ", "音楽")
        )
        assertTrue("Second prediction time should be positive", time2 > 0)
        val result2 = miniLMv2Sample.getLastResult()
        Log.i(TAG, "Prediction 2:\n$result2")
        assertTrue("Second result should not be empty", result2.isNotEmpty())
        // Result should be different from first
        assertNotEquals("Results should be different for different inputs", result1, result2)

        miniLMv2Sample.release()
        Log.i(TAG, "MiniLMv2 multiple predictions OK")
    }

    @Test
    fun testReleaseAndReinitialize() {
        Log.i(TAG, "=== Testing MiniLMv2 release and reinitialize ===")

        val downloaded = miniLMv2Sample.downloadModel()
        assertTrue("Download should succeed", downloaded)

        // First init + predict
        val init1 = miniLMv2Sample.initialize(0)
        assertTrue("First init should succeed", init1)
        val time1 = miniLMv2Sample.predict("Test sentence", listOf("label1", "label2"))
        assertTrue("First prediction should succeed", time1 > 0)

        // Release
        miniLMv2Sample.release()

        // Re-init + predict
        val init2 = miniLMv2Sample.initialize(0)
        assertTrue("Second init should succeed", init2)
        val time2 = miniLMv2Sample.predict("Another test", listOf("label1", "label2"))
        assertTrue("Second prediction should succeed", time2 > 0)

        miniLMv2Sample.release()
        Log.i(TAG, "MiniLMv2 release and reinitialize OK")
    }
}
