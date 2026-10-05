package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class OnnxRuntimeInstrumentedTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelDirectory = ModelDownloader.modelDirectory(context)

    @Test
    fun mobileNetV2RunsWithOnnxRuntime() {
        assertClassificationRuns(OnnxClassificationModelType.MOBILENETV2)
    }

    @Test
    fun resNet50RunsWithOnnxRuntime() {
        assertClassificationRuns(OnnxClassificationModelType.RESNET50)
    }

    @Test
    fun vitB16RunsWithOnnxRuntime() {
        assertClassificationRuns(OnnxClassificationModelType.VIT_B16)
    }

    @Test
    fun yoloxRunsWithOnnxRuntime() {
        val sample = OnnxRuntimeObjectDetectionSample(modelDirectory)
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeObjectDetection())
        try {
            assertNotNull(sample.detect(grayRgba(640, 640), 640, 640))
        } finally {
            sample.releaseObjectDetection()
        }
    }

    @Test
    fun detrRunsWithOnnxRuntime() {
        val sample = AiliaDetrSample(modelDirectory)
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeOnnxRuntime())
        val bitmap = personBitmap()
        try {
            assertNotNull(sample.detect(bitmap))
        } finally {
            sample.release()
            bitmap.recycle()
        }
    }

    @Test
    fun u2NetRunsWithOnnxRuntime() {
        val sample = AiliaU2NetSample(modelDirectory)
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeOnnxRuntime())
        val bitmap = personBitmap()
        try {
            val result = sample.predictMask(bitmap)
            assertNotNull(result)
            assertEquals(320 * 320, result!!.value.values.size)
        } finally {
            sample.release()
            bitmap.recycle()
        }
    }

    @Test
    fun miniLmRunsWithOnnxRuntime() {
        val sample = AiliaMiniLMv2Sample(modelDirectory)
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeOnnxRuntime())
        try {
            val time = sample.predict(
                "A new smartphone was released today.",
                listOf("smartphone", "sports", "politics"),
            )
            assertTrue(time >= 0)
            assertTrue(sample.getLastResult().contains("smartphone"))
        } finally {
            sample.release()
        }
    }

    @Test
    fun poseRunsWithOnnxRuntime() {
        val sample = AiliaPoseEstimatorSample(modelDirectory)
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeOnnxRuntime())
        val source = personBitmap()
        val rgba = ImageUtil().loadRawImage(source)
        val output = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        try {
            val time = sample.processPoseEstimation(
                rgba,
                Canvas(output),
                Paint(),
                source.width,
                source.height,
            )
            assertTrue(time >= 0)
            assertTrue(sample.getLastOnnxPoseCount() > 0)
        } finally {
            sample.releasePoseEstimator()
            source.recycle()
            output.recycle()
        }
    }

    @Test
    fun weSpeakerAndVadRunWithOnnxRuntime() {
        val sample = AiliaWeSpeakerSample(modelDirectory)
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeOnnxRuntime())
        val wav = AudioUtil().loadRawAudio(context.resources.openRawResource(R.raw.wespeaker_00001_spk1))
        try {
            val result = sample.extractEmbedding(wav.audioData, wav.channels, wav.sampleRate)
            assertEquals(256, result.embedding.size)
            assertTrue(result.processingTimeMs >= 0)
        } finally {
            sample.release()
        }
    }

    @Test
    fun voiceFilterModelsRunWithOnnxRuntime() {
        val sample = AiliaVoiceFilterSample(modelDirectory)
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeOnnxRuntime())
        val reference = AudioUtil().loadRawAudio(
            context.resources.openRawResource(R.raw.voicefilter_ref_ailia),
        )
        val mixed = AudioUtil().loadRawAudio(context.resources.openRawResource(R.raw.voicefilter_mixed))
        try {
            val embedding = sample.createEmbedding(
                reference.audioData,
                reference.channels,
                reference.sampleRate,
            ).embedding
            assertEquals(256, embedding.size)
            val result = sample.filter(
                embedding,
                mixed.audioData,
                mixed.channels,
                mixed.sampleRate,
            )
            assertTrue(result.outputAudio.isNotEmpty())
        } finally {
            sample.release()
        }
    }

    private fun personBitmap(): Bitmap = BitmapFactory.decodeResource(
        context.resources,
        R.raw.person,
        BitmapFactory.Options().apply { inScaled = false },
    )

    private fun assertClassificationRuns(modelType: OnnxClassificationModelType) {
        val sample = OnnxRuntimeClassificationSample(modelDirectory)
        sample.modelType = modelType
        assertTrue(sample.downloadModel())
        assertTrue(sample.initializeClassification())
        try {
            val result = sample.classify(grayRgba(224, 224), 224, 224)
            assertNotNull(result)
            assertEquals(CLASSIFICATION_TOP_COUNT, result!!.value.size)
        } finally {
            sample.releaseClassification()
        }
    }

    private fun grayRgba(width: Int, height: Int): ByteArray {
        val image = ByteArray(width * height * 4)
        for (offset in image.indices step 4) {
            image[offset] = 127
            image[offset + 1] = 127
            image[offset + 2] = 127
            image[offset + 3] = 0xFF.toByte()
        }
        return image
    }
}
