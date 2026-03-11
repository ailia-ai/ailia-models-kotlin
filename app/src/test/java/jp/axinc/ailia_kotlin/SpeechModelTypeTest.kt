package jp.axinc.ailia_kotlin

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for SpeechModelType enum.
 *
 * Note: Since SpeechModelType references AiliaSpeech constants which require JNI,
 * these tests verify the enum structure and properties that can be tested without
 * native library loading. The modelTypeId values are tested in instrumented tests.
 */
class SpeechModelTypeTest {

    @Test
    fun enumValues_hasFourModels() {
        val values = SpeechModelType.values()
        assertEquals("Should have 4 speech model types", 4, values.size)
    }

    @Test
    fun enumValues_containsExpectedModels() {
        val names = SpeechModelType.values().map { it.name }
        assertTrue("Should contain WHISPER_TINY", names.contains("WHISPER_TINY"))
        assertTrue("Should contain WHISPER_BASE", names.contains("WHISPER_BASE"))
        assertTrue("Should contain WHISPER_SMALL", names.contains("WHISPER_SMALL"))
        assertTrue("Should contain SENSEVOICE_SMALL", names.contains("SENSEVOICE_SMALL"))
    }

    @Test
    fun displayNames_areUnique() {
        val displayNames = SpeechModelType.values().map { it.displayName }
        assertEquals("Display names should be unique", displayNames.size, displayNames.distinct().size)
    }

    @Test
    fun displayNames_areCorrect() {
        assertEquals("Whisper Tiny", SpeechModelType.WHISPER_TINY.displayName)
        assertEquals("Whisper Base", SpeechModelType.WHISPER_BASE.displayName)
        assertEquals("Whisper Small", SpeechModelType.WHISPER_SMALL.displayName)
        assertEquals("SenseVoice Small", SpeechModelType.SENSEVOICE_SMALL.displayName)
    }

    @Test
    fun encoderFileNames_areUnique() {
        val fileNames = SpeechModelType.values().map { it.encoderFileName }
        assertEquals("Encoder file names should be unique", fileNames.size, fileNames.distinct().size)
    }

    @Test
    fun sensevoice_doesNotNeedDecoder() {
        assertFalse("SenseVoice should not need decoder", SpeechModelType.SENSEVOICE_SMALL.needsDecoder)
        assertTrue("SenseVoice decoder URL should be empty", SpeechModelType.SENSEVOICE_SMALL.decoderUrl.isEmpty())
        assertTrue("SenseVoice decoder filename should be empty", SpeechModelType.SENSEVOICE_SMALL.decoderFileName.isEmpty())
    }

    @Test
    fun allWhisperModels_needDecoder() {
        assertTrue("Whisper Tiny should need decoder", SpeechModelType.WHISPER_TINY.needsDecoder)
        assertTrue("Whisper Base should need decoder", SpeechModelType.WHISPER_BASE.needsDecoder)
        assertTrue("Whisper Small should need decoder", SpeechModelType.WHISPER_SMALL.needsDecoder)
    }

    @Test
    fun whisperModels_haveNonEmptyDecoderUrls() {
        assertTrue("Whisper Tiny decoder URL should not be empty", SpeechModelType.WHISPER_TINY.decoderUrl.isNotEmpty())
        assertTrue("Whisper Base decoder URL should not be empty", SpeechModelType.WHISPER_BASE.decoderUrl.isNotEmpty())
        assertTrue("Whisper Small decoder URL should not be empty", SpeechModelType.WHISPER_SMALL.decoderUrl.isNotEmpty())
    }

    @Test
    fun whisperModels_haveNonEmptyDecoderFileNames() {
        assertTrue("Whisper Tiny decoder filename should not be empty", SpeechModelType.WHISPER_TINY.decoderFileName.isNotEmpty())
        assertTrue("Whisper Base decoder filename should not be empty", SpeechModelType.WHISPER_BASE.decoderFileName.isNotEmpty())
        assertTrue("Whisper Small decoder filename should not be empty", SpeechModelType.WHISPER_SMALL.decoderFileName.isNotEmpty())
    }

    @Test
    fun allModels_haveNonEmptyEncoderUrls() {
        for (model in SpeechModelType.values()) {
            assertTrue("${model.name} encoder URL should not be empty", model.encoderUrl.isNotEmpty())
        }
    }

    @Test
    fun allModels_haveNonEmptyEncoderFileNames() {
        for (model in SpeechModelType.values()) {
            assertTrue("${model.name} encoder filename should not be empty", model.encoderFileName.isNotEmpty())
        }
    }

    @Test
    fun encoderUrls_areValidHttpsUrls() {
        for (model in SpeechModelType.values()) {
            assertTrue(
                "${model.name} encoder URL should start with https://",
                model.encoderUrl.startsWith("https://")
            )
        }
    }

    @Test
    fun decoderUrls_areValidHttpsUrls_orEmpty() {
        for (model in SpeechModelType.values()) {
            if (model.needsDecoder) {
                assertTrue(
                    "${model.name} decoder URL should start with https://",
                    model.decoderUrl.startsWith("https://")
                )
            } else {
                assertTrue(
                    "${model.name} decoder URL should be empty",
                    model.decoderUrl.isEmpty()
                )
            }
        }
    }

    @Test
    fun encoderFileNames_endWithOnnx() {
        for (model in SpeechModelType.values()) {
            assertTrue(
                "${model.name} encoder filename should end with .onnx",
                model.encoderFileName.endsWith(".onnx")
            )
        }
    }

    @Test
    fun decoderFileNames_endWithOnnx_orEmpty() {
        for (model in SpeechModelType.values()) {
            if (model.needsDecoder) {
                assertTrue(
                    "${model.name} decoder filename should end with .onnx",
                    model.decoderFileName.endsWith(".onnx")
                )
            }
        }
    }
}
