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
    fun enumValues_hasSixModels() {
        val values = SpeechModelType.values()
        assertEquals("Should have 7 speech model types", 7, values.size)
    }

    @Test
    fun enumValues_containsExpectedModels() {
        val names = SpeechModelType.values().map { it.name }
        assertTrue("Should contain WHISPER_TINY", names.contains("WHISPER_TINY"))
        assertTrue("Should contain WHISPER_BASE", names.contains("WHISPER_BASE"))
        assertTrue("Should contain WHISPER_SMALL", names.contains("WHISPER_SMALL"))
        assertTrue("Should contain WHISPER_MEDIUM", names.contains("WHISPER_MEDIUM"))
        assertTrue("Should contain WHISPER_LARGE_V3", names.contains("WHISPER_LARGE_V3"))
        assertTrue("Should contain WHISPER_LARGE_V3_TURBO", names.contains("WHISPER_LARGE_V3_TURBO"))
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
        assertEquals("Whisper Medium", SpeechModelType.WHISPER_MEDIUM.displayName)
        assertEquals("Whisper Large V3", SpeechModelType.WHISPER_LARGE_V3.displayName)
        assertEquals("Whisper Large V3 Turbo", SpeechModelType.WHISPER_LARGE_V3_TURBO.displayName)
        assertEquals("SenseVoice Small", SpeechModelType.SENSEVOICE_SMALL.displayName)
    }

    @Test
    fun speechSample_defaultsToSenseVoice() {
        assertEquals(SpeechModelType.SENSEVOICE_SMALL, AiliaSpeechSample.DEFAULT_MODEL_TYPE)
    }

    @Test
    fun encoderFileNames_areUnique() {
        val fileNames = SpeechModelType.values().map { it.encoderFileName }
        assertEquals("Encoder file names should be unique", fileNames.size, fileNames.distinct().size)
    }

    @Test
    fun sensevoice_usesAuxiliaryModelFile() {
        val senseVoice = SpeechModelType.SENSEVOICE_SMALL
        assertTrue("SenseVoice should use its auxiliary model", senseVoice.needsDecoder)
        assertTrue(
            "SenseVoice auxiliary URL should end with .model",
            senseVoice.decoderUrl.endsWith(".model")
        )
        assertTrue(
            "SenseVoice auxiliary filename should end with .model",
            senseVoice.decoderFileName.endsWith(".model")
        )
    }

    @Test
    fun allWhisperModels_needDecoder() {
        assertTrue("Whisper Tiny should need decoder", SpeechModelType.WHISPER_TINY.needsDecoder)
        assertTrue("Whisper Base should need decoder", SpeechModelType.WHISPER_BASE.needsDecoder)
        assertTrue("Whisper Small should need decoder", SpeechModelType.WHISPER_SMALL.needsDecoder)
        assertTrue("Whisper Medium should need decoder", SpeechModelType.WHISPER_MEDIUM.needsDecoder)
        assertTrue("Whisper Large V3 Turbo should need decoder", SpeechModelType.WHISPER_LARGE_V3_TURBO.needsDecoder)
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
    fun whisperLargeV3Turbo_requiresExternalWeights() {
        val extras = SpeechModelType.WHISPER_LARGE_V3_TURBO.extraFiles
        assertEquals("Turbo should have one extra file", 1, extras.size)
        // onnx内の外部データ参照名と一致している必要がある
        assertEquals("encoder_turbo_weights.opt.pb", extras[0].fileName)
        assertTrue(
            "Turbo weights URL should start with https://",
            extras[0].url.startsWith("https://")
        )
    }

    @Test
    fun whisperLargeV3_requiresExternalWeights() {
        // onnx内の外部データ参照名と一致している必要がある
        assertEquals(
            listOf("encoder_large_v3_weights.pb", "decoder_large_v3_fix_kv_cache_weights.pb"),
            SpeechModelType.WHISPER_LARGE_V3.extraFiles.map { it.fileName }
        )
    }

    @Test
    fun otherModels_haveNoExtraFiles() {
        for (model in SpeechModelType.values()) {
            if (model != SpeechModelType.WHISPER_LARGE_V3_TURBO && model != SpeechModelType.WHISPER_LARGE_V3) {
                assertTrue("${model.name} should have no extra files", model.extraFiles.isEmpty())
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
    fun secondaryModelFileNames_haveSupportedExtension() {
        for (model in SpeechModelType.values()) {
            if (model.needsDecoder) {
                assertTrue(
                    "${model.name} secondary model filename should end with .onnx or .model",
                    model.decoderFileName.endsWith(".onnx") ||
                        model.decoderFileName.endsWith(".model")
                )
            }
        }
    }
}
