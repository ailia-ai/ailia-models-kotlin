package jp.axinc.ailia_kotlin

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class QnnSupportTest {
    @Test
    fun `FP16 non capable SoC disables the ailia SDK QNN environment`() {
        assertFalse(QnnSupport.isFp16SupportedSoc("sm7635"))
    }

    @Test
    fun `FP16 capable SoC keeps the QNN environment selectable`() {
        assertTrue(QnnSupport.isFp16SupportedSoc("sm8475"))
        assertTrue(QnnSupport.isFp16SupportedSoc("sm8550"))
        assertTrue(QnnSupport.isFp16SupportedSoc("sm8650"))
        assertTrue(QnnSupport.isFp16SupportedSoc("sm8750"))
    }

    @Test
    fun `FP16 support does not follow the Hexagon version`() {
        // v73でもsm8550は対応、sm7635 / sm8635 は非対応。v69でもsm7475は非対応。
        assertTrue(QnnSupport.isFp16SupportedSoc("sm8550"))
        assertFalse(QnnSupport.isFp16SupportedSoc("sm8635"))
        assertFalse(QnnSupport.isFp16SupportedSoc("sm8735"))
        assertTrue(QnnSupport.isFp16SupportedSoc("sm8475"))
        assertFalse(QnnSupport.isFp16SupportedSoc("sm7475"))
    }

    @Test
    fun `unknown or missing SoC keeps the previous behaviour`() {
        assertTrue(QnnSupport.isFp16SupportedSoc(null))
        assertTrue(QnnSupport.isFp16SupportedSoc("sm9999"))
    }

    @Test
    fun `converted models are named after the SoC`() {
        assertEquals(
            "gemma4-e2b-sm8475-fp16-mtp.qnn",
            QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E2B),
        )
        assertEquals(
            "gemma4-e2b-sm7635-int16-mmproj.qnn",
            QnnSupport.llmQnnFileName("sm7635", LLMModelType.GEMMA_4_E2B, QnnSupport.QnnLlmFile.MMPROJ),
        )
        assertEquals(
            "gemma4-e2b-sm8475-fp16-mmproj.qnn",
            QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E2B, QnnSupport.QnnLlmFile.MMPROJ),
        )
    }

    @Test
    fun `MTP assistants match the SoC and precision of the target model`() {
        assertEquals(
            "gemma4-e2b-sm8475-fp16-mtp-assistant.qnn",
            QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E2B, QnnSupport.QnnLlmFile.MTP_ASSISTANT),
        )
        assertEquals(
            "gemma4-e4b-sm7635-int16-mtp-assistant.qnn",
            QnnSupport.llmQnnFileName("sm7635", LLMModelType.GEMMA_4_E4B, QnnSupport.QnnLlmFile.MTP_ASSISTANT),
        )
    }

    @Test
    fun `v1_5_0 format uses the previous target model without an MTP assistant`() {
        val format = QnnSupport.QnnModelFormat.V1_5_0
        assertEquals(
            "gemma4-e2b-sm8475.qnn",
            QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E2B, format = format),
        )
        assertEquals(
            "gemma4-e2b-sm8475-mmproj.qnn",
            QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E2B, QnnSupport.QnnLlmFile.MMPROJ, format),
        )
        assertNull(
            QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E2B, QnnSupport.QnnLlmFile.MTP_ASSISTANT, format),
        )
    }

    @Test
    fun `Gemma 4 E4B runs on QNN only on sm7635`() {
        assertEquals(
            "gemma4-e4b-sm7635-int16-mtp.qnn",
            QnnSupport.llmQnnFileName("sm7635", LLMModelType.GEMMA_4_E4B),
        )
        assertEquals(
            "gemma4-e4b-sm7635-int16-mmproj.qnn",
            QnnSupport.llmQnnFileName("sm7635", LLMModelType.GEMMA_4_E4B, QnnSupport.QnnLlmFile.MMPROJ),
        )
        // Hexagon v69はContext Binaryの2GB制約のためE4BをQNNで使わない
        assertNull(QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E4B))
    }

    @Test
    fun `models without a converted file are not offered on QNN`() {
        // Gemma 2は変換済みモデルがない
        assertNull(QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_2_2B))
        // モデルを公開していないSoC
        assertNull(QnnSupport.llmQnnFileName("sm8650", LLMModelType.GEMMA_4_E2B))
        assertNull(QnnSupport.llmQnnFileName(null, LLMModelType.GEMMA_4_E2B))
    }

    @Test
    fun `Whisper Small, Medium, Large V3 and Large V3 Turbo run on QNN`() {
        assertEquals(
            "whisper-small-sm7635-int16.qnn",
            QnnSupport.speechQnnFileName("sm7635", SpeechModelType.WHISPER_SMALL),
        )
        assertEquals(
            "whisper-large-v3-sc8380xp-int16.qnn",
            QnnSupport.speechQnnFileName("sc8380xp", SpeechModelType.WHISPER_LARGE_V3),
        )
        assertNull(QnnSupport.speechQnnFileName("sm8475", SpeechModelType.WHISPER_LARGE_V3))
        assertNull(QnnSupport.speechQnnFileName("qcs6490", SpeechModelType.WHISPER_LARGE_V3))
        assertNull(QnnSupport.speechQnnFileName("sm7635", SpeechModelType.WHISPER_TINY))
        assertEquals(
            "whisper-large-v3-turbo-sm7635-int16.qnn",
            QnnSupport.speechQnnFileName("sm7635", SpeechModelType.WHISPER_LARGE_V3_TURBO),
        )
        assertEquals(
            "whisper-medium-sm7635-int16.qnn",
            QnnSupport.speechQnnFileName("sm7635", SpeechModelType.WHISPER_MEDIUM),
        )
        assertEquals(
            "whisper-medium-sm8475-int16.qnn",
            QnnSupport.speechQnnFileName("sm8475", SpeechModelType.WHISPER_MEDIUM),
        )
        assertEquals(
            "whisper-large-v3-turbo-qcs6490-int16.qnn",
            QnnSupport.speechQnnFileName("qcs6490", SpeechModelType.WHISPER_LARGE_V3_TURBO),
        )
        assertNull(QnnSupport.speechQnnFileName("sm8650", SpeechModelType.WHISPER_LARGE_V3_TURBO))
        assertNull(QnnSupport.speechQnnFileName("sm7635", SpeechModelType.SENSEVOICE_SMALL))
        assertNull(QnnSupport.speechQnnFileName(null, SpeechModelType.WHISPER_LARGE_V3_TURBO))
    }
}
