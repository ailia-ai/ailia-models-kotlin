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
            "gemma4-e2b-sm8475.qnn",
            QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E2B),
        )
        assertEquals(
            "gemma4-e2b-sm7635-mmproj.qnn",
            QnnSupport.llmQnnFileName("sm7635", LLMModelType.GEMMA_4_E2B, mmproj = true),
        )
    }

    @Test
    fun `models without a converted file are not offered on QNN`() {
        // E4BとGemma 2は変換済みモデルが未公開
        assertNull(QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_4_E4B))
        assertNull(QnnSupport.llmQnnFileName("sm8475", LLMModelType.GEMMA_2_2B))
        // モデルを公開していないSoC
        assertNull(QnnSupport.llmQnnFileName("sm8650", LLMModelType.GEMMA_4_E2B))
        assertNull(QnnSupport.llmQnnFileName(null, LLMModelType.GEMMA_4_E2B))
    }
}
