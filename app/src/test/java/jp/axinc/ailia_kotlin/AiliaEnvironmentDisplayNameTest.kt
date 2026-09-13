package jp.axinc.ailia_kotlin

import axip.ailia.AiliaEnvironment
import org.junit.Assert.assertEquals
import org.junit.Test

class AiliaEnvironmentDisplayNameTest {
    @Test
    fun `formats Vulkan GPU properties`() {
        val environment = AiliaEnvironment(
            2,
            AiliaEnvironment.TYPE_GPU,
            "Adreno 740",
            AiliaEnvironment.BACKEND_VULKAN,
            AiliaEnvironment.PROPERTY_FP16 or AiliaEnvironment.PROPERTY_LOWPOWER,
        )

        assertEquals(
            "ailia SDK GPU (Vulkan, FP16, Low Power) - Adreno 740 (id:2)",
            environment.toBackendDisplayName(),
        )
    }

    @Test
    fun `formats CPU and BLAS environments`() {
        val cpu = AiliaEnvironment(
            0,
            AiliaEnvironment.TYPE_CPU,
            "CPU",
            AiliaEnvironment.BACKEND_NONE,
            AiliaEnvironment.PROPERTY_NORMAL,
        )
        val blas = AiliaEnvironment(
            1,
            AiliaEnvironment.TYPE_BLAS,
            "OpenBLAS",
            AiliaEnvironment.BACKEND_NONE,
            AiliaEnvironment.PROPERTY_NORMAL,
        )

        assertEquals("ailia SDK CPU - CPU (id:0)", cpu.toBackendDisplayName())
        assertEquals("ailia SDK CPU (BLAS) - OpenBLAS (id:1)", blas.toBackendDisplayName())
    }
}
