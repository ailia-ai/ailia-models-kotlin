package jp.axinc.ailia_kotlin

import axip.ailia.AiliaEnvironment

internal fun AiliaEnvironment.toBackendDisplayName(): String {
    val executionTarget = when (type) {
        AiliaEnvironment.TYPE_CPU -> "CPU"
        AiliaEnvironment.TYPE_BLAS -> "CPU (BLAS)"
        AiliaEnvironment.TYPE_GPU -> "GPU"
        AiliaEnvironment.TYPE_REMOTE -> "Remote"
        else -> "Environment"
    }
    val attributes = buildList {
        if (backend == AiliaEnvironment.BACKEND_VULKAN) {
            add("Vulkan")
        }
        if (props and AiliaEnvironment.PROPERTY_FP16 != 0) {
            add("FP16")
        }
        if (props and AiliaEnvironment.PROPERTY_LOWPOWER != 0) {
            add("Low Power")
        }
    }
    val attributeLabel = attributes.takeIf { it.isNotEmpty() }?.joinToString(", ", " (", ")").orEmpty()
    return "ailia SDK $executionTarget$attributeLabel - $name (id:$id)"
}
