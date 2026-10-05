package jp.axinc.ailia_kotlin

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal fun createLongTensor(
    environment: OrtEnvironment,
    values: LongArray,
    shape: LongArray,
): OnnxTensor {
    val byteBuffer = ByteBuffer.allocateDirect(values.size * Long.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    val longBuffer = byteBuffer.asLongBuffer()
    longBuffer.put(values)
    longBuffer.rewind()
    return OnnxTensor.createTensor(environment, longBuffer, shape)
}

internal fun createBooleanTensor(
    environment: OrtEnvironment,
    values: ByteArray,
    shape: LongArray,
): OnnxTensor {
    val byteBuffer = ByteBuffer.allocateDirect(values.size)
    byteBuffer.put(values)
    byteBuffer.rewind()
    return OnnxTensor.createTensor(environment, byteBuffer, shape, OnnxJavaType.BOOL)
}

internal fun runFloatTensor(
    session: OrtSession,
    inputs: Map<String, OnnxTensor>,
    outputName: String,
): FloatArray {
    val result = session.run(inputs, setOf(outputName))
    return try {
        readFloatTensor(result[0] as OnnxTensor)
    } finally {
        result.close()
    }
}

internal fun closeTensors(vararg tensors: OnnxTensor) {
    tensors.forEach { it.close() }
}

internal fun runOrtSileroVad(
    environment: OrtEnvironment,
    session: OrtSession,
    audio: FloatArray,
    windowSize: Int,
    contextSize: Int,
    stateSize: Int,
): FloatArray {
    var state = FloatArray(stateSize)
    var context = FloatArray(contextSize)
    val probabilities = FloatArray((audio.size + windowSize - 1) / windowSize)
    val sampleRateTensor = createLongTensor(environment, longArrayOf(16_000L), longArrayOf())
    try {
        probabilities.indices.forEach { windowIndex ->
            val chunk = FloatArray(contextSize + windowSize)
            System.arraycopy(context, 0, chunk, 0, contextSize)
            val audioOffset = windowIndex * windowSize
            val available = minOf(windowSize, audio.size - audioOffset)
            System.arraycopy(audio, audioOffset, chunk, contextSize, available)
            val inputTensor = createFloatTensor(
                environment,
                chunk,
                longArrayOf(1, chunk.size.toLong()),
            )
            val stateTensor = createFloatTensor(environment, state, longArrayOf(2, 1, 128))
            try {
                val result = session.run(
                    mapOf(
                        "input" to inputTensor,
                        "state" to stateTensor,
                        "sr" to sampleRateTensor,
                    ),
                    linkedSetOf("output", "stateN"),
                )
                try {
                    probabilities[windowIndex] = readFloatTensor(result[0] as OnnxTensor)[0]
                    state = readFloatTensor(result[1] as OnnxTensor)
                } finally {
                    result.close()
                }
            } finally {
                closeTensors(inputTensor, stateTensor)
            }
            context = chunk.copyOfRange(chunk.size - contextSize, chunk.size)
        }
    } finally {
        sampleRateTensor.close()
    }
    return probabilities
}
