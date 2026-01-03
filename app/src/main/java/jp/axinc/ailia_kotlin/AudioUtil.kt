package jp.axinc.ailia_kotlin

import android.widget.*
import androidx.camera.core.*
import axip.ailia.*
import axip.ailia_tflite.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

public class AudioUtil {
    data class WavFileData(
            val sampleRate: Int,
            val channels: Int,
            val audioData: FloatArray
    )

    fun loadRawAudio(inputStream: InputStream): WavFileData {
        inputStream.use { stream ->
                val riffHeader = ByteArray(12)
            if (stream.read(riffHeader) != 12) throw IllegalArgumentException("Invalid WAV file (header too short)")

            val riff = String(riffHeader, 0, 4, Charsets.US_ASCII)
            val wave = String(riffHeader, 8, 4, Charsets.US_ASCII)
            if (riff != "RIFF" || wave != "WAVE") throw IllegalArgumentException("Invalid WAV RIFF/WAVE header")

            var sampleRate = 0
            var channels = 0
            var bitsPerSample = 0

            // チャンク探索ループ
            while (true) {
                val header = ByteArray(8)
                if (stream.read(header) != 8) throw IllegalArgumentException("Unexpected EOF while reading chunk header")

                val chunkId = String(header, 0, 4, Charsets.US_ASCII)
                val chunkSize = ByteBuffer.wrap(header, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int

                when (chunkId) {
                    "fmt " -> {
                        val fmtData = ByteArray(chunkSize)
                        if (stream.read(fmtData) != chunkSize) throw IllegalArgumentException("Invalid fmt chunk")

                        val fmtBuf = ByteBuffer.wrap(fmtData).order(ByteOrder.LITTLE_ENDIAN)
                        val audioFormat = fmtBuf.short.toInt()
                        channels = fmtBuf.short.toInt()
                        sampleRate = fmtBuf.int
                        fmtBuf.int  // byte rate
                        fmtBuf.short // block align
                                bitsPerSample = fmtBuf.short.toInt()

                        if (audioFormat != 1 || bitsPerSample != 16) {
                            throw IllegalArgumentException("Unsupported WAV format: Only PCM 16-bit supported.")
                        }
                    }

                    "data" -> {
                        // PCMデータチャンク
                        val pcmData = ByteArray(chunkSize)
                        if (stream.read(pcmData) != chunkSize) throw IllegalArgumentException("Invalid data chunk")

                        val numSamples = chunkSize / 2
                        val floatArray = FloatArray(numSamples)
                        val buf = ByteBuffer.wrap(pcmData).order(ByteOrder.LITTLE_ENDIAN)
                        for (i in 0 until numSamples) {
                            floatArray[i] = buf.short / 32768.0f
                        }
                        return WavFileData(
                                sampleRate = sampleRate,
                                channels = channels,
                                audioData = floatArray
                        )
                    }

                    else -> {
                        // 未知チャンク（JUNK含む） → スキップ
                        // chunkSizeは偶数境界。偶数でなければ1バイトパディングがある
                        val skip = if (chunkSize % 2 == 1) chunkSize + 1 else chunkSize
                        stream.skip(skip.toLong())
                    }
                }
            }
        }
    }
}
