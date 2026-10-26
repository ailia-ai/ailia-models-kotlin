package jp.axinc.ailia_kotlin

import kotlin.math.sqrt

/** Excludes only the playback settling window; speech decisions belong to ailia Speech/Silero. */
internal class VoiceAgentInputGate(private val ignoreLeadingSamples: Int = 0) {
    private var samples = 0
    val hasInput: Boolean get() = samples > ignoreLeadingSamples

    fun accept(chunk: FloatArray): FloatArray {
        val skip = (ignoreLeadingSamples - samples).coerceIn(0, chunk.size)
        samples += chunk.size
        return if (skip == 0) chunk else chunk.copyOfRange(skip, chunk.size)
    }
}

/** Audio level is for animation only, never for deciding when to recognize a turn. */
internal fun voiceAgentAudioRms(samples: FloatArray): Float =
    if (samples.isEmpty()) 0f else sqrt(samples.sumOf { (it * it).toDouble() } / samples.size).toFloat()

/** Only remove the timestamp supplied by AiliaSpeechSample; preserve numbers in the utterance. */
internal fun voiceAgentTranscript(lines: List<String>): String = lines.joinToString(" ") {
    it.replaceFirst(Regex("^\\[\\d+:\\d+ - \\d+:\\d+\\]\\s*"), "").trim()
}.trim()
