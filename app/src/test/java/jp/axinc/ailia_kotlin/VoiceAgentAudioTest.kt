package jp.axinc.ailia_kotlin

import org.junit.Assert.*
import org.junit.Test

class VoiceAgentAudioTest {
    @Test fun excludesPlaybackTailBeforeForwardingInput() {
        val gate = VoiceAgentInputGate(6400)
        repeat(4) {
            assertTrue(gate.accept(FloatArray(1600) { .3f }).isEmpty())
            assertFalse(gate.hasInput)
        }
        val speech = FloatArray(1600) { .001f }
        assertSame(speech, gate.accept(speech))
        assertTrue(gate.hasInput)
    }

    @Test fun preservesSpeechWhenGuardEndsInsideAChunk() {
        val gate = VoiceAgentInputGate(6500)
        repeat(4) { gate.accept(FloatArray(1600)) }
        val input = FloatArray(1600) { it.toFloat() }
        assertArrayEquals(input.copyOfRange(100, 1600), gate.accept(input), 0f)
        assertTrue(gate.hasInput)
    }

    @Test fun silenceAndQuietSpeechAreBothForwardedToSileroWithoutAnAmplitudeGate() {
        val gate = VoiceAgentInputGate()
        for (level in listOf(0f, .0001f, .1f)) {
            val input = FloatArray(1600) { level }
            assertSame(input, gate.accept(input))
        }
    }

    @Test fun audioLevelRemainsAvailableForAnimation() {
        assertEquals(0f, voiceAgentAudioRms(FloatArray(0)), 0f)
        assertEquals(.1f, voiceAgentAudioRms(floatArrayOf(.1f, -.1f)), .00001f)
    }

    @Test fun removesOnlySpeechTimestamps() {
        assertEquals("10時に会いましょう。 [1, 2]を足して。", voiceAgentTranscript(listOf(
            "[00:00 - 00:03] 10時に会いましょう。", "[00:03 - 00:08] [1, 2]を足して。")))
        assertEquals("", voiceAgentTranscript(emptyList()))
    }
}
