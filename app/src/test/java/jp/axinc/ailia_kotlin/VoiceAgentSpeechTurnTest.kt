package jp.axinc.ailia_kotlin

import org.junit.Assert.*
import org.junit.Test
import java.util.ArrayDeque
import java.util.concurrent.CancellationException

class VoiceAgentSpeechTurnTest {
    private class Engine : VoiceAgentSpeechTurn.Engine {
        val events = mutableListOf<String>()
        val segments = ArrayDeque<List<String>>()
        var finalized = false
        var reportComplete = true
        override fun pushInputData(audio: FloatArray, channels: Int, sampleRate: Int) { events += "push" }
        override fun buffered() = segments.isNotEmpty()
        override fun finalizeInputData() { events += "finalize"; finalized = true }
        override fun transcribe(): List<String> { events += "transcribe"; return segments.removeFirst() }
        override fun complete() = finalized && segments.isEmpty() && reportComplete
        override fun resetTranscribeState() { events += "reset"; finalized = false }
    }

    @Test fun waitsForBufferedThenFinalizesDrainsAndResetsBeforeTheSecondQuestion() {
        val engine = Engine()
        val turn = VoiceAgentSpeechTurn(engine)
        repeat(2) { index ->
            assertFalse(turn.push(FloatArray(1600), 1, 16000))
            engine.segments += listOf("question $index")
            engine.segments += listOf("tail $index")
            assertTrue(turn.push(FloatArray(1600), 1, 16000))
            assertEquals(listOf("question $index", "tail $index"), turn.finish {})
        }
        assertEquals(List(2) { listOf("push", "push", "finalize", "transcribe", "transcribe", "reset") }.flatten(), engine.events)
    }

    @Test fun manualSubmissionFinalizesEvenBeforeBufferedAndHandlesNoRecognizedSpeech() {
        val engine = Engine()
        val turn = VoiceAgentSpeechTurn(engine)
        assertFalse(turn.push(FloatArray(1600), 1, 16000))
        assertTrue(turn.finish {}.isEmpty())
        assertEquals(listOf("push", "finalize", "reset"), engine.events)
    }

    @Test fun cancellationStopsFurtherTranscriptionAndDoesNotResetAnUnfinishedEngine() {
        val engine = Engine()
        val turn = VoiceAgentSpeechTurn(engine)
        engine.segments += listOf("first")
        engine.segments += listOf("second")
        var checks = 0
        try {
            turn.finish { if (++checks == 3) throw CancellationException() }
            fail("Expected cancellation")
        } catch (_: CancellationException) { }
        assertEquals(listOf("finalize", "transcribe"), engine.events)
        try {
            turn.push(FloatArray(1600), 1, 16000)
            fail("Cancelled engine must be released, not reused without reset")
        } catch (_: IllegalStateException) { }
    }

    @Test fun incompleteEngineWithoutBufferedDataFailsInsteadOfBusyWaiting() {
        val engine = Engine().apply { reportComplete = false }
        try {
            VoiceAgentSpeechTurn(engine).finish {}
            fail("Unexpected SDK state should fail")
        } catch (_: IllegalStateException) { }
        assertEquals(listOf("finalize"), engine.events)
    }
}
