package jp.axinc.ailia_kotlin

/** SDK protocol separated from JNI so finalize/drain/reset ordering can be verified. */
internal class VoiceAgentSpeechTurn(private val engine: Engine) {
    interface Engine {
        fun pushInputData(audio: FloatArray, channels: Int, sampleRate: Int)
        fun buffered(): Boolean
        fun finalizeInputData()
        fun transcribe(): List<String>
        fun complete(): Boolean
        fun resetTranscribeState()
    }

    private var finalized = false

    fun push(audio: FloatArray, channels: Int, sampleRate: Int): Boolean {
        check(!finalized) { "確定済みの音声入力に追加できません" }
        engine.pushInputData(audio, channels, sampleRate)
        return engine.buffered()
    }

    /** Called once at first Buffered, manual submit, or the recording length limit. */
    fun finish(checkActive: () -> Unit): List<String> {
        check(!finalized) { "音声入力はすでに確定しています" }
        checkActive()
        finalized = true
        engine.finalizeInputData()
        val lines = mutableListOf<String>()
        while (engine.buffered()) {
            checkActive()
            lines += engine.transcribe()
        }
        check(engine.complete()) { "音声認識が完了状態になりませんでした" }
        engine.resetTranscribeState()
        finalized = false
        return lines
    }
}
