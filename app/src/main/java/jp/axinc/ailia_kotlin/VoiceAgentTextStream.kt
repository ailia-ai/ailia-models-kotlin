package jp.axinc.ailia_kotlin

/** Preview only. Structured SDK responses remain authoritative for history and tool execution. */
internal class VoiceAgentTextStream(private val trimLeadingWhitespace: Boolean = true) {
    private val raw = StringBuilder()

    fun append(delta: String): String {
        raw.append(delta)
        val text = raw.toString()
        // Once a control section starts, defer the rest to the SDK's final parsed content.
        val end = CONTROL_MARKERS.map { text.indexOf(it) }.filter { it >= 0 }.minOrNull() ?: text.length
        val visible = text.substring(0, end)
        val opening = visible.lastIndexOf('<')
        // Markers may span multiple deltas. Hold their prefix so no internal syntax flashes onscreen.
        val safe = if (opening >= 0 && CONTROL_MARKERS.any { it.startsWith(visible.substring(opening)) }) {
            visible.substring(0, opening)
        } else visible
        return if (trimLeadingWhitespace) safe.trimStart() else safe
    }

    companion object {
        private val CONTROL_MARKERS = listOf("<|tool_call>", "<tool_call|>", "<|tool_response>",
            "<tool_response|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<eos>", "<think>")
    }
}
