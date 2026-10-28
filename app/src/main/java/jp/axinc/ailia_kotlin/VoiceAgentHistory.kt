package jp.axinc.ailia_kotlin

import axip.ailia_llm.AiliaLLM
import org.json.JSONObject

/** Retain a recent suffix, treating each parsed tool-call/result exchange as one indivisible unit. */
internal fun voiceAgentCompactHistory(
    history: MutableList<MutableList<JSONObject>>,
    currentTurn: MutableList<JSONObject>,
    budget: Int,
    maximumBudget: Int = budget,
    request: JSONObject? = null,
    tokens: (JSONObject) -> Int,
) {
    val allMessages = history.flatten()
    val requestIndex = allMessages.indexOfFirst { it === request }
    // An active request belongs before its replies, never before answers to older requests.
    val messages = if (request == null) allMessages else if (requestIndex >= 0) {
        allMessages.drop(requestIndex + 1)
    } else {
        // An oversized request may already have been replaced with a bounded text copy.
        allMessages.dropWhile { it.optString("role") == "user" }
    }
    val anchor = request?.let { message ->
        val limit = if (messages.isEmpty()) budget else budget / 3
        val fitted = fitVoiceAgentHistoryText(message, message.optString("content"), limit, keepTail = false, tokens)
        // Keep this node's identity so subsequent compactions do not duplicate the request.
        if (fitted == null) null else message.put("content", fitted.getString("content"))
    }
    val anchorTokens = anchor?.let(tokens) ?: 0
    val groups = mutableListOf<List<JSONObject>>()
    var index = 0
    while (index < messages.size) {
        val message = messages[index++]
        if (message.optString("role") == "assistant" && (message.optJSONArray("tool_calls")?.length() ?: 0) > 0) {
            val group = mutableListOf(message)
            while (index < messages.size && messages[index].optString("role") == "tool") group += messages[index++]
            groups += group
        } else if (message.optString("role") != "tool") {
            groups += listOf(message)
        }
    }
    val kept = mutableListOf<List<JSONObject>>()
    var remaining = budget - anchorTokens
    for (group in groups.asReversed()) {
        val cost = group.sumOf(tokens)
        if (cost <= remaining) {
            kept += group
            remaining -= cost
            continue
        }
        // Round up for one indivisible newest tool exchange/image if it still leaves generation room.
        if (kept.isEmpty() && cost <= maximumBudget - anchorTokens &&
            (group.size > 1 || group.singleOrNull()?.optJSONArray("content") != null)) {
            kept += group
            break
        }
        // A long latest answer/request can itself exceed the budget. Keep its most recent text.
        val message = group.singleOrNull()?.takeIf { (it.optJSONArray("tool_calls")?.length() ?: 0) == 0 }
        if (message != null) {
            val content = message.opt("content")
            val text = if (content is String) content else message.optJSONArray("content")?.let { parts ->
                (0 until parts.length()).mapNotNull { part ->
                    parts.getJSONObject(part).takeIf { it.optString("type") == "text" }?.optString("text")
                }.joinToString("\n")
            }
            if (!text.isNullOrEmpty()) {
                fitVoiceAgentHistoryText(message, text, remaining, keepTail = true, tokens)?.let { kept += listOf(it) }
            }
        }
        break
    }
    // Preserve the current turn's identity for cancellation bookkeeping and later tool results.
    currentTurn.clear()
    anchor?.let { currentTurn += it }
    currentTurn.addAll(kept.asReversed().flatten())
    history.clear()
    history += currentTurn
}

/** Keep a bounded prefix of the active request or suffix of a recent answer without splitting UTF-16 pairs. */
private fun fitVoiceAgentHistoryText(message: JSONObject, text: String, budget: Int,
                                    keepTail: Boolean, tokens: (JSONObject) -> Int): JSONObject? {
    fun fitted(length: Int): JSONObject {
        var start = if (keepTail) text.length - length else 0
        var end = if (keepTail) text.length else length
        if (start < text.length && text[start].isLowSurrogate()) start++
        if (end > start && text[end - 1].isHighSurrogate()) end--
        return JSONObject().put("role", message.getString("role")).put("content", text.substring(start, end))
    }
    var low = 0
    var high = text.length
    while (low < high) {
        val middle = (low + high + 1) / 2
        if (tokens(fitted(middle)) <= budget) low = middle else high = middle - 1
    }
    return if (low > 0) fitted(low).takeIf { it.getString("content").isNotEmpty() } else null
}

/** The JNI wrapper reports status codes in RuntimeException messages, without a typed exception. */
internal fun voiceAgentIsContextFull(error: RuntimeException): Boolean {
    val status = AiliaLLM.AILIA_LLM_STATUS_CONTEXT_FULL
    return generateSequence<Throwable>(error) { it.cause }.any {
        Regex("(?i)(?:status:|failed:)\\s*$status\\s*$").containsMatchIn(it.message.orEmpty())
    }
}
