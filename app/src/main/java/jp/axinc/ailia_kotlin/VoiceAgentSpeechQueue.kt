package jp.axinc.ailia_kotlin

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

private const val SPEECH_PUNCTUATION = "。、！？!?.,，．"
private const val SPEECH_CLOSINGS = "」』）)]】〉》”’\"'"

/** Keep punctuation and closing quotes with the preceding phrase; do not split decimal numbers. */
internal fun voiceAgentSpeechSegments(text: String): List<String> {
    val segments = mutableListOf<String>()
    val pending = StringBuilder()
    var boundary = false
    fun flush() {
        val phrase = pending.toString().trim()
        if (phrase.any { it.isLetterOrDigit() }) segments += phrase
        else if (phrase.isNotEmpty() && segments.isNotEmpty()) segments[segments.lastIndex] += phrase
        pending.setLength(0)
        boundary = false
    }
    for ((index, char) in text.withIndex()) {
        val numericSeparator = char in ".,，．" && index > 0 && index + 1 < text.length &&
            text[index - 1].isDigit() && text[index + 1].isDigit()
        val separator = char in SPEECH_PUNCTUATION && !numericSeparator
        if (char == '\n' || char == '\r') {
            flush()
            continue
        }
        if (boundary && !char.isWhitespace() && !separator && char !in SPEECH_CLOSINGS) flush()
        pending.append(char)
        if (separator) boundary = true
    }
    flush()
    return segments
}

/** Accept cumulative, filtered generation previews; flush the unterminated tail once per round. */
internal class VoiceAgentSpeechText {
    private var consumed = 0

    fun update(text: String, complete: Boolean): List<String> {
        val pending = text.substring(consumed.coerceAtMost(text.length))
        var readyEnd = 0
        var boundary = false
        for ((index, char) in pending.withIndex()) {
            if (char == '\n' || char == '\r') {
                readyEnd = index + 1
                boundary = false
                continue
            }
            // Wait one character for decimals and trailing quotes, including across MTP batches.
            if (char in ".,，．" && index > 0 && pending[index - 1].isDigit() &&
                index == pending.lastIndex && !complete) break
            val numericSeparator = char in ".,，．" && index > 0 && index < pending.lastIndex &&
                pending[index - 1].isDigit() && pending[index + 1].isDigit()
            val separator = char in SPEECH_PUNCTUATION && !numericSeparator
            if (boundary && !char.isWhitespace() && !separator && char !in SPEECH_CLOSINGS) readyEnd = index
            if (!char.isWhitespace() && char !in SPEECH_CLOSINGS) boundary = separator
        }
        if (complete) readyEnd = pending.length
        val result = voiceAgentSpeechSegments(pending.substring(0, readyEnd))
        consumed = if (complete) 0 else consumed + readyEnd
        return result
    }
}

/**
 * The LLM only queues text. A dedicated worker owns synthesis and a second worker owns playback.
 * Only text queues while generation/continuation runs; only two completed PCM clips can wait.
 * Playback is held until finish(), so generation keeps its UI state and failures speak nothing.
 */
internal class VoiceAgentStreamingSpeech<T : Any>(
    private val checkActive: () -> Unit,
    private val withSynthesis: (action: (synthesize: (String) -> T) -> Unit) -> Unit,
    private val play: (T, checkActive: () -> Unit) -> Unit,
) : AutoCloseable {
    private val text = VoiceAgentSpeechText()
    private val phrases = LinkedBlockingQueue<String>()
    private val playbackReady = CountDownLatch(1)
    private val stopped = AtomicBoolean(false)
    private val failure = AtomicReference<Throwable?>(null)
    // Load voice/reference while the LLM opens/prefills, before the first phrase arrives.
    private val worker = Thread({
        try {
            withSynthesis { synthesize ->
                voiceAgentQueueSpeech({ check -> nextPhrase(check) }, ::checkRunning, synthesize) { audio, check ->
                    while (!playbackReady.await(50, TimeUnit.MILLISECONDS)) check()
                    check()
                    play(audio, check)
                }
            }
        } catch (error: Throwable) {
            if (!stopped.get()) failure.compareAndSet(null, error)
        }
    }, "VoiceAgentSynthesis").also { it.start() }

    fun checkRunning() {
        checkActive()
        failure.get()?.let { throw it }
        if (stopped.get()) throw CancellationException()
    }

    fun update(text: String, complete: Boolean) {
        checkRunning()
        for (phrase in this.text.update(text, complete)) {
            phrases.add(phrase)
        }
    }

    private fun nextPhrase(check: () -> Unit): String? {
        while (true) {
            check()
            val phrase = phrases.poll(50, TimeUnit.MILLISECONDS) ?: continue
            return phrase.takeIf { it.isNotEmpty() } // Empty string is the end-of-generation marker.
        }
    }

    /** All rounds, including their final tails, must have been submitted before finishing. */
    fun finish() {
        checkRunning()
        phrases.add("")
        playbackReady.countDown()
        while (worker.isAlive) {
            checkRunning()
            worker.join(50)
        }
        checkRunning()
    }

    override fun close() {
        stopped.set(true)
        playbackReady.countDown()
        worker.interrupt()
        // Native synthesis cannot be aborted. Join its owner before recognition/the next turn.
        var interrupted = false
        while (worker.isAlive) {
            try { worker.join() } catch (_: InterruptedException) { interrupted = true }
        }
        phrases.clear()
        if (interrupted) Thread.currentThread().interrupt()
    }
}

/**
 * Synthesis stays on the calling worker; only PCM playback runs on a second thread.
 * At most two completed clips wait ahead of playback. Both sides observe failures/cancellation,
 * and this call always joins playback before another turn can start.
 */
internal fun <T : Any> voiceAgentQueueSpeech(
    segments: List<String>,
    checkActive: () -> Unit,
    synthesize: (String) -> T,
    play: (T, checkActive: () -> Unit) -> Unit,
) {
    val iterator = segments.iterator()
    voiceAgentQueueSpeech({ _ -> if (iterator.hasNext()) iterator.next() else null }, checkActive, synthesize, play)
}

private fun <T : Any> voiceAgentQueueSpeech(
    nextSegment: (checkRunning: () -> Unit) -> String?,
    checkActive: () -> Unit,
    synthesize: (String) -> T,
    play: (T, checkActive: () -> Unit) -> Unit,
) {
    val queue = ArrayBlockingQueue<T>(2)
    val producerDone = AtomicBoolean(false)
    val stopped = AtomicBoolean(false)
    val failure = AtomicReference<Throwable?>(null)
    fun checkRunning() {
        checkActive()
        failure.get()?.let { throw it }
        if (stopped.get()) throw CancellationException()
    }
    val playback = Thread({
        try {
            while (true) {
                checkRunning()
                val audio = queue.poll(50, TimeUnit.MILLISECONDS)
                if (audio != null) {
                    checkRunning()
                    play(audio, ::checkRunning)
                } else if (producerDone.get() && queue.isEmpty()) break
            }
        } catch (error: Throwable) {
            if (!stopped.get()) failure.compareAndSet(null, error)
        }
    }, "VoiceAgentPlayback")
    playback.start()
    try {
        while (true) {
            checkRunning()
            val segment = nextSegment(::checkRunning) ?: break
            val audio = synthesize(segment)
            checkRunning()
            while (!queue.offer(audio, 50, TimeUnit.MILLISECONDS)) checkRunning()
        }
        producerDone.set(true)
        while (playback.isAlive) {
            checkRunning()
            playback.join(50)
        }
        checkRunning()
    } finally {
        stopped.set(true)
        playback.interrupt()
        // Join even if the caller was interrupted: never let stale PCM escape into the next turn.
        var interrupted = false
        while (playback.isAlive) {
            try { playback.join() } catch (_: InterruptedException) { interrupted = true }
        }
        queue.clear()
        if (interrupted) Thread.currentThread().interrupt()
    }
}
