package jp.axinc.ailia_kotlin

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

class VoiceAgentSpeechQueueTest {
    @Test fun splitsPhrasesAndPreservesPunctuationQuotesAndUnterminatedTail() {
        assertEquals(listOf("こんにちは、", "私はアシスタントです。", "「元気ですか！？」", "はい！", "続きです"),
            voiceAgentSpeechSegments("こんにちは、私はアシスタントです。「元気ですか！？」 はい！\n続きです"))
        assertEquals(listOf("Hello,", "world!", "How are you?", "Fine."),
            voiceAgentSpeechSegments("Hello, world! How are you? Fine."))
        assertEquals(listOf("価格は1,234.56円、", "比率は０．５です。"),
            voiceAgentSpeechSegments("価格は1,234.56円、比率は０．５です。"))
        assertEquals(listOf("改行", "区切り"), voiceAgentSpeechSegments("改行\r\n区切り"))
        assertTrue(voiceAgentSpeechSegments("  \n。！？ ").isEmpty())
    }

    @Test(timeout = 5000) fun startsPlaybackBeforeAllSynthesisAndPreservesOrderWithBackpressure() {
        val playbackStarted = CountDownLatch(1)
        val fourSynthesized = CountDownLatch(1)
        val releasePlayback = CountDownLatch(1)
        val count = AtomicInteger()
        val played = mutableListOf<String>()
        val error = AtomicReference<Throwable?>()
        val producerThread = AtomicReference<Thread>()
        val caller = Thread {
            try {
                producerThread.set(Thread.currentThread())
                voiceAgentQueueSpeech((1..8).map { "$it。" }, {}, { text ->
                    assertSame(producerThread.get(), Thread.currentThread())
                    if (count.incrementAndGet() == 2) assertTrue(playbackStarted.await(2, TimeUnit.SECONDS))
                    if (count.get() == 4) fourSynthesized.countDown()
                    text
                }, { text, checkActive ->
                    assertNotSame(producerThread.get(), Thread.currentThread())
                    if (text == "1。") {
                        playbackStarted.countDown()
                        assertTrue(releasePlayback.await(2, TimeUnit.SECONDS))
                    }
                    checkActive()
                    played += text
                })
            } catch (failure: Throwable) { error.set(failure) }
        }
        caller.start()
        try {
            assertTrue(fourSynthesized.await(2, TimeUnit.SECONDS))
            // One playing, two queued, and the fourth clip blocked while enqueueing.
            Thread.sleep(100)
            assertEquals(4, count.get())
        } finally {
            releasePlayback.countDown()
            caller.join(2000)
        }
        assertFalse(caller.isAlive)
        error.get()?.let { throw it }
        assertEquals((1..8).map { "$it。" }, played)
    }

    @Test(timeout = 5000) fun cancellationDropsQueuedClipsAndJoinsPlayback() {
        val cancelled = AtomicBoolean(false)
        val playbackStarted = CountDownLatch(1)
        val fourSynthesized = CountDownLatch(1)
        val playbackExited = AtomicBoolean(false)
        val error = AtomicReference<Throwable?>()
        val played = mutableListOf<String>()
        val count = AtomicInteger()
        val caller = Thread {
            try {
                voiceAgentQueueSpeech((1..8).map { "$it。" }, {
                    if (cancelled.get()) throw CancellationException()
                }, { text ->
                    if (count.incrementAndGet() == 2) assertTrue(playbackStarted.await(2, TimeUnit.SECONDS))
                    if (count.get() == 4) fourSynthesized.countDown()
                    text
                }, { text, checkActive ->
                    played += text
                    playbackStarted.countDown()
                    try {
                        while (true) { checkActive(); Thread.sleep(5) }
                    } finally { playbackExited.set(true) }
                })
            } catch (failure: Throwable) { error.set(failure) }
        }
        caller.start()
        try {
            assertTrue(fourSynthesized.await(2, TimeUnit.SECONDS))
        } finally {
            cancelled.set(true)
            caller.join(2000)
        }
        assertFalse(caller.isAlive)
        assertTrue(error.get() is CancellationException)
        assertTrue(playbackExited.get())
        assertEquals(listOf("1。"), played)
    }

    @Test(timeout = 5000) fun playbackFailureReachesProducerInsteadOfLeavingItBlocked() {
        val failure = IllegalStateException("playback failed")
        try {
            voiceAgentQueueSpeech((1..10).map { "$it。" }, {}, { it }, { _, _ -> throw failure })
            fail("Playback failure should propagate")
        } catch (actual: IllegalStateException) { assertSame(failure, actual) }
    }

    @Test(timeout = 5000) fun synthesisFailureStopsAndJoinsPlayback() {
        val playbackStarted = CountDownLatch(1)
        val playbackExited = AtomicBoolean(false)
        val failure = IllegalStateException("synthesis failed")
        try {
            voiceAgentQueueSpeech(listOf("最初。", "次。"), {}, { text ->
                if (text == "次。") {
                    assertTrue(playbackStarted.await(2, TimeUnit.SECONDS))
                    throw failure
                }
                text
            }, { _, checkActive ->
                playbackStarted.countDown()
                try { while (true) { checkActive(); Thread.sleep(5) } }
                finally { playbackExited.set(true) }
            })
            fail("Synthesis failure should propagate")
        } catch (actual: IllegalStateException) { assertSame(failure, actual) }
        assertTrue(playbackExited.get())
    }
}
