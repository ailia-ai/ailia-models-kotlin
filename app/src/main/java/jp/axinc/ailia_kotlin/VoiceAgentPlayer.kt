package jp.axinc.ailia_kotlin

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import android.util.Log
import kotlin.math.roundToInt

/** PCM output and playback-position waveform sampling, independent of models and session state. */
internal class VoiceAgentPlayer {
    private val playbackLock = Any()
    private var playbackTrack: AudioTrack? = null
    @Volatile private var lastOutputEndMs = 0L

    /** The speaker/room can still be audible just after AudioTrack has drained or been flushed. */
    val remainingOutputSettleMs: Long get() = if (lastOutputEndMs == 0L) 0 else
        (OUTPUT_SETTLE_MS - (SystemClock.elapsedRealtime() - lastOutputEndMs)).coerceAtLeast(0)

    companion object {
        const val OUTPUT_SETTLE_MS = 400L
    }

    /** Call serially on a worker. Small buffers keep cancellation responsive during long answers. */
    fun play(clip: AudioUtil.WavFileData, checkActive: () -> Unit, onWaveform: (FloatArray) -> Unit) {
        checkActive()
        val audio = clip.audioData
        val channels = clip.channels
        val rate = clip.sampleRate
        require(audio.isNotEmpty() && channels in 1..2 && rate > 0) { "合成音声の形式が不正です" }
        val mask = if (channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO
        val minimum = AudioTrack.getMinBufferSize(rate, mask, AudioFormat.ENCODING_PCM_16BIT)
        check(minimum > 0) { "音声出力を初期化できません" }
        val track = AudioTrack(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build(),
            AudioFormat.Builder().setSampleRate(rate).setChannelMask(mask).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build(),
            maxOf(minimum, rate / 5 * channels * 2), AudioTrack.MODE_STREAM, android.media.AudioManager.AUDIO_SESSION_ID_GENERATE)
        try {
            check(track.state == AudioTrack.STATE_INITIALIZED) { "音声出力を初期化できません" }
            val pcm = ShortArray(audio.size) { (audio[it].coerceIn(-1f, 1f) * 32767).roundToInt().toShort() }
            synchronized(playbackLock) {
                checkActive()
                playbackTrack = track
                track.play()
            }
            var offset = 0
            while (offset < pcm.size) {
                checkActive()
                val count = minOf(rate / 20 * channels, pcm.size - offset)
                val written = track.write(pcm, offset, count)
                check(written > 0) { "音声出力に失敗しました ($written)" }
                offset += written
                waveform(clip, track.playbackHeadPosition * channels, onWaveform)
            }
            val deadline = SystemClock.elapsedRealtime() + 5000
            while (track.playbackHeadPosition < pcm.size / channels) {
                checkActive()
                check(SystemClock.elapsedRealtime() < deadline) { "音声出力が停止しました" }
                waveform(clip, track.playbackHeadPosition * channels, onWaveform)
                Thread.sleep(40)
            }
        } finally {
            synchronized(playbackLock) {
                if (playbackTrack === track) playbackTrack = null
                try { track.stop() } finally {
                    track.release()
                    lastOutputEndMs = SystemClock.elapsedRealtime()
                }
            }
        }
    }

    /** May be called from the UI thread; the playback worker remains responsible for release. */
    fun silence() = synchronized(playbackLock) {
        playbackTrack?.let { track ->
            lastOutputEndMs = SystemClock.elapsedRealtime()
            runCatching { track.pause(); track.flush() }
                .onFailure { Log.w("VoiceAgent", "Failed to pause playback", it) }
        }
    }

    private fun waveform(clip: AudioUtil.WavFileData, offset: Int, onWaveform: (FloatArray) -> Unit) {
        val start = offset.coerceIn(0, clip.audioData.size)
        onWaveform(clip.audioData.copyOfRange(start,
            minOf(start + clip.sampleRate / 20 * clip.channels, clip.audioData.size)))
    }
}
