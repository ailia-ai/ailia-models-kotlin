package jp.axinc.ailia_kotlin

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
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

        /** これより短い音声は聞き取れないため再生しない(記号だけの文で数サンプルの音声が返ることがある)。 */
        private const val MIN_PLAYABLE_MS = 20
    }

    /** Call serially on a worker. Small buffers keep cancellation responsive during long answers. */
    fun play(clip: AudioUtil.WavFileData, checkActive: () -> Unit, onWaveform: (FloatArray) -> Unit) {
        checkActive()
        val audio = clip.audioData
        val channels = clip.channels
        val rate = clip.sampleRate
        require(audio.isNotEmpty() && channels in 1..2 && rate > 0) { "合成音声の形式が不正です" }
        val frames = audio.size / channels
        if (frames < rate * MIN_PLAYABLE_MS / 1000) {
            Log.d("VoiceAgent", "Skipping a $frames-frame clip that is too short to hear")
            return
        }
        val mask = if (channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO
        val minimum = AudioTrack.getMinBufferSize(rate, mask, AudioFormat.ENCODING_PCM_16BIT)
        check(minimum > 0) { "音声出力を初期化できません" }
        val bufferBytes = maxOf(minimum, rate / 5 * channels * 2)
        val track = AudioTrack(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build(),
            AudioFormat.Builder().setSampleRate(rate).setChannelMask(mask).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build(),
            bufferBytes, AudioTrack.MODE_STREAM, android.media.AudioManager.AUDIO_SESSION_ID_GENERATE)
        try {
            check(track.state == AudioTrack.STATE_INITIALIZED) { "音声出力を初期化できません" }
            // AudioTrackはバッファが開始しきい値まで埋まらないと再生を始めず、再生位置が終端に届かない。
            // しきい値より短い音声は末尾を無音で埋めて再生させる
            val playedFrames = maxOf(frames, startThresholdFrames(track, bufferBytes / (channels * 2)))
            val pcm = ShortArray(playedFrames * channels) {
                if (it < audio.size) (audio[it].coerceIn(-1f, 1f) * 32767).roundToInt().toShort() else 0
            }
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

    /** 再生を始めるのに必要なフレーム数。取得できない古いOSでは要求したバッファの大きさを使う。 */
    private fun startThresholdFrames(track: AudioTrack, requestedFrames: Int): Int = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> maxOf(track.startThresholdInFrames, track.bufferSizeInFrames)
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.M -> track.bufferSizeInFrames
        else -> requestedFrames
    }

    private fun waveform(clip: AudioUtil.WavFileData, offset: Int, onWaveform: (FloatArray) -> Unit) {
        // 無音で埋めた末尾を再生している間は波形を送らない
        if (offset >= clip.audioData.size) return
        val start = offset.coerceAtLeast(0)
        onWaveform(clip.audioData.copyOfRange(start,
            minOf(start + clip.sampleRate / 20 * clip.channels, clip.audioData.size)))
    }
}
