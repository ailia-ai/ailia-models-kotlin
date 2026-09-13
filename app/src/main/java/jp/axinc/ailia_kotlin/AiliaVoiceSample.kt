package jp.axinc.ailia_kotlin

import android.util.Log
import axip.ailia_voice.AiliaVoice
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_G2P_EN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_G2P_CN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_G2PW
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_ZH

import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import kotlin.math.roundToInt

enum class VoiceModelType(val displayName: String, val defaultText: String) {
    // V1は英語モデルのためデフォルトは英文
    GPT_SOVITS_V1("GPT-SoVITS V1", "Hello world. We will introduce ailia AI voice."),
    GPT_SOVITS_V2("GPT-SoVITS V2", "こんにちは。今日はいい天気ですね。"),
    GPT_SOVITS_V3("GPT-SoVITS V3", "こんにちは。今日はいい天気ですね。"),
    GPT_SOVITS_V2_PRO("GPT-SoVITS V2-Pro", "こんにちは。今日はいい天気ですね。"),
    GPT_SOVITS_V2_PRO_DISTILL_SMALL("GPT-SoVITS V2-Pro Distill Small", "こんにちは。今日はいい天気ですね。"),
    GPT_SOVITS_V2_PRO_DISTILL_BASE("GPT-SoVITS V2-Pro Distill Base", "こんにちは。今日はいい天気ですね。"),
}

class AiliaVoiceSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"

        /**
         * 入力テキストが日本語か英語かを判定してG2Pの言語("ja"/"en")を返す。
         * ひらがな/カタカナ/漢字が含まれていれば日本語と判定する。
         */
        fun detectLanguage(text: String): String {
            for (ch in text) {
                val block = Character.UnicodeBlock.of(ch)
                if (block == Character.UnicodeBlock.HIRAGANA ||
                    block == Character.UnicodeBlock.KATAKANA ||
                    block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) {
                    return "ja"
                }
            }
            return "en"
        }
    }

    private fun isDistillModel(type: VoiceModelType): Boolean =
        type == VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_SMALL ||
        type == VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE

    /** 蒸留モデルのt2sサイズ("small"/"base")を返す。 */
    private fun distillModelSize(type: VoiceModelType): String = when (type) {
        VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE -> "base"
        else -> "small"
    }

    private var voice: AiliaVoice? = null
    private var isInitialized = false
    private var audioTrack: AudioTrack? = null
    var modelType: VoiceModelType = VoiceModelType.GPT_SOVITS_V1
    private var downloadListener: ModelDownloadListener? = null

    // 直近の合成結果(波形表示用)
    var lastAudioData: FloatArray? = null
        private set
    var lastAudioChannels: Int = 1
        private set
    var lastAudioSampleRate: Int = 0
        private set

    private fun download(link: String, name: String): String {
        val file = ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(link, name), downloadListener)
            ?: error("Model download failed: $name")
        return file.absolutePath
    }

    private fun downloadFiles(baseUrl: String, prefix: String, files: List<String>) {
        for (item in files) {
            val url = "$baseUrl/$item"
            download(url, "$prefix$item")
        }
    }

    private fun downloadJapaneseDictionary() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/open_jtalk/open_jtalk_dic_utf_8-1.11",
            "",
            listOf("char.bin", "COPYING", "left-id.def", "matrix.bin", "pos-id.def",
                   "rewrite.def", "right-id.def", "sys.dic", "unk.dic")
        )
    }

    private fun downloadEnglishDictionary() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/g2p_en",
            "",
            listOf("averaged_perceptron_tagger_classes.txt", "averaged_perceptron_tagger_tagdict.txt",
                   "averaged_perceptron_tagger_weights.txt", "cmudict", "g2p_decoder.onnx",
                   "g2p_encoder.onnx", "homographs.en")
        )
    }

    private fun downloadChineseDictionary() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/g2p_cn",
            "g2p_cn/",
            listOf("pinyin.txt", "opencpop-strict.txt", "jieba.dict.utf8",
                   "hmm_model.utf8", "user.dict.utf8", "idf.utf8", "stop_words.utf8")
        )
    }

    private fun downloadG2pwDictionary() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/g2pw/1.1",
            "g2pw/",
            listOf("g2pW.onnx", "POLYPHONIC_CHARS.txt", "bopomofo_to_pinyin_wo_tune_dict.json")
        )
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/g2pw/gpt-sovits",
            "g2pw/",
            listOf("polyphonic.rep", "polyphonic-fix.rep")
        )
    }

    private fun downloadUserDictionary() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v3",
            "",
            listOf("user.dict")
        )
    }

    private fun downloadModelV1() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits",
            "",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt3.onnx",
                   "vits.onnx", "cnhubert.onnx")
        )
    }

    private fun downloadModelV2() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v2",
            "gpt-sovits-v2/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "vits.onnx", "cnhubert.onnx", "chinese-roberta.onnx", "vocab.txt")
        )
    }

    private fun downloadModelV3() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v3",
            "gpt-sovits-v3/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "cnhubert.onnx", "vq_model.onnx", "vq_cfm.onnx",
                   "bigvgan_model.onnx", "chinese-roberta.onnx", "vocab.txt")
        )
    }

    private fun downloadModelV2Pro() {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro",
            "gpt-sovits-v2-pro/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "cnhubert.onnx", "vits.onnx", "sv.onnx",
                   "chinese-roberta.onnx", "vocab.txt")
        )
    }

    /**
     * 蒸留モデル(small/base)のダウンロード。
     * t2s(encoder/fsdec/sdec)はサイズ別、vitsは常にvits_distill_small.onnxを使用する。
     */
    private fun downloadModelV2ProDistill(size: String) {
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro-distill",
            "gpt-sovits-v2-pro-distill/",
            listOf("t2s_encoder_distill_${size}.onnx", "t2s_fsdec_distill_${size}.onnx",
                   "t2s_sdec_distill_${size}.opt.onnx", "vits_distill_small.onnx")
        )
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro",
            "gpt-sovits-v2-pro/",
            listOf("cnhubert.onnx", "sv.onnx")
        )
    }

    fun initializeVoice(envId: Int = -1, listener: ModelDownloadListener? = null): Boolean {
        this.downloadListener = listener
        return try {
            Log.i(TAG, "Begin model download for $modelType")

            downloadJapaneseDictionary()
            downloadEnglishDictionary()
            downloadChineseDictionary()
            downloadG2pwDictionary()
            downloadUserDictionary()

            when (modelType) {
                VoiceModelType.GPT_SOVITS_V1 -> downloadModelV1()
                VoiceModelType.GPT_SOVITS_V2 -> downloadModelV2()
                VoiceModelType.GPT_SOVITS_V3 -> downloadModelV3()
                VoiceModelType.GPT_SOVITS_V2_PRO -> downloadModelV2Pro()
                VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_SMALL -> downloadModelV2ProDistill("small")
                VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE -> downloadModelV2ProDistill("base")
            }

            Log.i(TAG, "End model download")

            if (isInitialized) {
                releaseVoice()
            }

            Log.i(TAG, "Initializing voice with envId=$envId")
            voice = AiliaVoice(envId = envId)

            val dir = modelDirectory.absolutePath
            voice?.setUserDictionaryFile(path = "${dir}/user.dict", AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK)
            voice?.openDictionaryFile(path = dir, dictionaryType = AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK)
            voice?.openDictionaryFile(path = dir, dictionaryType = AILIA_VOICE_DICTIONARY_TYPE_G2P_EN)
            voice?.openDictionaryFile(path = "${dir}/g2p_cn", dictionaryType = AILIA_VOICE_DICTIONARY_TYPE_G2P_CN)
            voice?.openDictionaryFile(path = "${dir}/g2pw", dictionaryType = AILIA_VOICE_DICTIONARY_TYPE_G2PW)

            when (modelType) {
                VoiceModelType.GPT_SOVITS_V1 -> {
                    voice?.openGPTSoVITSV1ModelFile(
                        encoder = "${dir}/t2s_encoder.onnx",
                        decoder1 = "${dir}/t2s_fsdec.onnx",
                        decoder2 = "${dir}/t2s_sdec.opt3.onnx",
                        wave = "${dir}/vits.onnx",
                        ssl = "${dir}/cnhubert.onnx"
                    )
                }
                VoiceModelType.GPT_SOVITS_V2 -> {
                    val v2 = "${dir}/gpt-sovits-v2"
                    voice?.openGPTSoVITSV2ModelFile(
                        encoder = "${v2}/t2s_encoder.onnx",
                        decoder1 = "${v2}/t2s_fsdec.onnx",
                        decoder2 = "${v2}/t2s_sdec.opt.onnx",
                        wave = "${v2}/vits.onnx",
                        ssl = "${v2}/cnhubert.onnx",
                        chineseBert = "${v2}/chinese-roberta.onnx",
                        vocab = "${v2}/vocab.txt"
                    )
                }
                VoiceModelType.GPT_SOVITS_V3 -> {
                    val v3 = "${dir}/gpt-sovits-v3"
                    voice?.openGPTSoVITSV3ModelFile(
                        encoder = "${v3}/t2s_encoder.onnx",
                        decoder1 = "${v3}/t2s_fsdec.onnx",
                        decoder2 = "${v3}/t2s_sdec.opt.onnx",
                        ssl = "${v3}/cnhubert.onnx",
                        vq = "${v3}/vq_model.onnx",
                        cfm = "${v3}/vq_cfm.onnx",
                        bigvgan = "${v3}/bigvgan_model.onnx",
                        chineseBert = "${v3}/chinese-roberta.onnx",
                        vocab = "${v3}/vocab.txt"
                    )
                    voice?.setSampleSteps(4)
                }
                VoiceModelType.GPT_SOVITS_V2_PRO -> {
                    val v2pro = "${dir}/gpt-sovits-v2-pro"
                    voice?.openGPTSoVITSV2ProModelFile(
                        encoder = "${v2pro}/t2s_encoder.onnx",
                        decoder1 = "${v2pro}/t2s_fsdec.onnx",
                        decoder2 = "${v2pro}/t2s_sdec.opt.onnx",
                        ssl = "${v2pro}/cnhubert.onnx",
                        vits = "${v2pro}/vits.onnx",
                        sv = "${v2pro}/sv.onnx",
                        chineseBert = "${v2pro}/chinese-roberta.onnx",
                        vocab = "${v2pro}/vocab.txt"
                    )
                }
                VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_SMALL,
                VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE -> {
                    val size = distillModelSize(modelType)
                    val distill = "${dir}/gpt-sovits-v2-pro-distill"
                    val v2pro = "${dir}/gpt-sovits-v2-pro"
                    // vitsはsmall/baseどちらもvits_distill_small.onnxで固定
                    voice?.openGPTSoVITSV2ProModelFile(
                        encoder = "${distill}/t2s_encoder_distill_${size}.onnx",
                        decoder1 = "${distill}/t2s_fsdec_distill_${size}.onnx",
                        decoder2 = "${distill}/t2s_sdec_distill_${size}.opt.onnx",
                        ssl = "${v2pro}/cnhubert.onnx",
                        vits = "${distill}/vits_distill_small.onnx",
                        sv = "${v2pro}/sv.onnx",
                        chineseBert = null,
                        vocab = null
                    )
                }
            }

            isInitialized = true
            Log.i(TAG, "Voice initialized successfully with $modelType")
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize voice: ${e.javaClass.name}: ${e.message}")
            listener?.onError(e.message ?: "Unknown error")
            releaseVoice()
            false
        } finally {
            this.downloadListener = null
        }
    }

    fun releaseVoice() {
        releaseAudioTrack()
        try {
            voice?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing voice: ${e.javaClass.name}: ${e.message}")
        } finally {
            voice = null
            isInitialized = false
            Log.i(TAG, "Voice released")
        }
    }

    private fun g2pTypeForLang(lang: String): Int {
        return when (lang) {
            "en" -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN
            "zh" -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_ZH
            else -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA
        }
    }

    fun textToSpeech(audio: FloatArray, channels: Int, sampleRate: Int, refText: String, refLang: String, text: String, textLang: String) : Long {
        Log.d(TAG, "Starting ailia Voice JNI sample ($modelType)")

        if (voice == null) {
            return -1
        }

        try {
            val refG2pText = voice?.g2p(refText, g2pTypeForLang(refLang))!!
            Log.d(TAG, "Ref text: $refText")
            Log.d(TAG, "Ref Features: $refG2pText")
            voice?.setReferenceAudio(audio, audio.size * 4, channels, sampleRate, refG2pText)

            val targetLanguage = if (isDistillModel(modelType)) "ja" else textLang
            val g2pText = voice?.g2p(text, g2pTypeForLang(targetLanguage))!!
            Log.d(TAG, "Text: $text")
            Log.d(TAG, "Features: $g2pText")

            Log.d(TAG, "Inference run")
            val startTime = System.nanoTime()
            val inferenceResult: AiliaVoice.AudioData = voice?.synthesizeVoice(g2pText)!!
            val endTime = System.nanoTime()
            Log.d(TAG, "Inference result samples ${inferenceResult.data.size} channels ${inferenceResult.channels} sampleRate ${inferenceResult.samplingRate}")
            lastAudioData = inferenceResult.data
            lastAudioChannels = inferenceResult.channels
            lastAudioSampleRate = inferenceResult.samplingRate
            playAudio(inferenceResult.data, inferenceResult.channels, inferenceResult.samplingRate)
            return (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Exception during sample execution", e)
        }
        return -1
    }

    fun playAudio(waveBuffer: FloatArray, channels: Int, samplingRate: Int) {
        try {
            releaseAudioTrack()
            val channelConfig = if (channels == 1)
                AudioFormat.CHANNEL_OUT_MONO
            else
                AudioFormat.CHANNEL_OUT_STEREO

            val pcm16 = ShortArray(waveBuffer.size) { i ->
                (waveBuffer[i].coerceIn(-1.0f, 1.0f) * Short.MAX_VALUE).roundToInt().toShort()
            }

            val minBuffer = AudioTrack.getMinBufferSize(
                samplingRate,
                channelConfig,
                AudioFormat.ENCODING_PCM_16BIT
            )
            val bufferSize = maxOf(minBuffer, pcm16.size * 2)

            val track = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(samplingRate)
                            .setChannelMask(channelConfig)
                            .build()
                    )
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .setBufferSizeInBytes(bufferSize)
                    .build()
            } else {
                @Suppress("DEPRECATION")
                AudioTrack(
                    AudioManager.STREAM_MUSIC,
                    samplingRate,
                    channelConfig,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize,
                    AudioTrack.MODE_STATIC,
                )
            }
            audioTrack = track

            val written = track.write(pcm16, 0, pcm16.size)
            Log.d(TAG, "Wrote $written samples")
            check(written == pcm16.size) { "AudioTrack write failed: $written/${pcm16.size}" }
            track.setVolume(1.0f)
            track.setNotificationMarkerPosition(pcm16.size / channels.coerceAtLeast(1))
            track.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(completedTrack: AudioTrack?) {
                    Log.d(TAG, "Track Finish")
                    if (audioTrack === completedTrack) releaseAudioTrack()
                }

                override fun onPeriodicNotification(completedTrack: AudioTrack?) {}
            })
            track.play()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to play wave: ${e.message}", e)
            releaseAudioTrack()
        }
    }

    @Synchronized
    private fun releaseAudioTrack() {
        val track = audioTrack ?: return
        audioTrack = null
        try {
            if (track.playState == AudioTrack.PLAYSTATE_PLAYING) track.stop()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to stop AudioTrack: ${e.message}")
        }
        try {
            track.release()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to release AudioTrack: ${e.message}")
        }
    }
}
