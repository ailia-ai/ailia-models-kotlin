package jp.axinc.ailia_kotlin

import android.os.Environment
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
import java.io.FileOutputStream
import java.net.URL
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlin.math.roundToInt

enum class VoiceModelType {
    GPT_SOVITS_V1,
    GPT_SOVITS_V2,
    GPT_SOVITS_V3,
    GPT_SOVITS_V2_PRO,
}

class AiliaVoiceSample {
    companion object {
        private const val TAG = "AILIA_Main"
        private var voice: AiliaVoice? = null
        private var isInitialized = false
    }

    var modelType: VoiceModelType = VoiceModelType.GPT_SOVITS_V1

    private fun modelDirectory() : String{
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath
    }

    private fun download(link: String, name: String): String {
        val dir: String = modelDirectory()
        val path: String = "$dir/$name"
        try {
            if (File(path).exists()) {
                return path
            }
            File(path).parentFile?.mkdirs()
            URL(link).openStream().copyTo(FileOutputStream(File(path)))
        } catch (e: Exception) {
            Log.e("AILIA_Main", "Model Download Failed: $name", e)
            return ""
        }
        return path
    }

    private fun downloadFiles(baseUrl: String, prefix: String, files: List<String>) {
        val executor = Executors.newFixedThreadPool(2)
        val futures = mutableListOf<Future<String>>()
        for (item in files) {
            val url = "$baseUrl/$item"
            val future = executor.submit(Callable {
                download(url, "$prefix$item")
            })
            futures.add(future)
        }
        futures.map { it.get() }
        executor.shutdown()
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
            "https://raw.githubusercontent.com/axinc-ai/ailia-models/master/audio_processing/gpt-sovits-v2/text/g2pw",
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

    fun initializeVoice(): Boolean {
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
            }

            Log.i(TAG, "End model download")

            if (isInitialized) {
                releaseVoice()
            }

            voice = AiliaVoice()

            val dir: String = modelDirectory()
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
            }

            isInitialized = true
            Log.i(TAG, "Voice initialized successfully with $modelType")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize voice: ${e.javaClass.name}: ${e.message}")
            releaseVoice()
            false
        }
    }

    fun releaseVoice() {
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

            val g2pText = voice?.g2p(text, g2pTypeForLang(textLang))!!
            Log.d(TAG, "Text: $text")
            Log.d(TAG, "Features: $g2pText")

            Log.d(TAG, "Inference run")
            val startTime = System.nanoTime()
            val inferenceResult: AiliaVoice.AudioData = voice?.synthesizeVoice(g2pText)!!
            val endTime = System.nanoTime()
            Log.d(TAG, "Inference result samples ${inferenceResult.data.size} channels ${inferenceResult.channels} sampleRate ${inferenceResult.samplingRate}")
            playAudio(inferenceResult.data, inferenceResult.channels, inferenceResult.samplingRate)
            return (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Exception during sample execution", e)
        }
        return 0
    }

    fun playAudio(waveBuffer: FloatArray, channels: Int, samplingRate: Int) {
        try {
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

            val audioTrack = AudioTrack.Builder()
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

            val written = audioTrack.write(pcm16, 0, pcm16.size)
            Log.d(TAG, "Wrote $written samples")
            audioTrack.setVolume(1.0f)
            audioTrack.setNotificationMarkerPosition(pcm16.size)
            audioTrack.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack?) {
                    Log.d(TAG, "Track Finish")
                    track?.stop()
                    track?.release()
                }

                override fun onPeriodicNotification(track: AudioTrack?) {}
            })
            audioTrack.play()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to play wave: ${e.message}", e)
        }
    }
}
