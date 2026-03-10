package jp.axinc.ailia_kotlin

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import axip.ailia_voice.AiliaVoice
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_G2P_EN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_G2P_CN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_G2PW
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_ZH
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.net.URL

@RunWith(AndroidJUnit4::class)
class VoiceInstrumentedTest {
    companion object {
        private const val TAG = "VoiceTest"
    }

    private fun modelDirectory(): String {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return context.filesDir.absolutePath
    }

    @Before
    fun cleanupModelFiles() {
        val dir = File(modelDirectory())
        if (dir.exists()) {
            dir.listFiles()?.forEach { file ->
                if (file.name.endsWith(".onnx") || file.name == "user.dict") {
                    file.delete()
                }
                if (file.isDirectory && file.name.startsWith("gpt-sovits")) {
                    file.deleteRecursively()
                }
            }
        }
        Log.i(TAG, "Cleaned up model files")
    }

    private fun download(link: String, name: String): String {
        val dir = modelDirectory()
        val path = "$dir/$name"
        try {
            if (File(path).exists()) {
                return path
            }
            File(path).parentFile?.mkdirs()
            URL(link).openStream().use { input ->
                FileOutputStream(File(path)).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download Failed: $name", e)
            fail("Download failed for $name: ${e.message}")
        }
        return path
    }

    private fun downloadFiles(baseUrl: String, prefix: String, files: List<String>) {
        for (item in files) {
            download("$baseUrl/$item", "$prefix$item")
        }
    }

    private fun downloadBaseDictionaries() {
        Log.i(TAG, "Downloading base dictionaries...")
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/open_jtalk/open_jtalk_dic_utf_8-1.11", "",
            listOf("char.bin", "COPYING", "left-id.def", "matrix.bin", "pos-id.def",
                   "rewrite.def", "right-id.def", "sys.dic", "unk.dic")
        )
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/g2p_en", "",
            listOf("averaged_perceptron_tagger_classes.txt", "averaged_perceptron_tagger_tagdict.txt",
                   "averaged_perceptron_tagger_weights.txt", "cmudict", "g2p_decoder.onnx",
                   "g2p_encoder.onnx", "homographs.en")
        )
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v3", "",
            listOf("user.dict")
        )
        Log.i(TAG, "Base dictionaries downloaded")
    }

    private fun downloadChineseDictionaries() {
        Log.i(TAG, "Downloading Chinese dictionaries...")
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/g2p_cn", "g2p_cn/",
            listOf("pinyin.txt", "opencpop-strict.txt", "jieba.dict.utf8",
                   "hmm_model.utf8", "user.dict.utf8", "idf.utf8", "stop_words.utf8")
        )
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/g2pw/1.1", "g2pw/",
            listOf("g2pW.onnx", "POLYPHONIC_CHARS.txt", "bopomofo_to_pinyin_wo_tune_dict.json")
        )
        downloadFiles(
            "https://raw.githubusercontent.com/axinc-ai/ailia-models/master/audio_processing/gpt-sovits-v2/text/g2pw", "g2pw/",
            listOf("polyphonic.rep", "polyphonic-fix.rep")
        )
        Log.i(TAG, "Chinese dictionaries downloaded")
    }

    private fun openBaseDictionaries(voice: AiliaVoice) {
        val dir = modelDirectory()
        voice.setUserDictionaryFile("${dir}/user.dict", AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK)
        voice.openDictionaryFile(dir, AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK)
        voice.openDictionaryFile(dir, AILIA_VOICE_DICTIONARY_TYPE_G2P_EN)
    }

    private fun openChineseDictionaries(voice: AiliaVoice) {
        val dir = modelDirectory()
        voice.openDictionaryFile("${dir}/g2p_cn", AILIA_VOICE_DICTIONARY_TYPE_G2P_CN)
        voice.openDictionaryFile("${dir}/g2pw", AILIA_VOICE_DICTIONARY_TYPE_G2PW)
    }

    private fun loadReferenceAudio(): AudioUtil.WavFileData {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val resId = context.resources.getIdentifier("reference_audio_girl", "raw", context.packageName)
        assertTrue("Reference audio resource should exist", resId != 0)
        return AudioUtil().loadRawAudio(context.resources.openRawResource(resId))
    }

    private fun inferAndVerify(voice: AiliaVoice, refAudio: AudioUtil.WavFileData, refText: String, refLang: String, text: String, textLang: String, label: String) {
        val refG2pType = when (refLang) {
            "en" -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN
            "zh" -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_ZH
            else -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA
        }
        val textG2pType = when (textLang) {
            "en" -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN
            "zh" -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_ZH
            else -> AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA
        }

        val refFeatures = voice.g2p(refText, refG2pType)
        Log.i(TAG, "[$label] Ref features: $refFeatures")
        assertTrue("[$label] Ref features should not be empty", refFeatures.isNotEmpty())

        voice.setReferenceAudio(refAudio.audioData, refAudio.audioData.size * 4, refAudio.channels, refAudio.sampleRate, refFeatures)

        val textFeatures = voice.g2p(text, textG2pType)
        Log.i(TAG, "[$label] Text features: $textFeatures")
        assertTrue("[$label] Text features should not be empty", textFeatures.isNotEmpty())

        val startTime = System.nanoTime()
        val result = voice.synthesizeVoice(textFeatures)
        val elapsed = (System.nanoTime() - startTime) / 1000000
        Log.i(TAG, "[$label] Inference completed in ${elapsed}ms, samples=${result.data.size}, channels=${result.channels}, sampleRate=${result.samplingRate}")

        assertTrue("[$label] Audio data should not be empty", result.data.isNotEmpty())
        assertTrue("[$label] Sample rate should be positive", result.samplingRate > 0)
        assertTrue("[$label] Channels should be positive", result.channels > 0)
    }

    @Test
    fun testV1Japanese() {
        Log.i(TAG, "=== Test V1 Japanese ===")
        downloadBaseDictionaries()
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits", "",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt3.onnx", "vits.onnx", "cnhubert.onnx")
        )

        val dir = modelDirectory()
        val voice = AiliaVoice()
        try {
            openBaseDictionaries(voice)
            voice.openGPTSoVITSV1ModelFile(
                "${dir}/t2s_encoder.onnx", "${dir}/t2s_fsdec.onnx",
                "${dir}/t2s_sdec.opt3.onnx", "${dir}/vits.onnx", "${dir}/cnhubert.onnx"
            )

            val refAudio = loadReferenceAudio()
            inferAndVerify(voice, refAudio,
                "水をマレーシアから買わなくてはならない。", "ja",
                "こんにちは。今日はいい天気ですね。", "ja",
                "V1_JA")
        } finally {
            voice.close()
        }
    }

    @Test
    fun testV1English() {
        Log.i(TAG, "=== Test V1 English ===")
        downloadBaseDictionaries()
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits", "",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt3.onnx", "vits.onnx", "cnhubert.onnx")
        )

        val dir = modelDirectory()
        val voice = AiliaVoice()
        try {
            openBaseDictionaries(voice)
            voice.openGPTSoVITSV1ModelFile(
                "${dir}/t2s_encoder.onnx", "${dir}/t2s_fsdec.onnx",
                "${dir}/t2s_sdec.opt3.onnx", "${dir}/vits.onnx", "${dir}/cnhubert.onnx"
            )

            val refAudio = loadReferenceAudio()
            inferAndVerify(voice, refAudio,
                "水をマレーシアから買わなくてはならない。", "ja",
                "Hello world.", "en",
                "V1_EN")
        } finally {
            voice.close()
        }
    }

    @Test
    fun testV2Japanese() {
        Log.i(TAG, "=== Test V2 Japanese ===")
        downloadBaseDictionaries()
        downloadChineseDictionaries()
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v2", "gpt-sovits-v2/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "vits.onnx", "cnhubert.onnx", "chinese-roberta.onnx", "vocab.txt")
        )

        val dir = modelDirectory()
        val v2 = "${dir}/gpt-sovits-v2"
        val voice = AiliaVoice()
        try {
            openBaseDictionaries(voice)
            openChineseDictionaries(voice)
            voice.openGPTSoVITSV2ModelFile(
                "${v2}/t2s_encoder.onnx", "${v2}/t2s_fsdec.onnx", "${v2}/t2s_sdec.opt.onnx",
                "${v2}/vits.onnx", "${v2}/cnhubert.onnx",
                "${v2}/chinese-roberta.onnx", "${v2}/vocab.txt"
            )

            val refAudio = loadReferenceAudio()
            inferAndVerify(voice, refAudio,
                "水をマレーシアから買わなくてはならない。", "ja",
                "こんにちは。今日はいい天気ですね。", "ja",
                "V2_JA")
        } finally {
            voice.close()
        }
    }

    @Test
    fun testV2Chinese() {
        Log.i(TAG, "=== Test V2 Chinese ===")
        downloadBaseDictionaries()
        downloadChineseDictionaries()
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v2", "gpt-sovits-v2/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "vits.onnx", "cnhubert.onnx", "chinese-roberta.onnx", "vocab.txt")
        )

        val dir = modelDirectory()
        val v2 = "${dir}/gpt-sovits-v2"
        val voice = AiliaVoice()
        try {
            openBaseDictionaries(voice)
            openChineseDictionaries(voice)
            voice.openGPTSoVITSV2ModelFile(
                "${v2}/t2s_encoder.onnx", "${v2}/t2s_fsdec.onnx", "${v2}/t2s_sdec.opt.onnx",
                "${v2}/vits.onnx", "${v2}/cnhubert.onnx",
                "${v2}/chinese-roberta.onnx", "${v2}/vocab.txt"
            )

            val refAudio = loadReferenceAudio()
            inferAndVerify(voice, refAudio,
                "水をマレーシアから買わなくてはならない。", "ja",
                "今天天气真好。", "zh",
                "V2_ZH")
        } finally {
            voice.close()
        }
    }

    @Test
    fun testV3Japanese() {
        Log.i(TAG, "=== Test V3 Japanese ===")
        downloadBaseDictionaries()
        downloadChineseDictionaries()
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v3", "gpt-sovits-v3/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "cnhubert.onnx", "vq_model.onnx", "vq_cfm.onnx",
                   "bigvgan_model.onnx", "chinese-roberta.onnx", "vocab.txt")
        )

        val dir = modelDirectory()
        val v3 = "${dir}/gpt-sovits-v3"
        val voice = AiliaVoice()
        try {
            openBaseDictionaries(voice)
            openChineseDictionaries(voice)
            voice.openGPTSoVITSV3ModelFile(
                "${v3}/t2s_encoder.onnx", "${v3}/t2s_fsdec.onnx", "${v3}/t2s_sdec.opt.onnx",
                "${v3}/cnhubert.onnx", "${v3}/vq_model.onnx", "${v3}/vq_cfm.onnx",
                "${v3}/bigvgan_model.onnx", "${v3}/chinese-roberta.onnx", "${v3}/vocab.txt"
            )
            voice.setSampleSteps(4)

            val refAudio = loadReferenceAudio()
            inferAndVerify(voice, refAudio,
                "水をマレーシアから買わなくてはならない。", "ja",
                "こんにちは。今日はいい天気ですね。", "ja",
                "V3_JA")
        } finally {
            voice.close()
        }
    }

    @Test
    fun testV3Chinese() {
        Log.i(TAG, "=== Test V3 Chinese ===")
        downloadBaseDictionaries()
        downloadChineseDictionaries()
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v3", "gpt-sovits-v3/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "cnhubert.onnx", "vq_model.onnx", "vq_cfm.onnx",
                   "bigvgan_model.onnx", "chinese-roberta.onnx", "vocab.txt")
        )

        val dir = modelDirectory()
        val v3 = "${dir}/gpt-sovits-v3"
        val voice = AiliaVoice()
        try {
            openBaseDictionaries(voice)
            openChineseDictionaries(voice)
            voice.openGPTSoVITSV3ModelFile(
                "${v3}/t2s_encoder.onnx", "${v3}/t2s_fsdec.onnx", "${v3}/t2s_sdec.opt.onnx",
                "${v3}/cnhubert.onnx", "${v3}/vq_model.onnx", "${v3}/vq_cfm.onnx",
                "${v3}/bigvgan_model.onnx", "${v3}/chinese-roberta.onnx", "${v3}/vocab.txt"
            )
            voice.setSampleSteps(4)

            val refAudio = loadReferenceAudio()
            inferAndVerify(voice, refAudio,
                "水をマレーシアから買わなくてはならない。", "ja",
                "今天天气真好。", "zh",
                "V3_ZH")
        } finally {
            voice.close()
        }
    }

    @Test
    fun testV2ProJapanese() {
        Log.i(TAG, "=== Test V2-Pro Japanese ===")
        downloadBaseDictionaries()
        downloadChineseDictionaries()
        downloadFiles(
            "https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro", "gpt-sovits-v2-pro/",
            listOf("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.opt.onnx",
                   "cnhubert.onnx", "vits.onnx", "sv.onnx",
                   "chinese-roberta.onnx", "vocab.txt")
        )

        val dir = modelDirectory()
        val v2pro = "${dir}/gpt-sovits-v2-pro"
        val voice = AiliaVoice()
        try {
            openBaseDictionaries(voice)
            openChineseDictionaries(voice)
            voice.openGPTSoVITSV2ProModelFile(
                "${v2pro}/t2s_encoder.onnx", "${v2pro}/t2s_fsdec.onnx", "${v2pro}/t2s_sdec.opt.onnx",
                "${v2pro}/cnhubert.onnx", "${v2pro}/vits.onnx", "${v2pro}/sv.onnx",
                "${v2pro}/chinese-roberta.onnx", "${v2pro}/vocab.txt"
            )

            val refAudio = loadReferenceAudio()
            inferAndVerify(voice, refAudio,
                "水をマレーシアから買わなくてはならない。", "ja",
                "こんにちは。今日はいい天気ですね。", "ja",
                "V2PRO_JA")
        } finally {
            voice.close()
        }
    }
}
