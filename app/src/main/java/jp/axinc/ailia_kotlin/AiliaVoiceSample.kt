package jp.axinc.ailia_kotlin

import android.os.Environment
import android.util.Log
import axip.ailia_voice.AiliaVoice
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_CLEANER_TYPE_BASIC
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_G2P_EN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_MODEL_TYPE_GPT_SOVITS
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA

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
import kotlin.concurrent.thread

class AiliaVoiceSample {
    companion object {
        private const val TAG = "AILIA_Main"
        private var voice: AiliaVoice? = null
        private var isInitialized = false
    }

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
            URL(link).openStream().copyTo(FileOutputStream(File(path)))
        } catch (e: Exception) {
            Log.e("AILIA_Main", "Model Download Failed", e)
            return ""
        }
        return path
    }

    private fun downloadJapaneseDictionry() {
        val executor = Executors.newFixedThreadPool(2)

        val baseUrl = "https://storage.googleapis.com/ailia-models"
        val tasks = listOf(
            "char.bin",
            "COPYING",
            "left-id.def",
            "matrix.bin",
            "pos-id.def",
            "rewrite.def",
            "right-id.def",
            "sys.dic",
            "unk.dic"
        )

        val futures = mutableListOf<Future<String>>()

        for (item in tasks) {
            val url = "$baseUrl/open_jtalk/open_jtalk_dic_utf_8-1.11/$item"
            val future = executor.submit(Callable {
                download(url, item)
            })
            futures.add(future)
        }

        val openJTalkPaths = futures.map { it.get() }

        executor.shutdown()
    }

    private fun downloadEnglishDictionry() {
        val executor = Executors.newFixedThreadPool(2)

        val baseUrl = "https://storage.googleapis.com/ailia-models"
        val tasks = listOf(
            "averaged_perceptron_tagger_classes.txt",
            "averaged_perceptron_tagger_tagdict.txt",
            "averaged_perceptron_tagger_weights.txt",
            "cmudict",
            "g2p_decoder.onnx",
            "g2p_encoder.onnx",
            "homographs.en",
        )

        val futures = mutableListOf<Future<String>>()

        for (item in tasks) {
            val url = "$baseUrl/g2p_en/$item"
            val future = executor.submit(Callable {
                download(url, item)
            })
            futures.add(future)
        }

        val g2penPaths = futures.map { it.get() }

        executor.shutdown()
    }

    private fun downloadUserDictionry() {
        val executor = Executors.newFixedThreadPool(2)

        val baseUrl = "https://storage.googleapis.com/ailia-models"
        val tasks = listOf(
            "user.dict",
        )

        val futures = mutableListOf<Future<String>>()

        for (item in tasks) {
            val url = "$baseUrl/gpt-sovits-v3/$item"
            val future = executor.submit(Callable {
                download(url, item)
            })
            futures.add(future)
        }

        val userDictPaths = futures.map { it.get() }

        executor.shutdown()
    }

    private fun downloadModel() {
        val executor = Executors.newFixedThreadPool(2)

        val baseUrl = "https://storage.googleapis.com/ailia-models"

        val modelFutures = listOf(
            executor.submit(Callable {
                download("$baseUrl/gpt-sovits/t2s_encoder.onnx", "t2s_encoder.onnx")
            }),
            executor.submit(Callable {
                download("$baseUrl/gpt-sovits/t2s_fsdec.onnx", "t2s_fsdec.onnx")
            }),
            executor.submit(Callable {
                download("$baseUrl/gpt-sovits/t2s_sdec.opt3.onnx", "t2s_sdec.opt3.onnx")
            }),
            executor.submit(Callable {
                download("$baseUrl/gpt-sovits/vits.onnx", "vits.onnx")
            }),
            executor.submit(Callable {
                download("$baseUrl/gpt-sovits/cnhubert.onnx", "cnhubert.onnx")
            })
        )

        val modelPaths = modelFutures.map { it.get() }

        executor.shutdown()
    }

    fun initializeVoice(): Boolean {
        val executor = Executors.newFixedThreadPool(2)

        return try {
            Log.i("AILIA_Main", "Begin model download")

            downloadJapaneseDictionry()
            downloadEnglishDictionry()
            downloadUserDictionry()
            downloadModel()

            Log.i("AILIA_Main", "End model download")

            if (isInitialized) {
                releaseVoice()
            }

            voice = AiliaVoice()

            val dir: String = modelDirectory()
            voice?.setUserDictionaryFile(path = "${dir}/user.dict", AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK)
            voice?.openDictionaryFile(path = dir, dictionaryType = AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK)
            voice?.openDictionaryFile(path = dir, dictionaryType = AILIA_VOICE_DICTIONARY_TYPE_G2P_EN)

            val encoderPath = "${dir}/t2s_encoder.onnx"
            val decoderPath ="${dir}/t2s_fsdec.onnx"
            val postnetPath = "${dir}/t2s_sdec.opt3.onnx"
            val waveglowPath = "${dir}/vits.onnx"
            val sslPath = "${dir}/cnhubert.onnx"

            voice?.openModelFile(encoder = encoderPath, decoder1 = decoderPath, decoder2 = postnetPath, wave = waveglowPath, ssl = sslPath,
                          modelType = AILIA_VOICE_MODEL_TYPE_GPT_SOVITS,
                          cleanerType = AILIA_VOICE_CLEANER_TYPE_BASIC)

            isInitialized = true
            Log.i(AiliaVoiceSample.Companion.TAG, "Voice initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(AiliaVoiceSample.Companion.TAG, "Failed to initialize voice: ${e.javaClass.name}: ${e.message}")
            releaseVoice()
            false
        }
    }

    fun releaseVoice() {
        try {
            voice?.close()
        } catch (e: Exception) {
            Log.e(AiliaVoiceSample.Companion.TAG, "Error releasing voice: ${e.javaClass.name}: ${e.message}")
        } finally {
            voice = null
            isInitialized = false
            Log.i(AiliaVoiceSample.Companion.TAG, "Voice released")
        }
    }

    fun textToSpeech(audio: FloatArray, channels: Int, sampleRate: Int, refText: String, refLang: String, text: String, textLang: String) : Long{
        Log.d(AiliaVoiceSample.Companion.TAG, "Starting ailia Voice JNI sample")

        if (voice == null){
            return -1
        }

        try {
            var refG2pText : String = ""
            if (refLang == "en") {
                refG2pText = voice?.g2p(refText, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN)!!
            } else {
                refG2pText = voice?.g2p(refText, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)!!
            }
            Log.d(AiliaVoiceSample.Companion.TAG, "Ref text: $refText")
            Log.d(AiliaVoiceSample.Companion.TAG, "Ref Features: $refG2pText")
            voice?.setReferenceAudio(audio, audio.size * 4, channels, sampleRate, refG2pText)

            var g2pText : String = ""
            if (textLang == "en"){
                g2pText = voice?.g2p(text, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN)!!
            }else {
                g2pText = voice?.g2p(text, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)!!
            }
            Log.d(AiliaVoiceSample.Companion.TAG, "Text: $text")
            Log.d(AiliaVoiceSample.Companion.TAG, "Features: $g2pText")

            Log.d(AiliaVoiceSample.Companion.TAG, "Inference run")
            val startTime = System.nanoTime()
            val inferenceResult : AiliaVoice.AudioData = voice?.synthesizeVoice(g2pText)!!
            val endTime = System.nanoTime()
            Log.d(AiliaVoiceSample.Companion.TAG, "Inference result samples ${inferenceResult.data.size} channels ${inferenceResult.channels} sampleRate ${inferenceResult.samplingRate}")
            playAudio(inferenceResult.data, inferenceResult.channels, inferenceResult.samplingRate)
            return (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(AiliaVoiceSample.Companion.TAG, "Exception during sample execution", e)
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
                .setTransferMode(AudioTrack.MODE_STATIC) // 全サンプルを一度に渡すなら
                .setBufferSizeInBytes(bufferSize)
                .build()

            val written = audioTrack.write(pcm16, 0, pcm16.size)
            Log.d(AiliaVoiceSample.Companion.TAG, "Wrote $written samples")
            audioTrack.setVolume(1.0f)
            audioTrack.setNotificationMarkerPosition(pcm16.size)
            audioTrack.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack?) {
                    Log.d(AiliaVoiceSample.Companion.TAG, "Track Finish")
                    track?.stop()
                    track?.release()
                }

                override fun onPeriodicNotification(track: AudioTrack?) {}
            })
            audioTrack.play()
        } catch (e: Exception) {
            Log.e(AiliaVoiceSample.Companion.TAG, "Failed to play wave: ${e.message}", e)
        }
    }

}
