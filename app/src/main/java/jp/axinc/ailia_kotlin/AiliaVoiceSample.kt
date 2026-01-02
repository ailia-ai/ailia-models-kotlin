package jp.axinc.ailia_kotlin

import android.os.Environment
import android.util.Log
import axip.ailia_speech.AiliaSpeech
import axip.ailia_voice.AiliaVoice
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_CLEANER_TYPE_BASIC
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_MODEL_TYPE_TACOTRON2
import axip.ailia_voice.AiliaVoice.Companion.AILIA_VOICE_MODEL_TYPE_GPT_SOVITS

import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future

class AiliaVoiceSample {
    companion object {
        private const val TAG = "AiliaVoiceSample"
        private var voice: AiliaVoice? = null
        private var isInitialized = false
    }

    fun download(link: String, name: String): String {
        val dir: String = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath
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

    fun initializeVoice(): Boolean {
        val executor = Executors.newFixedThreadPool(2)

        return try {
            Log.i("AILIA_Main", "Begin model download")

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

            // 結果を受け取る（ブロッキング）
            val openJTalkPaths = futures.map { it.get() }
            val encoderPath = modelFutures[0].get()
            val decoderPath = modelFutures[1].get()
            val postnetPath = modelFutures[2].get()
            val waveglowPath = modelFutures[3].get()
            val sslPath = modelFutures[4].get()

            executor.shutdown()

            println("OpenJTalk Files: $openJTalkPaths")
            println("encoderPath = $encoderPath")
            println("decoderPath = $decoderPath")
            println("postnetPath = $postnetPath")
            println("waveglowPath = $waveglowPath")
            println("sslPath = $sslPath")

            Log.i("AILIA_Main", "End model download")

            if (isInitialized) {
                releaseVoice()
            }

            voice = AiliaVoice()

            val createResult = voice?.open()
            if (!createResult!!) {
                Log.e(AiliaVoiceSample.Companion.TAG, "Failed to initialize voice")
                false
            }

            val dir: String = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath
            val status = voice?.openDictionaryFile(path = dir, dictionaryType = AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK)
            if (status!! != 0) {
                Log.e(AiliaVoiceSample.Companion.TAG, "Failed to openDictionaryFile")
                false
            }

            val status2 =voice?.openModelFile(encoder = encoderPath, decoder1 = decoderPath, decoder2 = postnetPath, wave = waveglowPath, ssl = sslPath,
                              modelType = AILIA_VOICE_MODEL_TYPE_GPT_SOVITS,
                              cleanerType = AILIA_VOICE_CLEANER_TYPE_BASIC)
            if (status2!! != 0) {
                Log.e(AiliaVoiceSample.Companion.TAG, "Failed to openModelFile")
                false
            }

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

    fun textToSpeech() {
        Log.d(TAG, "Starting ailia Voice JNI sample")

        if (voice == null){
            return;
        }

        try {
            val testText = "こんにちは、世界"
            Log.d(TAG, "Processing text: $testText")

            val g2pResult = voice?.graphemeToPhoneme(testText, AiliaVoice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)
            if (g2pResult != AiliaVoice.AILIA_STATUS_SUCCESS) {
                Log.e(TAG, "Failed to perform G2P: $g2pResult")
                val errorDetail = voice?.getErrorDetail()
                if (errorDetail != null) {
                    Log.e(TAG, "Error detail: $errorDetail")
                }
                return
            }
            Log.d(TAG, "G2P processing successful")

            val featureLength = voice?.getFeatureLength()
            if (featureLength!! > 0) {
                Log.d(TAG, "Feature length: $featureLength")

                val features = voice?.getFeatures()
                if (features != null) {
                    Log.d(TAG, "Features extracted successfully (length: ${features.length})")
                    Log.d(TAG, "Features preview: ${features.take(100)}...")
                } else {
                    Log.e(TAG, "Failed to get features")
                }
            } else {
                Log.e(TAG, "Invalid feature length: $featureLength")
            }

            val inferenceResult = voice?.inference(testText)
            if (inferenceResult != AiliaVoice.AILIA_STATUS_SUCCESS) {
                Log.e(TAG, "Failed to perform inference: $inferenceResult")
                val errorDetail = voice?.getErrorDetail()
                if (errorDetail != null) {
                    Log.e(TAG, "Error detail: $errorDetail")
                }
            } else {
                Log.d(TAG, "Inference successful")

                val waveInfo = voice?.getWaveInfo()
                if (waveInfo != null && waveInfo.size >= 3) {
                    val samples = waveInfo[0]
                    val channels = waveInfo[1]
                    val samplingRate = waveInfo[2]

                    Log.d(TAG, "Wave info - Samples: $samples, Channels: $channels, Sampling rate: $samplingRate")

                    if (samples > 0) {
                        val bufferSize = samples * channels * 4
                        val waveBuffer = FloatArray(samples * channels)

                        val getWaveResult = voice?.getWave(waveBuffer, bufferSize)
                        if (getWaveResult == AiliaVoice.AILIA_STATUS_SUCCESS) {
                            Log.d(TAG, "Successfully retrieved wave data (${waveBuffer.size} samples)")
                            Log.d(TAG, "First few samples: ${waveBuffer.take(10).joinToString(", ")}")
                        } else {
                            Log.e(TAG, "Failed to get wave data: $getWaveResult")
                        }
                    }
                } else {
                    Log.e(TAG, "Failed to get wave info")
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Exception during sample execution", e)
        } finally {
            Log.d(TAG, "ailia Voice instance closed")
        }

        Log.d(TAG, "ailia Voice JNI sample completed")
    }
}
