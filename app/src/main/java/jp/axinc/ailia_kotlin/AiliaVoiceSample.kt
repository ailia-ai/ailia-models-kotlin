package axip.ailia_voice

import android.util.Log
import axip.ailia_voice.AiliaVoice
//import jp.axinc.ailia_kotlin.AiliaVoiceSample

class AiliaVoiceSample {
    companion object {
        private const val TAG = "AiliaVoiceSample"
        private var voce: AiliaVoice? = null
        private var isInitialized = false

        fun initializeVoice(): Boolean {
            return try {
                if (isInitialized) {
                    releaseVoice()
                }

                voice = AiliaVoice()

                val createResult = voice.create()
                if (!createResult) {
                    Log.e(AiliaTrackerSample.Companion.TAG, "Failed to initialize voice")
                    false
                }
                isInitialized = true
                Log.i(AiliaTrackerSample.Companion.TAG, "Voice initialized successfully")
                true
            } catch (e: Exception) {
                Log.e(AiliaTrackerSample.Companion.TAG, "Failed to initialize voice: ${e.javaClass.name}: ${e.message}")
                releaseTracker()
                false
            }
        }

        fun releaseVoice() {
            try {
                voice.close()
            } catch (e: Exception) {
                Log.e(AiliaTrackerSample.Companion.TAG, "Error releasing voice: ${e.javaClass.name}: ${e.message}")
            } finally {
                voice = null
                isInitialized = false
                trajectoryPoints.clear()
                Log.i(AiliaTrackerSample.Companion.TAG, "Voice released")
            }
        }

        fun textToSpeech() {
            Log.d(TAG, "Starting ailia Voice JNI sample")
            
            try {
                val testText = "こんにちは、世界"
                Log.d(TAG, "Processing text: $testText")
                
                val g2pResult = voice.graphemeToPhoneme(testText, AiliaVoice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)
                if (g2pResult != AiliaVoice.AILIA_STATUS_SUCCESS) {
                    Log.e(TAG, "Failed to perform G2P: $g2pResult")
                    val errorDetail = voice.getErrorDetail()
                    if (errorDetail != null) {
                        Log.e(TAG, "Error detail: $errorDetail")
                    }
                    return
                }
                Log.d(TAG, "G2P processing successful")
                
                val featureLength = voice.getFeatureLength()
                if (featureLength > 0) {
                    Log.d(TAG, "Feature length: $featureLength")
                    
                    val features = voice.getFeatures()
                    if (features != null) {
                        Log.d(TAG, "Features extracted successfully (length: ${features.length})")
                        Log.d(TAG, "Features preview: ${features.take(100)}...")
                    } else {
                        Log.e(TAG, "Failed to get features")
                    }
                } else {
                    Log.e(TAG, "Invalid feature length: $featureLength")
                }
                
                val inferenceResult = voice.inference(testText)
                if (inferenceResult != AiliaVoice.AILIA_STATUS_SUCCESS) {
                    Log.e(TAG, "Failed to perform inference: $inferenceResult")
                    val errorDetail = voice.getErrorDetail()
                    if (errorDetail != null) {
                        Log.e(TAG, "Error detail: $errorDetail")
                    }
                } else {
                    Log.d(TAG, "Inference successful")
                    
                    val waveInfo = voice.getWaveInfo()
                    if (waveInfo != null && waveInfo.size >= 3) {
                        val samples = waveInfo[0]
                        val channels = waveInfo[1]
                        val samplingRate = waveInfo[2]
                        
                        Log.d(TAG, "Wave info - Samples: $samples, Channels: $channels, Sampling rate: $samplingRate")
                        
                        if (samples > 0) {
                            val bufferSize = samples * channels * 4
                            val waveBuffer = FloatArray(samples * channels)
                            
                            val getWaveResult = voice.getWave(waveBuffer, bufferSize)
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
}
