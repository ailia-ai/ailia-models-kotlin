package jp.axinc.ailia_kotlin

import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.nio.file.Files

import axip.ailia_speech.AiliaSpeech
import axip.ailia_speech.AiliaSpeechText

import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future

class AiliaSpeechSample {
    private var speech: AiliaSpeech? = null
    private var isInitialized = false
    private var lastTokenizationResult: String = ""

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

    fun initializeSpeech(): Boolean {
        val executor = Executors.newFixedThreadPool(2)

        return try {
            Log.i("AILIA_Main", "Begin model download")
            val encoderFuture: Future<String> = executor.submit(Callable {
                download("https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.opt3.onnx", "encoder_tiny.onnx")
            })

            val decoderFuture: Future<String> = executor.submit(Callable {
                download("https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt3.onnx", "decoder_tiny.onnx")
            })

            val encoder_path = encoderFuture.get() // Blocking call
            val decoder_path = decoderFuture.get() // Blocking call
            Log.i("AILIA_Main", "End model download")

            if (encoder_path == "" || decoder_path == ""){
                Log.e("AILIA_Main", "Model download failed")
            }

            if (isInitialized) {
                releaseSpeech()
            }

            Log.i("AILIA_Main", encoder_path)
            Log.i("AILIA_Main", decoder_path)

            speech = AiliaSpeech(AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE)
            speech?.openModel(encoder_path, decoder_path, AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_TINY)
            isInitialized = true
            Log.i("AILIA_Main", "Speech initialized successfully")
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to initialize speech: ${e.javaClass.name}: ${e.message}")
            releaseSpeech()
            false
        }
    }

    fun process(audio: FloatArray, channels: Int, sampleRate: Int) : String{
        speech?.pushInputData(audio, channels, audio.size / channels, sampleRate)
        speech?.finalizeInputData()
        speech?.transcribe()
        var count : Int? = speech?.getTextCount()
        if (count == null){
            return ""
        }
        var ret = ""
        for (i in 0 until count) {
            var text : AiliaSpeechText? = speech?.getText(i)
            if (text == null){
                continue
            }
            ret = ret + text.text + "\n"
        }
        speech?.resetTranscribeState()
        return ret
    }

    fun releaseSpeech() {
        try {
            speech?.close()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing speech: ${e.javaClass.name}: ${e.message}")
        } finally {
            speech = null
            isInitialized = false
            Log.i("AILIA_Main", "Speech released")
        }
    }
}
