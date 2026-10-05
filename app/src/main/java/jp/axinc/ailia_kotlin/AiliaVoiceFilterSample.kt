package jp.axinc.ailia_kotlin

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

data class VoiceFilterEmbeddingResult(
    val embedding: FloatArray,
    val referenceDurationMs: Long,
    val processingTimeMs: Long,
)

data class VoiceFilterResult(
    val inputAudio: FloatArray,
    val outputAudio: FloatArray,
    val sampleRate: Int,
    val processingTimeMs: Long,
)

/**
 * Standalone port of ailia-models/audio_processing/voicefilter.
 *
 * The class downloads and owns the VoiceFilter mask model and its dynamic d-vector
 * embedder together with Silero VAD v6. It also exposes a small microphone
 * recorder and PCM player so the sample can be copied independently of MainActivity.
 */
class AiliaVoiceFilterSample(private val modelDirectory: File) {
    interface RecordingListener {
        fun onWaveform(samples: FloatArray, sampleRate: Int)
        fun onCompleted(audio: FloatArray, sampleRate: Int)
        fun onError(error: String)
    }

    companion object {
        private const val TAG = "AILIA_VoiceFilter"
        private const val REMOTE_PATH = "https://storage.googleapis.com/ailia-models/voicefilter/"
        private const val FILTER_MODEL_URL = "${REMOTE_PATH}model.onnx"
        private const val FILTER_MODEL_FILE = "voicefilter_model.onnx"
        private const val FILTER_PROTO_URL = "${REMOTE_PATH}model.onnx.prototxt"
        private const val FILTER_PROTO_FILE = "voicefilter_model.onnx.prototxt"
        private const val EMBEDDER_MODEL_URL = "${REMOTE_PATH}embedder_dynamic.onnx"
        private const val EMBEDDER_MODEL_FILE = "voicefilter_embedder_dynamic.onnx"
        private const val EMBEDDER_PROTO_URL = "${REMOTE_PATH}embedder_dynamic.onnx.prototxt"
        private const val EMBEDDER_PROTO_FILE = "voicefilter_embedder_dynamic.onnx.prototxt"
        private const val VAD_MODEL_URL =
            "https://storage.googleapis.com/ailia-models/silero-vad/silero_vad_v6.onnx"
        private const val VAD_MODEL_FILE = "silero_vad_v6.onnx"
        private const val VAD_PROTO_URL =
            "https://storage.googleapis.com/ailia-models/silero-vad/silero_vad_v6.onnx.prototxt"
        private const val VAD_PROTO_FILE = "silero_vad_v6.onnx.prototxt"
        private const val VAD_WINDOW_SIZE = 512
        private const val VAD_CONTEXT_SIZE = 64
        private const val VAD_STATE_SIZE = 2 * 1 * 128
        private const val EMBEDDING_SIZE = 256
        private const val MAX_RECORDING_SECONDS = 30

        init {
            System.loadLibrary("ailia")
        }
    }

    private var filterModel: AiliaModel? = null
    private var embedderModel: AiliaModel? = null
    private var vadModel: AiliaModel? = null
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var ortFilterSession: OrtSession? = null
    private var ortEmbedderSession: OrtSession? = null
    private var ortVadSession: OrtSession? = null
    @Volatile private var initialized = false
    @Volatile var lastErrorMessage: String? = null
        private set

    private val recording = AtomicBoolean(false)
    private val cancelRecording = AtomicBoolean(false)
    private var audioRecord: AudioRecord? = null
    private var recordingExecutor: ExecutorService? = null
    private var audioTrack: AudioTrack? = null

    val isRecording: Boolean
        get() = recording.get()

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            listOf(
                ModelFileSpec(FILTER_PROTO_URL, FILTER_PROTO_FILE),
                ModelFileSpec(FILTER_MODEL_URL, FILTER_MODEL_FILE),
                ModelFileSpec(EMBEDDER_PROTO_URL, EMBEDDER_PROTO_FILE),
                ModelFileSpec(EMBEDDER_MODEL_URL, EMBEDDER_MODEL_FILE),
                ModelFileSpec(VAD_PROTO_URL, VAD_PROTO_FILE),
                ModelFileSpec(VAD_MODEL_URL, VAD_MODEL_FILE),
            ).forEach { spec ->
                check(ModelDownloader.downloadFile(modelDirectory, spec, listener) != null)
            }
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initialize(envId: Int): Boolean {
        releaseModels()
        lastErrorMessage = null
        return try {
            filterModel = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                File(modelDirectory, FILTER_PROTO_FILE).absolutePath,
                File(modelDirectory, FILTER_MODEL_FILE).absolutePath,
            )
            embedderModel = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                File(modelDirectory, EMBEDDER_PROTO_FILE).absolutePath,
                File(modelDirectory, EMBEDDER_MODEL_FILE).absolutePath,
            )
            vadModel = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                File(modelDirectory, VAD_PROTO_FILE).absolutePath,
                File(modelDirectory, VAD_MODEL_FILE).absolutePath,
            )
            initialized = true
            true
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            lastErrorMessage = "VoiceFilter initialization failed: ${errorReason(e)}"
            releaseModels()
            false
        }
    }

    fun initializeOnnxRuntime(): Boolean {
        releaseModels()
        lastErrorMessage = null
        return try {
            val options = OrtSession.SessionOptions()
            try {
                ortFilterSession = ortEnvironment.createSession(
                    File(modelDirectory, FILTER_MODEL_FILE).absolutePath,
                    options,
                )
                ortEmbedderSession = ortEnvironment.createSession(
                    File(modelDirectory, EMBEDDER_MODEL_FILE).absolutePath,
                    options,
                )
                ortVadSession = ortEnvironment.createSession(
                    File(modelDirectory, VAD_MODEL_FILE).absolutePath,
                    options,
                )
            } finally {
                options.close()
            }
            initialized = true
            Log.i(TAG, "VoiceFilter and Silero VAD initialized with ONNX Runtime CPU")
            true
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime initialization failed", e)
            lastErrorMessage = "VoiceFilter ONNX Runtime initialization failed: ${errorReason(e)}"
            releaseModels()
            false
        }
    }

    fun createEmbedding(
        interleavedAudio: FloatArray,
        channels: Int,
        sampleRate: Int,
    ): VoiceFilterEmbeddingResult {
        check(initialized) { "VoiceFilter is not initialized" }
        val startedAt = System.nanoTime()
        val mono = VoiceFilterAudio.toMono16k(interleavedAudio, channels, sampleRate)
        require(mono.isNotEmpty()) { "Reference audio is empty" }
        val speech = separateSpeech(mono)
        val embedderAudio = VoiceFilterAudio.padForDynamicEmbedder(speech)
        val mel = VoiceFilterAudio.melSpectrogram(embedderAudio)
        val frameCount = mel.size / VoiceFilterAudio.MEL_BINS
        lastErrorMessage = null
        val embedding = ortEmbedderSession?.let { runEmbedder(it, mel, frameCount) }
            ?: runEmbedder(checkNotNull(embedderModel), mel, frameCount)
        return VoiceFilterEmbeddingResult(
            embedding = embedding,
            referenceDurationMs = speech.size * 1_000L / VoiceFilterAudio.SAMPLE_RATE,
            processingTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
        )
    }

    private fun runEmbedder(model: AiliaModel, mel: FloatArray, frameCount: Int): FloatArray {
        return try {
            val inputIndex = Ailia.FindBlobIndexByName(model.handle, "dvec_mel")
            model.setInputBlobShapeND(intArrayOf(VoiceFilterAudio.MEL_BINS, frameCount), inputIndex)
            model.setInputBlobData(mel, mel.size * Float.SIZE_BYTES, inputIndex)
            model.update()
            FloatArray(EMBEDDING_SIZE).also { embedding ->
                model.getBlobData(
                    embedding,
                    embedding.size * Float.SIZE_BYTES,
                    Ailia.FindBlobIndexByName(model.handle, "dvec"),
                )
            }
        } catch (e: Exception) {
            val detail = runCatching { Ailia.GetErrorDetail(model.handle) }
                .getOrNull()
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
            val reason = detail ?: errorReason(e)
            lastErrorMessage = "VoiceFilter embedder inference failed: $reason"
            throw IllegalStateException(lastErrorMessage, e)
        }
    }

    private fun runEmbedder(session: OrtSession, mel: FloatArray, frameCount: Int): FloatArray {
        return try {
            val tensor = createFloatTensor(
                ortEnvironment,
                mel,
                longArrayOf(VoiceFilterAudio.MEL_BINS.toLong(), frameCount.toLong()),
            )
            try {
                runFloatTensor(session, mapOf("dvec_mel" to tensor), "dvec")
            } finally {
                tensor.close()
            }
        } catch (e: Exception) {
            lastErrorMessage = "VoiceFilter embedder inference failed: ${errorReason(e)}"
            throw IllegalStateException(lastErrorMessage, e)
        }
    }

    private fun errorReason(error: Exception): String =
        error.message?.takeIf { it.isNotBlank() } ?: error.javaClass.simpleName

    fun filter(
        referenceEmbedding: FloatArray,
        interleavedAudio: FloatArray,
        channels: Int,
        sampleRate: Int,
        applyVad: Boolean = false,
    ): VoiceFilterResult {
        check(initialized) { "VoiceFilter is not initialized" }
        require(referenceEmbedding.size == EMBEDDING_SIZE) { "Invalid VoiceFilter speaker embedding" }
        val startedAt = System.nanoTime()
        val mono = VoiceFilterAudio.toMono16k(interleavedAudio, channels, sampleRate)
        require(mono.isNotEmpty()) { "Input audio is empty" }
        val filterInput = if (applyVad) separateSpeech(mono) else mono
        require(filterInput.size >= VoiceFilterAudio.SAMPLE_RATE / 2) {
            "At least 0.5 seconds of input audio is required"
        }
        val spectrum = VoiceFilterAudio.analysisSpectrum(filterInput)
        val mask = ortFilterSession?.let { session ->
            val magnitudeTensor = createFloatTensor(
                ortEnvironment,
                spectrum.magnitude,
                longArrayOf(
                    1,
                    spectrum.frameCount.toLong(),
                    VoiceFilterAudio.FILTER_FREQUENCY_BINS.toLong(),
                ),
            )
            val embeddingTensor = createFloatTensor(
                ortEnvironment,
                referenceEmbedding,
                longArrayOf(1, EMBEDDING_SIZE.toLong()),
            )
            try {
                runFloatTensor(
                    session,
                    mapOf("mag" to magnitudeTensor, "dvec" to embeddingTensor),
                    "mask",
                )
            } finally {
                closeTensors(magnitudeTensor, embeddingTensor)
            }
        } ?: run {
            val model = checkNotNull(filterModel)
            val magnitudeIndex = Ailia.FindBlobIndexByName(model.handle, "mag")
            val embeddingIndex = Ailia.FindBlobIndexByName(model.handle, "dvec")
            model.setInputBlobShapeND(
                intArrayOf(1, spectrum.frameCount, VoiceFilterAudio.FILTER_FREQUENCY_BINS),
                magnitudeIndex,
            )
            model.setInputBlobShapeND(intArrayOf(1, EMBEDDING_SIZE), embeddingIndex)
            model.setInputBlobData(
                spectrum.magnitude,
                spectrum.magnitude.size * Float.SIZE_BYTES,
                magnitudeIndex,
            )
            model.setInputBlobData(
                referenceEmbedding,
                referenceEmbedding.size * Float.SIZE_BYTES,
                embeddingIndex,
            )
            model.update()
            FloatArray(spectrum.magnitude.size).also {
                model.getBlobData(
                    it,
                    it.size * Float.SIZE_BYTES,
                    Ailia.FindBlobIndexByName(model.handle, "mask"),
                )
            }
        }
        val output = VoiceFilterAudio.reconstruct(spectrum, mask)
        return VoiceFilterResult(
            inputAudio = filterInput.copyOf(output.size),
            outputAudio = output,
            sampleRate = VoiceFilterAudio.SAMPLE_RATE,
            processingTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
        )
    }

    /** Removes non-speech regions with the same Silero VAD v6 settings as WeSpeaker. */
    private fun separateSpeech(audio: FloatArray): FloatArray {
        ortVadSession?.let { session ->
            val probabilities = runOrtSileroVad(
                ortEnvironment,
                session,
                audio,
                VAD_WINDOW_SIZE,
                VAD_CONTEXT_SIZE,
                VAD_STATE_SIZE,
            )
            val segments = SpeakerVerificationAudio.speechSegments(probabilities, audio.size)
            require(segments.isNotEmpty()) { "Silero VAD did not detect speech" }
            return SpeakerVerificationAudio.concatenateSegments(audio, segments)
        }
        val model = checkNotNull(vadModel)
        val inputIndex = Ailia.FindBlobIndexByName(model.handle, "input")
        val stateIndex = Ailia.FindBlobIndexByName(model.handle, "state")
        val sampleRateIndex = Ailia.FindBlobIndexByName(model.handle, "sr")
        val probabilityIndex = Ailia.FindBlobIndexByName(model.handle, "output")
        val nextStateIndex = Ailia.FindBlobIndexByName(model.handle, "stateN")
        var state = FloatArray(VAD_STATE_SIZE)
        var context = FloatArray(VAD_CONTEXT_SIZE)
        val probabilities = FloatArray((audio.size + VAD_WINDOW_SIZE - 1) / VAD_WINDOW_SIZE)

        model.setInputBlobShapeND(intArrayOf(1, VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE), inputIndex)
        model.setInputBlobShapeND(intArrayOf(2, 1, 128), stateIndex)
        // Silero v6 requires a scalar int64 sample-rate input. Reuse the typed
        // constant embedded in the graph because Java's setter accepts float arrays.
        val sampleRateConstantIndex = Ailia.FindBlobIndexByName(model.handle, "Constant_0_output")
        model.copyBlob(sampleRateIndex, sampleRateConstantIndex, model)

        probabilities.indices.forEach { windowIndex ->
            val chunk = FloatArray(VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE)
            System.arraycopy(context, 0, chunk, 0, VAD_CONTEXT_SIZE)
            val audioOffset = windowIndex * VAD_WINDOW_SIZE
            val available = minOf(VAD_WINDOW_SIZE, audio.size - audioOffset)
            System.arraycopy(audio, audioOffset, chunk, VAD_CONTEXT_SIZE, available)
            model.setInputBlobData(chunk, chunk.size * Float.SIZE_BYTES, inputIndex)
            model.setInputBlobData(state, state.size * Float.SIZE_BYTES, stateIndex)
            model.update()

            val probability = FloatArray(1)
            model.getBlobData(probability, Float.SIZE_BYTES, probabilityIndex)
            probabilities[windowIndex] = probability[0]
            val nextState = FloatArray(VAD_STATE_SIZE)
            model.getBlobData(nextState, nextState.size * Float.SIZE_BYTES, nextStateIndex)
            state = nextState
            context = chunk.copyOfRange(chunk.size - VAD_CONTEXT_SIZE, chunk.size)
        }

        val segments = SpeakerVerificationAudio.speechSegments(probabilities, audio.size)
        require(segments.isNotEmpty()) { "Silero VAD did not detect speech" }
        return SpeakerVerificationAudio.concatenateSegments(audio, segments)
    }

    @SuppressLint("MissingPermission")
    fun startRecording(listener: RecordingListener): Boolean {
        if (!initialized) {
            listener.onError("VoiceFilter is not initialized")
            return false
        }
        if (!recording.compareAndSet(false, true)) return false
        cancelRecording.set(false)

        val sampleRate = VoiceFilterAudio.SAMPLE_RATE
        val useFloat = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
        val encoding = if (useFloat) AudioFormat.ENCODING_PCM_FLOAT else AudioFormat.ENCODING_PCM_16BIT
        val bytesPerSample = if (useFloat) Float.SIZE_BYTES else Short.SIZE_BYTES
        val minimumBuffer = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, encoding)
        val bufferBytes = maxOf(minimumBuffer, sampleRate * bytesPerSample / 2)

        return try {
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                encoding,
                bufferBytes,
            )
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "Failed to initialize AudioRecord" }
            audioRecord = recorder
            recordingExecutor = Executors.newSingleThreadExecutor { runnable ->
                Thread({
                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO)
                    runnable.run()
                }, "ailia-voicefilter-recording")
            }
            recorder.startRecording()
            recordingExecutor?.execute { recordLoop(recorder, useFloat, listener) }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            recording.set(false)
            releaseRecorder()
            listener.onError(e.message ?: "Failed to start recording")
            false
        }
    }

    private fun recordLoop(recorder: AudioRecord, useFloat: Boolean, listener: RecordingListener) {
        val chunkSize = VoiceFilterAudio.SAMPLE_RATE / 10
        val floatBuffer = FloatArray(chunkSize)
        val shortBuffer = if (useFloat) null else ShortArray(chunkSize)
        val chunks = mutableListOf<FloatArray>()
        var sampleCount = 0
        val maximumSamples = VoiceFilterAudio.SAMPLE_RATE * MAX_RECORDING_SECONDS
        try {
            while (recording.get() && sampleCount < maximumSamples) {
                val count = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && useFloat) {
                    recorder.read(floatBuffer, 0, floatBuffer.size, AudioRecord.READ_BLOCKING)
                } else {
                    @Suppress("DEPRECATION")
                    val read = recorder.read(shortBuffer!!, 0, shortBuffer.size)
                    if (read > 0) for (i in 0 until read) floatBuffer[i] = shortBuffer[i] / 32768f
                    read
                }
                if (count < 0) error("AudioRecord read failed: $count")
                if (count == 0) continue
                val chunk = floatBuffer.copyOf(count)
                chunks += chunk
                sampleCount += count
                listener.onWaveform(chunk, VoiceFilterAudio.SAMPLE_RATE)
            }
        } catch (e: Exception) {
            if (!cancelRecording.get() && recording.get()) {
                listener.onError(e.message ?: "Recording failed")
            }
        } finally {
            recording.set(false)
            releaseRecorder()
            recordingExecutor?.shutdown()
            recordingExecutor = null
            if (!cancelRecording.get() && sampleCount > 0) {
                val audio = FloatArray(sampleCount)
                var offset = 0
                chunks.forEach { chunk ->
                    System.arraycopy(chunk, 0, audio, offset, chunk.size)
                    offset += chunk.size
                }
                listener.onCompleted(audio, VoiceFilterAudio.SAMPLE_RATE)
            }
        }
    }

    fun stopRecording() {
        recording.set(false)
        try {
            audioRecord?.stop()
        } catch (_: Exception) {
        }
    }

    fun cancelRecording() {
        cancelRecording.set(true)
        stopRecording()
    }

    fun playAudio(
        audio: FloatArray,
        sampleRate: Int = VoiceFilterAudio.SAMPLE_RATE,
        onComplete: (() -> Unit)? = null,
    ) {
        require(audio.isNotEmpty()) { "Audio is empty" }
        stopPlayback()
        try {
            val pcm16 = ShortArray(audio.size) { index ->
                (audio[index].coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt().toShort()
            }
            val minimumBuffer = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            val bufferBytes = maxOf(minimumBuffer, pcm16.size * Short.SIZE_BYTES)
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
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(bufferBytes)
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .build()
            } else {
                @Suppress("DEPRECATION")
                AudioTrack(
                    AudioManager.STREAM_MUSIC,
                    sampleRate,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferBytes,
                    AudioTrack.MODE_STATIC,
                )
            }
            audioTrack = track
            val written = track.write(pcm16, 0, pcm16.size)
            check(written == pcm16.size) { "AudioTrack write failed: $written/${pcm16.size}" }
            track.setNotificationMarkerPosition(pcm16.size)
            track.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(completedTrack: AudioTrack?) {
                    if (audioTrack === completedTrack) {
                        stopPlayback()
                        onComplete?.invoke()
                    }
                }

                override fun onPeriodicNotification(completedTrack: AudioTrack?) = Unit
            })
            track.play()
        } catch (e: Exception) {
            Log.e(TAG, "Audio playback failed", e)
            stopPlayback()
            throw e
        }
    }

    @Synchronized
    fun stopPlayback() {
        val track = audioTrack ?: return
        audioTrack = null
        try {
            if (track.playState == AudioTrack.PLAYSTATE_PLAYING) track.stop()
        } catch (_: Exception) {
        }
        try {
            track.release()
        } catch (_: Exception) {
        }
    }

    fun release() {
        cancelRecording()
        stopPlayback()
        releaseModels()
    }

    private fun releaseModels() {
        try {
            filterModel?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release VoiceFilter", e)
        }
        try {
            embedderModel?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release VoiceFilter embedder", e)
        }
        try {
            vadModel?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release Silero VAD", e)
        }
        try {
            ortFilterSession?.close()
            ortEmbedderSession?.close()
            ortVadSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release ONNX Runtime audio sessions", e)
        }
        filterModel = null
        embedderModel = null
        vadModel = null
        ortFilterSession = null
        ortEmbedderSession = null
        ortVadSession = null
        initialized = false
    }

    @Synchronized
    private fun releaseRecorder() {
        try {
            audioRecord?.release()
        } catch (_: Exception) {
        }
        audioRecord = null
    }
}
