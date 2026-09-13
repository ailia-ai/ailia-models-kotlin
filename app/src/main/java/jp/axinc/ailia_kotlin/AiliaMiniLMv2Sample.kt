package jp.axinc.ailia_kotlin

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import axip.ailia_tokenizer.AiliaTokenizer
import java.io.File
import java.util.Locale

/**
 * MiniLMv2 zero-shot classification sample using ailia Tokenizer + ailia SDK.
 *
 * Pipeline:
 * 1. Tokenize sentence-hypothesis pairs with XLM-RoBERTa tokenizer
 * 2. Run ONNX inference with ailia SDK
 * 3. Apply softmax to entailment logits for classification scores
 */
class AiliaMiniLMv2Sample(private val modelDirectory: File) {

    companion object {
        private const val TAG = "AILIA_Main"
        init { System.loadLibrary("ailia") }
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/multilingual-minilmv2/minilm_l12.onnx"
        private const val MODEL_FILE = "minilm_l12.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/multilingual-minilmv2/minilm_l12.onnx.prototxt"
        private const val PROTO_FILE = "minilm_l12.onnx.prototxt"
        private const val TOKENIZER_URL = "https://storage.googleapis.com/ailia-models/multilingual-minilmv2/sentencepiece.bpe.model"
        private const val TOKENIZER_FILE = "sentencepiece.bpe.model"

        // XLM-RoBERTa special token IDs
        private const val CLS_TOKEN = 0   // <s>
        private const val PAD_TOKEN = 1   // <pad>
        private const val SEP_TOKEN = 2   // </s>

        private const val MAX_LENGTH = 128
    }

    private var tokenizer: AiliaTokenizer? = null
    private var model: AiliaModel? = null
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var isInitialized = false
    private var lastResult: String = ""

    /**
     * Downloads all required model files (ONNX model, prototxt, sentencepiece tokenizer).
     */
    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting MiniLMv2 model download/check...")
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(PROTO_URL, PROTO_FILE), listener) != null)
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(TOKENIZER_URL, TOKENIZER_FILE), listener) != null)
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(MODEL_URL, MODEL_FILE), listener) != null)
            listener?.onComplete()
            Log.i(TAG, "MiniLMv2 model download/check complete")
            true
        } catch (e: Exception) {
            Log.e(TAG, "MiniLMv2 Download Failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    /**
     * Initializes the XLM-RoBERTa tokenizer and ailia ONNX model.
     */
    fun initialize(envId: Int): Boolean {
        if (isInitialized) {
            release()
        }

        return try {
            val tokenizerPath = File(modelDirectory, TOKENIZER_FILE).absolutePath
            val protoPath = File(modelDirectory, PROTO_FILE).absolutePath
            val modelPath = File(modelDirectory, MODEL_FILE).absolutePath

            // Initialize tokenizer
            tokenizer = AiliaTokenizer(AiliaTokenizer.AILIA_TOKENIZER_TYPE_XLM_ROBERTA)
            tokenizer?.loadFiles(modelPath = tokenizerPath)
            Log.i(TAG, "MiniLMv2 tokenizer initialized")

            // Initialize ONNX model
            model = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)
            Log.i(TAG, "MiniLMv2 ONNX model initialized with envId=$envId")

            isInitialized = true
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MiniLMv2: ${e.javaClass.name}: ${e.message}")
            release()
            false
        }
    }

    fun initializeOnnxRuntime(): Boolean {
        release()
        return try {
            tokenizer = AiliaTokenizer(AiliaTokenizer.AILIA_TOKENIZER_TYPE_XLM_ROBERTA)
            tokenizer?.loadFiles(modelPath = File(modelDirectory, TOKENIZER_FILE).absolutePath)
            val options = OrtSession.SessionOptions()
            try {
                ortSession = ortEnvironment.createSession(
                    File(modelDirectory, MODEL_FILE).absolutePath,
                    options,
                )
            } finally {
                options.close()
            }
            isInitialized = true
            Log.i(TAG, "MiniLMv2 initialized with ONNX Runtime CPU")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MiniLMv2 with ONNX Runtime", e)
            release()
            false
        }
    }

    /**
     * Runs zero-shot classification on the given sentence with candidate labels.
     *
     * @param sentence Input sentence to classify
     * @param labels Candidate labels
     * @param hypothesisTemplate Template string with {} placeholder for label
     * @return Processing time in milliseconds
     */
    fun predict(
        sentence: String,
        labels: List<String>,
        hypothesisTemplate: String = "This example is {}."
    ): Long {
        if (!isInitialized || tokenizer == null || (model == null && ortSession == null)) {
            Log.e(TAG, "MiniLMv2 not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()

            // 1. Tokenize each (sentence, hypothesis) pair
            val tokenizedPairs = labels.map { label ->
                val hypothesis = hypothesisTemplate.replace("{}", label)
                tokenizePair(sentence, hypothesis)
            }

            val batchSize = tokenizedPairs.size
            val maxSeqLen = tokenizedPairs.maxOf { it.first.size }.coerceAtMost(MAX_LENGTH)
            Log.i(TAG, "MiniLMv2 predict: batchSize=$batchSize, maxSeqLen=$maxSeqLen")

            // 2. Create padded batch arrays. The ONNX graph uses int64 inputs.
            val inputIds = FloatArray(batchSize * maxSeqLen)
            val attentionMask = FloatArray(batchSize * maxSeqLen)

            for (i in tokenizedPairs.indices) {
                val (ids, mask) = tokenizedPairs[i]
                val seqLen = ids.size.coerceAtMost(maxSeqLen)
                for (j in 0 until seqLen) {
                    inputIds[i * maxSeqLen + j] = ids[j].toFloat()
                    attentionMask[i * maxSeqLen + j] = mask[j].toFloat()
                }
                // Padding (PAD_TOKEN=1, mask=0) - already 0 from array init
                for (j in seqLen until maxSeqLen) {
                    inputIds[i * maxSeqLen + j] = PAD_TOKEN.toFloat()
                    // attentionMask already 0
                }
            }

            // 3. Run inference and get output logits (batchSize, 3).
            val output = ortSession?.let { session ->
                val shape = longArrayOf(batchSize.toLong(), maxSeqLen.toLong())
                val idsTensor = createLongTensor(
                    ortEnvironment,
                    LongArray(inputIds.size) { inputIds[it].toLong() },
                    shape,
                )
                val maskTensor = createLongTensor(
                    ortEnvironment,
                    LongArray(attentionMask.size) { attentionMask[it].toLong() },
                    shape,
                )
                try {
                    runFloatTensor(
                        session,
                        mapOf("input_ids" to idsTensor, "attention_mask" to maskTensor),
                        "logits",
                    )
                } finally {
                    closeTensors(idsTensor, maskTensor)
                }
            } ?: run {
                val ailiaModel = checkNotNull(model)
                val inputIdx0 = ailiaModel.getBlobIndexByInputIndex(0)
                val inputIdx1 = ailiaModel.getBlobIndexByInputIndex(1)
                ailiaModel.setInputBlobShapeND(intArrayOf(batchSize, maxSeqLen), inputIdx0)
                ailiaModel.setInputBlobShapeND(intArrayOf(batchSize, maxSeqLen), inputIdx1)
                ailiaModel.setInputBlobData(inputIds, inputIds.size * 4, inputIdx0)
                ailiaModel.setInputBlobData(attentionMask, attentionMask.size * 4, inputIdx1)
                ailiaModel.update()
                FloatArray(batchSize * 3).also {
                    ailiaModel.getBlobData(
                        it,
                        it.size * 4,
                        ailiaModel.getBlobIndexByOutputIndex(0),
                    )
                }
            }

            // 6. Extract entailment logits and apply softmax
            val entailmentLogits = FloatArray(batchSize)
            for (i in 0 until batchSize) {
                entailmentLogits[i] = output[i * 3 + 0]  // entailment is index 0
            }
            val scores = softmax(entailmentLogits)

            // 7. Format results (sorted by score descending)
            val results = labels.zip(scores.toList())
                .sortedByDescending { it.second }
                .joinToString("\n") { (label, score) ->
                    "$label: ${String.format(Locale.ROOT, "%.1f", score * 100)}%"
                }

            lastResult = results
            Log.i(TAG, "MiniLMv2 results:\n$results")

            val endTime = System.nanoTime()
            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "MiniLMv2 predict failed: ${e.javaClass.name}: ${e.message}")
            lastResult = "Error: ${e.message}"
            -1
        }
    }

    /**
     * Tokenizes a sentence-hypothesis pair in XLM-RoBERTa format:
     * [CLS] sentence_tokens [SEP] [SEP] hypothesis_tokens [SEP]
     *
     * @return Pair of (input_ids, attention_mask)
     */
    private fun tokenizePair(sentence: String, hypothesis: String): Pair<IntArray, IntArray> {
        val sentTokens = tokenizer!!.encode(sentence)
        val hypoTokens = tokenizer!!.encode(hypothesis)

        // Build: [CLS] sent [SEP] [SEP] hypo [SEP]
        val totalLen = 1 + sentTokens.size + 2 + hypoTokens.size + 1
        val inputIds = IntArray(totalLen)
        val attentionMask = IntArray(totalLen) { 1 }

        var pos = 0
        inputIds[pos++] = CLS_TOKEN
        for (t in sentTokens) inputIds[pos++] = t
        inputIds[pos++] = SEP_TOKEN
        inputIds[pos++] = SEP_TOKEN
        for (t in hypoTokens) inputIds[pos++] = t
        inputIds[pos] = SEP_TOKEN

        return Pair(inputIds, attentionMask)
    }

    /**
     * Computes softmax over a float array.
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { Math.exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sumExp = exps.sum()
        return FloatArray(logits.size) { exps[it] / sumExp }
    }

    fun release() {
        try {
            tokenizer?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing tokenizer: ${e.message}")
        }
        try {
            model?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing model: ${e.message}")
        }
        try {
            ortSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ONNX Runtime MiniLMv2: ${e.message}")
        }
        tokenizer = null
        model = null
        ortSession = null
        isInitialized = false
        Log.i(TAG, "MiniLMv2 released")
    }

    fun getLastResult(): String = lastResult
}
