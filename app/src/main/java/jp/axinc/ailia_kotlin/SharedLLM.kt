package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Log
import axip.ailia_llm.AiliaLLM
import java.io.File

/**
 * LLM / VLM / ALM / Tool Useで共有するAiliaLLM。
 *
 * 画面を切り替えるたびにモデルを開き直すとQNNでは数秒かかるため、
 * 同じモデル・MTP・mmprojの組み合わせなら1つのインスタンスを使い回す。
 * mmprojは後から追加できない(最初のプロンプトより前に開く必要がある)ため、
 * 対応するモデルでは常にmmprojも開き、どの画面からでも使えるようにする。
 *
 * QNNのモデルは初期化したスレッドで解放する必要があるため、
 * [acquire]と[release]は推論スレッド(MainActivityのcameraExecutor)から呼び出す。
 */
object SharedLLM {
    private const val TAG = "SharedLLM"

    /** 開いているモデルの組み合わせ。同じなら使い回す。 */
    data class Config(
        val modelPath: String,
        val nCtx: Int,
        val mtpPath: String?,
        val projectorPath: String?,
    )

    private var llm: AiliaLLM? = null
    private var config: Config? = null

    /** 開いているモデルでMTPを有効にできたか。 */
    @Volatile
    var isMtpActive: Boolean = false
        private set

    /** mmproj(画像・音声エンコーダ)のファイル名。mmprojのないモデル(Gemma 2など)ではnull。 */
    fun projectorFileName(modelType: LLMModelType, backend: LLMBackend): String? = when (backend) {
        LLMBackend.CPU -> modelType.mmprojFileName
        LLMBackend.QNN -> QnnSupport.llmQnnMmprojFileName(modelType)
    }

    /** モデル本体・MTP・mmprojのいずれかをバックエンドに応じた配置先からダウンロードする。 */
    fun download(
        context: Context,
        fileName: String,
        backend: LLMBackend,
        listener: ModelDownloader.DownloadListener?,
    ): File? = LLMMtp.download(context, fileName, backend, listener)

    fun isDownloaded(context: Context, fileName: String, backend: LLMBackend): Boolean =
        LLMMtp.isDownloaded(context, fileName, backend)

    /**
     * [config]のモデルを返す。同じ組み合わせが開いていればそれを使い回し、
     * 異なる場合は閉じてから開き直す。失敗した場合は例外を投げる。
     */
    @Synchronized
    fun acquire(config: Config): AiliaLLM {
        llm?.takeIf { this.config == config }?.let {
            Log.i(TAG, "Reusing LLM: ${config.modelPath}")
            return it
        }
        release()
        val model = AiliaLLM()
        try {
            Log.i(TAG, "Opening model file: ${config.modelPath}")
            model.openModelFile(config.modelPath, config.nCtx)
            // Assistantは本体の後、mmprojと最初のプロンプトより前に開く
            val mtpActive = config.mtpPath != null && LLMMtp.open(model, config.mtpPath)
            config.projectorPath?.let {
                Log.i(TAG, "Opening multimodal projector: $it")
                model.openMultimodalProjectorFile(it)
            }
            llm = model
            this.config = config
            isMtpActive = mtpActive
            Log.i(TAG, "LLM opened. Context size: ${model.getContextSize()}, MTP: $mtpActive")
            return model
        } catch (e: Exception) {
            model.destroy()
            throw e
        }
    }

    /** 共有しているモデルを閉じる。LLM系以外の画面に移ったときやモデルの設定を変えたときに呼ぶ。 */
    @Synchronized
    fun release() {
        val model = llm ?: return
        try {
            model.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing LLM: ${e.message}")
        } finally {
            llm = null
            config = null
            isMtpActive = false
            Log.i(TAG, "LLM released")
        }
    }
}
