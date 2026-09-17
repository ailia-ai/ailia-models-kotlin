package jp.axinc.ailia_kotlin

import android.util.Log
import axip.ailia_llm.AiliaLLM

/**
 * 端末のSoCに対するQNN(Qualcomm AI Engine Direct)の対応状況をまとめたヘルパ。
 *
 * SoC名はailia LLMの ailiaLLMGetQNNModelName API から取得する(例: "sm8475")。
 * ailia SDKのQNN環境の可否と、ailia LLMのQNNモデルの選択可否の両方でこの名前を使う。
 */
object QnnSupport {
    private const val TAG = "QnnSupport"

    /**
     * NPU(HTP)がFP16に対応しないSoC。
     *
     * ailia SDKのQNNバックエンドはFP16を前提とするため、これらのSoCでは選択できない。
     * FP16対応はHexagonのバージョンでは決まらない(例: v73のsm8550は対応、sm7635やsm8635は非対応)。
     *
     * 一覧はQAIRT 2.47のSDKが持つSoC情報から生成している。
     * libPyBackendInfo の PyBackendInfo("HTP", soc).get_soc_info_subset().supportsFp16
     * (qti.aisw.tools.core.utilities.devices が SocDetails.supports_fp16 として公開している値)で、
     * QAIRTを更新したときは同じ方法で再生成する。
     */
    private val FP16_UNSUPPORTED_SOCS = setOf(
        // DSPアーキテクチャ未定義(HTP非搭載)
        "cq4390m", "cq4390s", "qcm2290", "qcm4490", "qcs2290", "qcs4490", "sa525m", "sg4250",
        "sg4250p", "sm4450", "sm4635", "sm4850", "sm4850p", "sm6435", "sm6450q", "sm6475q",
        "sm6850", "sm6850q",
        // Hexagon v65
        "sm7150",
        // Hexagon v66
        "cq2390m", "cq2390s", "qcm6125", "qcs403", "qcs405", "qcs410", "qcs610", "qcs6125",
        "qcs615", "qcs7230", "qrb4210", "qrb5165", "sa8195", "sm4250", "sm4350", "sm4375",
        "sm6115", "sm6115p", "sm6125", "sm6150", "sm6250", "sm6350", "sm6370", "sm6375", "sm7225",
        "sm7250", "sm8150", "sm8250",
        // Hexagon v68
        "qcm5430", "qcm6490", "qcs5430", "qcs6490", "sc7280x", "sc8280x", "sm7315", "sm7325",
        "sm7350", "sm8325", "sm8350", "sm8350p",
        // Hexagon v69
        "sm7475",
        // Hexagon v73
        "cq7790m", "cq7790s", "qcm6690", "qcs6690", "qmb715", "qna715", "sg6150", "sg6150p",
        "sm6450", "sm6475", "sm6475p", "sm6650", "sm6650p", "sm7435", "sm7435p", "sm7525",
        "sm7550", "sm7550p", "sm7635", "sm7635p", "sm7675", "sm7675p", "sm7750", "sm7750p",
        "sm7775", "sm8635", "sm8635p", "sm8735", "sm8735p", "ssg2115p", "ssg2125p", "sxr1230p",
    )

    /** 変換済みQNNモデルを公開しているSoC。 */
    private val LLM_SUPPORTED_SOCS = setOf("sm8475", "sm7635")

    /**
     * QNNモデルを公開しているLLMモデルと、そのファイル名の接頭辞。
     * ファイル名は "<prefix>-<soc>.qnn" (画像/音声エンコーダは "<prefix>-<soc>-mmproj.qnn")。
     * Gemma 4 E4Bは変換済みモデルが未公開のため、公開後にここへ追加する。
     */
    private val LLM_FILE_PREFIXES = mapOf(
        LLMModelType.GEMMA_4_E2B to "gemma4-e2b",
    )

    /** ailia LLMのAPIで取得したSoC名(例: "sm8475")。QNNを利用できない端末ではnull。 */
    val socName: String? by lazy {
        runCatching { AiliaLLM.getQNNModelName() }
            .onFailure { Log.i(TAG, "QNN model name is not available: ${it.message}") }
            .getOrNull()
            ?.trim()
            ?.lowercase()
            ?.takeIf { it.isNotEmpty() }
            ?.also { Log.i(TAG, "QNN SoC name: $it") }
    }

    /**
     * SoCのNPUがFP16に対応するか。
     * SoC名を取得できない場合(QNN非対応端末など)は従来通り選択可能として扱う。
     */
    fun isFp16Supported(): Boolean = isFp16SupportedSoc(socName)

    /** 指定したLLMモデルにこの端末向けのQNNモデルがあるか。 */
    fun isLLMQnnAvailable(modelType: LLMModelType): Boolean = llmQnnFileName(modelType) != null

    /** 指定したLLMモデルのQNNモデルファイル名。この端末向けのモデルがなければnull。 */
    fun llmQnnFileName(modelType: LLMModelType): String? = llmQnnFileName(socName, modelType)

    /** VLM/ALMで使用する画像・音声エンコーダ(mmproj)のQNNモデルファイル名。 */
    fun llmQnnMmprojFileName(modelType: LLMModelType): String? =
        llmQnnFileName(socName, modelType, mmproj = true)

    /** SoC名を引数で受け取る判定本体(端末を用意せずテストできるようにするため)。 */
    internal fun isFp16SupportedSoc(soc: String?): Boolean =
        soc?.let { it !in FP16_UNSUPPORTED_SOCS } ?: true

    /** SoC名を引数で受け取るファイル名生成本体。 */
    internal fun llmQnnFileName(
        soc: String?,
        modelType: LLMModelType,
        mmproj: Boolean = false,
    ): String? {
        if (soc == null || soc !in LLM_SUPPORTED_SOCS) return null
        val prefix = LLM_FILE_PREFIXES[modelType] ?: return null
        return if (mmproj) "$prefix-$soc-mmproj.qnn" else "$prefix-$soc.qnn"
    }

    /** ailia SDKの環境名がQNNバックエンドかどうかを判定する。 */
    fun isQnnEnvironmentName(name: String): Boolean = name.contains("QNN", ignoreCase = true)
}
