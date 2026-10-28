package jp.axinc.ailia_kotlin

import org.json.JSONObject
import org.json.JSONArray
import java.io.File
import java.util.concurrent.CancellationException

/** Tool definitions and sample device state, owned by the model worker. */
internal class VoiceAgentTools {
    private var temperature: Double? = null
    fun reset() { temperature = null }

    fun definitions(): String = JSONArray(DEFINITIONS).apply {
        temperature?.let {
            getJSONObject(0).getJSONObject("function").put("description", "エアコンの設定温度を変更します。現在は${it}度です。")
        }
    }.toString()

    fun execute(name: String, arguments: String, onTemperature: (Double) -> Unit,
                takePhoto: () -> File): String = try {
        val args = JSONObject(arguments)
        when (name) {
            AiliaToolUseSample.TOOL_NAME -> {
                val value = args.getDouble("temperature")
                require(value.isFinite()) { "設定温度は有限の数値で指定してください" }
                temperature = value
                onTemperature(value)
                JSONObject().put("status", "ok").put("temperature", value).toString()
            }
            TAKE_PHOTO -> {
                takePhoto()
                JSONObject().put("status", "ok").put("message", "写真を撮影しました。").toString()
            }
            else -> JSONObject().put("error", "unknown tool: $name").toString()
        }
    } catch (cancelled: CancellationException) {
        throw cancelled
    } catch (error: Exception) {
        JSONObject().put("error", error.message ?: "ツールを実行できませんでした").toString()
    }

    companion object {
        const val TAKE_PHOTO = "take_photo"
        val DEFINITIONS = """
            [{"type":"function","function":{
                "name":"${AiliaToolUseSample.TOOL_NAME}",
                "description":"Set the air conditioner temperature.",
                "parameters":{"type":"object","properties":{
                    "temperature":{"type":"number","description":"Target temperature in degrees Celsius."}},
                    "required":["temperature"]}
            }},{"type":"function","function":{
                "name":"$TAKE_PHOTO",
                "description":"Take one photo using the device camera.",
                "parameters":{"type":"object","properties":{}}
            }}]
        """.trimIndent()
    }
}
