package jp.axinc.ailia_kotlin

import android.content.Context

/**
 * Prefill(PPS)計測用のプロンプトを組み立てる。
 *
 * 先頭に「下記を要約してください。」を置き、本文はailia.aiのアイリア紹介文([R.raw.ailia_intro])。
 * 指定したトークン数になるまで段落を繰り返し、最後に文字数の二分探索でトークン数を合わせる。
 * トークン数はモデルのトークナイザ([AiliaLLM.getTokenCount])で数えるため、
 * モデルが未初期化の場合は文字数からの概算になる。
 */
object BenchmarkPrompt {
    /** 既定の評価トークン数。 */
    const val DEFAULT_TARGET_TOKENS = 2048

    /** 評価テキストの先頭に付ける指示。指示を含めて目標トークン数に合わせる。 */
    private const val INSTRUCTION = "下記を要約してください。\n\n"

    /**
     * モデル未初期化時に使う概算。日本語主体の本文に対するトークンあたりの文字数。
     * Gemma 4のトークナイザで [R.raw.ailia_intro] を数えた実測値(約2.09文字/トークン)に基づく。
     */
    private const val CHARS_PER_TOKEN_ESTIMATE = 2.05

    /** 二分探索の許容誤差(トークン)。 */
    private const val TOLERANCE_TOKENS = 2

    /** 本文を繰り返す際の上限。トークナイザが想定外の値を返しても暴走しないようにする。 */
    private const val MAX_CHARS = 1_000_000

    /**
     * 指定トークン数の評価テキストを作る。
     *
     * @param countTokens トークン数を数える関数。nullなら文字数から概算する。
     * @return 評価テキストと、そのトークン数(概算の場合は推定値)
     */
    fun build(
        context: Context,
        targetTokens: Int = DEFAULT_TARGET_TOKENS,
        countTokens: ((String) -> Int)? = null,
    ): Result {
        val source = loadSource(context)
        if (countTokens == null) {
            val chars = (targetTokens * CHARS_PER_TOKEN_ESTIMATE).toInt() - INSTRUCTION.length
            val body = repeatToLength(source, chars).take(chars.coerceAtLeast(1))
            return Result(INSTRUCTION + body, targetTokens, exact = false)
        }

        // 目標トークン数を超えるまで本文を繰り返してから、文字数の二分探索で切り詰める
        var body = source
        var tokens = countTokens(INSTRUCTION + body)
        while (tokens < targetTokens && body.length < MAX_CHARS) {
            body = repeatToLength(source, (body.length * 2).coerceAtMost(MAX_CHARS))
            tokens = countTokens(INSTRUCTION + body)
        }
        if (tokens <= targetTokens) {
            return Result(INSTRUCTION + body, tokens, exact = true)
        }

        var low = 1
        var high = body.length
        var best = INSTRUCTION + body.take(low)
        var bestTokens = countTokens(best)
        while (low <= high) {
            val middle = (low + high) / 2
            val candidate = INSTRUCTION + body.take(middle)
            val candidateTokens = countTokens(candidate)
            if (candidateTokens <= targetTokens) {
                best = candidate
                bestTokens = candidateTokens
                if (targetTokens - candidateTokens <= TOLERANCE_TOKENS) break
                low = middle + 1
            } else {
                high = middle - 1
            }
        }
        return Result(best, bestTokens, exact = true)
    }

    private fun loadSource(context: Context): String =
        context.resources.openRawResource(R.raw.ailia_intro)
            .bufferedReader()
            .use { it.readText() }
            .trim()

    /** 本文を段落区切りで繰り返し、指定文字数以上の文章にする。 */
    private fun repeatToLength(source: String, length: Int): String {
        val builder = StringBuilder(source)
        while (builder.length < length) {
            builder.append("\n\n").append(source)
        }
        return builder.toString()
    }

    /**
     * @property text 評価用テキスト
     * @property tokens [text]のトークン数(推定の場合は目標値)
     * @property exact モデルのトークナイザで数えた正確な値かどうか
     */
    data class Result(val text: String, val tokens: Int, val exact: Boolean)
}
