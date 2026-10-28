# Running the LLM samples on the NPU (QNN)

The LLM / VLM / ALM / Tool Use samples can run Gemma 4 on the Qualcomm NPU through the ailia LLM QNN backend.
Gemma 4 E2B has QNN models for `sm8475` and `sm7635`. Gemma 4 E4B runs on QNN only on `sm7635`, because it
does not fit the 2 GB context binary limit of Hexagon v69 (`sm8475`); on other SoCs E4B selects CPU automatically.
The `CPU` / `QNN` selector next to the Send button defaults to `QNN (NPU) <soc>`, and that entry appears only when
[`AiliaLLM.getQNNModelName()`](ailia-llm-jni/src/main/kotlin/axip/ailia_llm/Ailiallm.kt) reports a SoC that has a
converted model (currently `sm8475` and `sm7635`, see
[QnnSupport.kt](app/src/main/java/jp/axinc/ailia_kotlin/QnnSupport.kt)); otherwise only `CPU` is offered.

QNN models are SoC specific and self contained: `gemma4-e2b-<soc>.qnn` holds the prefill/decode graphs and
`gemma4-e2b-<soc>-mmproj.qnn` holds the image / audio encoder used by the VLM and ALM samples. The context
length is fixed at conversion time, so the samples open them with `n_ctx = 0`.

The LLM sample also has a `2048 tok` button next to Send. It pastes an ailia introduction text taken from
[ailia.ai](https://ailia.ai/) ([ailia_intro.txt](app/src/main/res/raw/ailia_intro.txt)), trimmed with the model
tokenizer to 2048 tokens, and clears the chat history so the next Send measures the prefill of that text alone.
After generation the status line reports the measured prefill throughput (PPS) and decode throughput, for example
`Prefill 2065 tokens 259.45 tokens/s (7959 ms) / Decode 894 tokens 7.48 tokens/s`.

The ailia SDK QNN backend requires FP16 on the NPU. On SoCs without FP16 support the environment list shows
`This SoC QNN FP16 not supported` and the QNN entries cannot be selected. FP16 support does not follow the
Hexagon version (SM8550 supports it while SM7635 and SM8635 do not, although all three are v73), so
[QnnSupport.kt](app/src/main/java/jp/axinc/ailia_kotlin/QnnSupport.kt) carries the per-SoC list taken from the
QAIRT 2.47 SDK (`libPyBackendInfo`: `PyBackendInfo("HTP", soc).get_soc_info_subset().supportsFp16`).
