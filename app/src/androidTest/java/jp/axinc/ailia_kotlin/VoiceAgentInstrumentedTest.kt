package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.appcompat.widget.SwitchCompat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sin

@RunWith(AndroidJUnit4::class)
class VoiceAgentInstrumentedTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    @Test fun setupAndSessionAreSeparateAndStopRestoresSettings() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val menu = activity.findViewById<Spinner>(R.id.algorithmSpinner)
                assertEquals("ToolUse", menu.getItemAtPosition(MainActivity.AlgorithmType.VOICE_AGENT.ordinal - 1))
                menu.setSelection(MainActivity.AlgorithmType.VOICE_AGENT.ordinal)
            }
            instrumentation.waitForIdleSync()
            screenshot("voice-agent-setup.png")
            // Exercise the real UI transition while keeping heavyweight model work off the UI thread.
            instrumentation.uiAutomation.executeShellCommand("pm grant ${instrumentation.targetContext.packageName} android.permission.RECORD_AUDIO").use {
                FileInputStream(it.fileDescriptor).use { stream -> stream.readBytes() }
            }
            scenario.onActivity { activity ->
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
                assertFalse(activity.findViewById<View>(R.id.modelSpinner).isShown)
                val mode = activity.findViewById<RadioGroup>(R.id.voiceAgentInputMode)
                val queries = activity.findViewById<EditText>(R.id.voiceAgentTestQueries)
                val speechOutput = activity.findViewById<SwitchCompat>(R.id.voiceAgentSpeechOutput)
                assertFalse("Speech output should be disabled by default", speechOutput.isChecked)
                speechOutput.isChecked = false
                assertFalse(activity.findViewById<Spinner>(R.id.voiceAgentVoiceBackend).isEnabled)
                assertEquals(R.id.voiceAgentMicInput, mode.checkedRadioButtonId)
                assertEquals(View.GONE, activity.findViewById<View>(R.id.voiceAgentQueryPanel).visibility)
                mode.check(R.id.voiceAgentTestInput)
                assertTrue("Test query synthesis still needs the voice backend",
                    activity.findViewById<Spinner>(R.id.voiceAgentVoiceBackend).isEnabled)
                assertEquals(View.VISIBLE, activity.findViewById<View>(R.id.voiceAgentQueryPanel).visibility)
                queries.setText(" \n ")
                activity.findViewById<Button>(R.id.voiceAgentStart).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentError).isShown)
                assertFalse(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                queries.setText(R.string.voice_agent_default_queries)
                // Starting and stopping in the same UI callback also exercises cancellation of test synthesis.
                activity.findViewById<Button>(R.id.voiceAgentStart).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                assertEquals(View.GONE, activity.findViewById<View>(R.id.voiceAgentSend).visibility)
                assertFalse(activity.findViewById<View>(R.id.rootScrollView).isShown)
                assertFalse(activity.findViewById<View>(R.id.llmChatContainer).isShown)
                assertFalse(activity.findViewById<View>(R.id.processingTimeTextView).isShown)
                activity.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
            }
            waitForStartEnabled(scenario)
            scenario.onActivity { activity ->
                activity.findViewById<RadioGroup>(R.id.voiceAgentInputMode).check(R.id.voiceAgentMicInput)
                val speechOutput = activity.findViewById<SwitchCompat>(R.id.voiceAgentSpeechOutput)
                assertFalse("Preserve the selected setting when returning from a session", speechOutput.isChecked)
                assertFalse(activity.findViewById<Spinner>(R.id.voiceAgentVoiceBackend).isEnabled)
                speechOutput.isChecked = true
                assertTrue(activity.findViewById<Spinner>(R.id.voiceAgentVoiceBackend).isEnabled)
                assertEquals(View.GONE, activity.findViewById<View>(R.id.voiceAgentQueryPanel).visibility)
                activity.findViewById<Button>(R.id.voiceAgentStart).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSend).isShown)
                assertFalse(activity.findViewById<Button>(R.id.voiceAgentSend).isEnabled)
                activity.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                assertFalse(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
            }
        }
    }

    /** Capture a short real microphone buffer and submit it before the silence endpoint can fire. */
    @Test(timeout = 60_000) fun microphoneSendButtonSubmitsWithoutWaitingForVad() {
        assumeTrue("Enable explicitly on a device with cached models", InstrumentationRegistry.getArguments().getString("voiceAgentNative") == "true")
        val context = instrumentation.targetContext
        System.loadLibrary("ailia")
        val speechFile = QnnSupport.speechQnnFileName(SpeechModelType.WHISPER_MEDIUM)
        val llmFile = QnnSupport.llmQnnFileName(LLMModelType.GEMMA_4_E2B)
        assumeTrue(speechFile != null && llmFile != null)
        val directory = ModelDownloader.modelDirectory(context)
        assumeTrue(File(directory, speechFile!!).exists() && File(directory, llmFile!!).exists())
        instrumentation.uiAutomation.executeShellCommand("pm grant ${context.packageName} android.permission.RECORD_AUDIO").use {
            FileInputStream(it.fileDescriptor).use { stream -> stream.readBytes() }
        }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { it.findViewById<Spinner>(R.id.algorithmSpinner).setSelection(MainActivity.AlgorithmType.VOICE_AGENT.ordinal) }
            instrumentation.waitForIdleSync()
            scenario.onActivity { it.findViewById<Button>(R.id.voiceAgentStart).performClick() }
            val deadline = SystemClock.elapsedRealtime() + 15_000
            var ready = false
            while (!ready && SystemClock.elapsedRealtime() < deadline) {
                scenario.onActivity { ready = it.findViewById<Button>(R.id.voiceAgentSend).isEnabled }
                if (!ready) SystemClock.sleep(20)
            }
            assertTrue("Microphone did not become ready", ready)
            screenshot("voice-agent-microphone-send.png")
            scenario.onActivity { activity ->
                val send = activity.findViewById<Button>(R.id.voiceAgentSend)
                assertTrue(send.isShown)
                send.performClick()
                assertTrue("Switch immediately from send to cancel", send.isEnabled)
                assertEquals(activity.getString(R.string.voice_agent_interrupt), send.contentDescription)
                assertEquals("音声を送信しています", activity.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
            }
            var recognizing = false
            val recognitionDeadline = SystemClock.elapsedRealtime() + 30_000
            while (!recognizing && SystemClock.elapsedRealtime() < recognitionDeadline) {
                scenario.onActivity {
                    recognizing = it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription == "音声を認識しています"
                }
                if (!recognizing) SystemClock.sleep(20)
            }
            assertTrue("Manual send did not forward the recorded audio to recognition", recognizing)
            scenario.onActivity { activity ->
                assertTrue(activity.findViewById<Button>(R.id.voiceAgentSend).isEnabled)
                activity.findViewById<Button>(R.id.voiceAgentSend).performClick()
            }
            waitForOrbState(scenario, "音声入力を待っています", 5000)
            waitForSendEnabled(scenario)
            scenario.onActivity { activity ->
                assertEquals(activity.getString(R.string.voice_agent_send), activity.findViewById<Button>(R.id.voiceAgentSend).contentDescription)
                activity.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
            }
            waitForStartEnabled(scenario, 15_000)
        }
    }

    @Test fun slimeMovesAtRestAndReactsToMicrophoneAndPlaybackAudio() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            fun frame(state: AiliaVoiceAgentSample.State?, audio: FloatArray?): Bitmap {
                val bitmap = Bitmap.createBitmap(400, 400, Bitmap.Config.ARGB_8888)
                scenario.onActivity {
                    val orb = it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb)
                    orb.layout(0, 0, 400, 400)
                    if (state != null) orb.setState(state)
                    if (audio != null) orb.setAudio(audio)
                    orb.draw(Canvas(bitmap))
                }
                return bitmap
            }
            val idle = frame(AiliaVoiceAgentSample.State.LISTENING, null)
            SystemClock.sleep(300)
            val movingIdle = frame(null, null)
            assertFalse("The character should keep moving without audio", idle.sameAs(movingIdle))
            assertEquals(Color.BLACK, idle.getPixel(0, 0))
            val audio = FloatArray(1600) { .45f * sin(it * .13f) }
            var microphone = idle
            repeat(12) {
                if (microphone !== idle) microphone.recycle()
                microphone = frame(null, audio)
                SystemClock.sleep(20)
            }
            val quietArea = whiteArea(idle)
            assertTrue("Microphone amplitude should expand the body", whiteArea(microphone) > quietArea * 1.2)
            var speaking = frame(AiliaVoiceAgentSample.State.SPEAKING, audio)
            repeat(12) {
                speaking.recycle()
                speaking = frame(null, audio)
                SystemClock.sleep(20)
            }
            assertTrue("Playback amplitude should expand the same body", whiteArea(speaking) > quietArea * 1.2)
            val directory = instrumentation.targetContext.getExternalFilesDir(null)
            for ((name, bitmap) in listOf("slime-idle.png" to idle, "slime-speaking.png" to speaking)) {
                File(directory, name).outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
            idle.recycle(); movingIdle.recycle(); microphone.recycle(); speaking.recycle()
        }
    }

    @Test(timeout = 600_000) fun responseButtonCancelsGenerationSynthesisAndPlayback() {
        assumeTrue("Enable explicitly on a device with cached models", InstrumentationRegistry.getArguments().getString("voiceAgentNative") == "true")
        val context = instrumentation.targetContext
        System.loadLibrary("ailia")
        val speech = QnnSupport.speechQnnFileName(SpeechModelType.WHISPER_MEDIUM)
        val llm = QnnSupport.llmQnnFileName(LLMModelType.GEMMA_4_E2B)
        assumeTrue(speech != null && llm != null)
        val directory = ModelDownloader.modelDirectory(context)
        assumeTrue(File(directory, speech!!).exists() && File(directory, llm!!).exists())
        instrumentation.uiAutomation.executeShellCommand("pm grant ${context.packageName} android.permission.RECORD_AUDIO").use {
            FileInputStream(it.fileDescriptor).use { stream -> stream.readBytes() }
        }
        val results = JSONArray()
        val stageFilter = InstrumentationRegistry.getArguments().getString("voiceAgentCancelStage")
        for ((stage, description) in listOf("thinking" to "応答を考えています", "synthesis" to "音声を合成しています", "playback" to "応答を読み上げています")) {
            if (stageFilter != null && stage != stageFilter) continue
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                scenario.onActivity { it.findViewById<Spinner>(R.id.algorithmSpinner).setSelection(MainActivity.AlgorithmType.VOICE_AGENT.ordinal) }
                instrumentation.waitForIdleSync()
                scenario.onActivity {
                    it.findViewById<RadioGroup>(R.id.voiceAgentInputMode).check(R.id.voiceAgentTestInput)
                    it.findViewById<EditText>(R.id.voiceAgentTestQueries).setText("こんにちは。自己紹介をしてください。")
                    it.findViewById<Button>(R.id.voiceAgentStart).performClick()
                }
                waitForOrbState(scenario, description, 180_000)
                if (stage == "synthesis") SystemClock.sleep(8000) // Allow model loading to enter the blocking synthesis call.
                screenshot("voice-agent-slime-$stage.png")
                val beforeCancel = SystemClock.elapsedRealtime()
                scenario.onActivity {
                    val button = it.findViewById<Button>(R.id.voiceAgentSend)
                    assertTrue(button.isEnabled)
                    assertEquals(it.getString(R.string.voice_agent_interrupt), button.contentDescription)
                    assertEquals(description, it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
                    button.performClick()
                }
                waitForOrbState(scenario, "音声入力を待っています", 5000)
                waitForSendEnabled(scenario)
                results.put(JSONObject().put("stage", stage).put("microphone_ready_ms", SystemClock.elapsedRealtime() - beforeCancel))
                if (stage == "synthesis") {
                    // Submit the next recording while the old blocking call is finishing, then cancel that queued turn too.
                    scenario.onActivity { it.findViewById<Button>(R.id.voiceAgentSend).performClick() }
                    SystemClock.sleep(250)
                    scenario.onActivity {
                        assertEquals("音声を送信しています", it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
                        assertEquals(it.getString(R.string.voice_agent_interrupt), it.findViewById<Button>(R.id.voiceAgentSend).contentDescription)
                        it.findViewById<Button>(R.id.voiceAgentSend).performClick()
                    }
                    waitForOrbState(scenario, "音声入力を待っています", 5000)
                    waitForSendEnabled(scenario)
                }
                SystemClock.sleep(300)
                scenario.onActivity {
                    assertEquals("音声入力を待っています", it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
                    assertEquals("", it.findViewById<TextView>(R.id.voiceAgentOutput).text.toString())
                    assertEquals(it.getString(R.string.voice_agent_send), it.findViewById<Button>(R.id.voiceAgentSend).contentDescription)
                    it.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                }
                waitForStartEnabled(scenario, 60_000)
            }
        }
        File(context.getExternalFilesDir(null), "voice-agent-cancellation-test.json").writeText(results.toString(2))
    }

    private fun whiteArea(bitmap: Bitmap): Int {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return pixels.count { Color.red(it) > 245 && Color.green(it) > 245 && Color.blue(it) > 245 }
    }

    private fun waitForSendEnabled(scenario: ActivityScenario<MainActivity>) {
        val deadline = SystemClock.elapsedRealtime() + 5000
        while (SystemClock.elapsedRealtime() < deadline) {
            var ready = false
            scenario.onActivity { ready = it.findViewById<Button>(R.id.voiceAgentSend).isEnabled }
            if (ready) return
            SystemClock.sleep(20)
        }
        fail("Microphone send did not become ready")
    }

    private fun waitForOrbState(scenario: ActivityScenario<MainActivity>, state: String, timeoutMs: Long) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        var actual = ""
        while (SystemClock.elapsedRealtime() < deadline) {
            scenario.onActivity { actual = it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription.toString() }
            if (actual == state) return
            SystemClock.sleep(20)
        }
        fail("Expected $state but was $actual")
    }

    /** Two generated questions pass through VAD; then real microphone capture must remain idle. */
    @Test(timeout = 600_000) fun synthesizedQueriesCompleteTwoVoiceTurns() {
        assumeTrue("Enable explicitly on a device with cached models", InstrumentationRegistry.getArguments().getString("voiceAgentNative") == "true")
        val context = instrumentation.targetContext
        System.loadLibrary("ailia")
        axip.ailia.Ailia.SetTemporaryCachePath(context.cacheDir.absolutePath)
        val speechModelType = InstrumentationRegistry.getArguments().getString("voiceAgentSpeechModel")
            ?.let { SpeechModelType.valueOf(it) } ?: SpeechModelType.WHISPER_MEDIUM
        val qnnSpeech = QnnSupport.speechQnnFileName(speechModelType)
        val qnnLlm = QnnSupport.llmQnnFileName(LLMModelType.GEMMA_4_E2B)
        assumeTrue(qnnSpeech != null && qnnLlm != null)
        val directory = ModelDownloader.modelDirectory(context)
        assumeTrue(File(directory, qnnSpeech!!).exists() && File(directory, qnnLlm!!).exists())
        val cpu = axip.ailia.AiliaModel.getEnvironments().first {
            it.type == axip.ailia.AiliaEnvironment.TYPE_CPU && !QnnSupport.isQnnEnvironmentName(it.name)
        }
        val executor = Executors.newSingleThreadExecutor()
        val sample = AiliaVoiceAgentSample(context, executor)
        val states = CopyOnWriteArrayList<AiliaVoiceAgentSample.State>()
        val inputs = CopyOnWriteArrayList<String>()
        val answers = CopyOnWriteArrayList<String>()
        val completed = CountDownLatch(1)
        val stopped = CountDownLatch(1)
        val observingMicrophone = AtomicBoolean(false)
        val microphoneReady = CountDownLatch(1)
        val queries = context.getString(R.string.voice_agent_default_queries).lines()
        val speechOutputEnabled = InstrumentationRegistry.getArguments().getString("voiceAgentSpeechOutput") != "false"
        assertEquals("Exercise a second question after the first answer", 2, queries.size)
        instrumentation.uiAutomation.executeShellCommand("pm grant ${context.packageName} android.permission.RECORD_AUDIO").use {
            FileInputStream(it.fileDescriptor).use { stream -> stream.readBytes() }
        }
        var latestOutput = ""
        var failure: String? = null
        try {
            instrumentation.runOnMainSync {
                sample.start(AiliaVoiceAgentSample.Config(cpu.id, true, LLMBackend.QNN, cpu.id,
                    AiliaVoiceAgentSample.InputMode.SYNTHESIZED_QUERIES, queries, speechOutputEnabled,
                    speechModelType = speechModelType),
                    object : AiliaVoiceAgentSample.Listener {
                        override fun onState(state: AiliaVoiceAgentSample.State) {
                            states += state
                            if (state == AiliaVoiceAgentSample.State.SPEAKING ||
                                (!speechOutputEnabled && latestOutput.isNotBlank() &&
                                    state in setOf(AiliaVoiceAgentSample.State.SYNTHESIZING_QUERY, AiliaVoiceAgentSample.State.TEST_COMPLETE))) {
                                answers += latestOutput
                            }
                            if (state == AiliaVoiceAgentSample.State.TEST_COMPLETE) completed.countDown()
                        }
                        override fun onInput(text: String) { inputs += text }
                        override fun onOutput(text: String) { latestOutput = text }
                        override fun onWaveform(samples: FloatArray) {
                            if (observingMicrophone.get()) microphoneReady.countDown()
                        }
                        override fun onDownload(fileName: String, downloaded: Long, total: Long) = Unit
                        override fun onStopped(error: String?) {
                            failure = error
                            completed.countDown()
                            stopped.countDown()
                        }
                    })
            }
            assertTrue("Synthesized query test timed out", completed.await(540, TimeUnit.SECONDS))
            instrumentation.waitForIdleSync()
            val report = File(context.getExternalFilesDir(null), "voice-agent-synthesized-test.json")
            val result = JSONObject().put("queries", JSONArray(queries)).put("recognized", JSONArray(inputs))
                    .put("speech_output_enabled", speechOutputEnabled)
                    .put("answers", JSONArray(answers)).put("states", JSONArray(states.map { it.name }))
                    .put("error", failure ?: JSONObject.NULL)
            report.writeText(result.toString(2))
            assertNull("Voice agent failed", failure)
            assertEquals(2, inputs.size)
            assertTrue(inputs.first().contains("自己紹介"))
            assertTrue("The second question must also be recognized: ${inputs.last()}",
                inputs.last().contains("1") || inputs.last().contains("一"))
            assertEquals(2, answers.size)
            assertTrue(answers.all { it.isNotBlank() })
            assertTrue("Expected an answer of two: ${answers.last()}", answers.last().contains("2") || answers.last().contains("二"))
            assertFalse(states.contains(AiliaVoiceAgentSample.State.LISTENING))
            val turn = listOf(AiliaVoiceAgentSample.State.SYNTHESIZING_QUERY, AiliaVoiceAgentSample.State.WAITING_QUERY,
                AiliaVoiceAgentSample.State.PLAYING_QUERY, AiliaVoiceAgentSample.State.SUBMITTING,
                AiliaVoiceAgentSample.State.RECOGNIZING, AiliaVoiceAgentSample.State.THINKING) +
                if (speechOutputEnabled) listOf(AiliaVoiceAgentSample.State.SYNTHESIZING, AiliaVoiceAgentSample.State.SPEAKING)
                else emptyList()
            assertEquals(listOf(AiliaVoiceAgentSample.State.PREPARING) + turn + turn +
                AiliaVoiceAgentSample.State.TEST_COMPLETE, states.toList())

            // Normal turn completion must also survive the speaker-to-real-microphone handover.
            val baseline = states.size
            val beforeResume = SystemClock.elapsedRealtime()
            observingMicrophone.set(true)
            instrumentation.runOnMainSync { sample.resumeMicrophoneInput() }
            assertTrue("Microphone did not resume", microphoneReady.await(5, TimeUnit.SECONDS))
            // Capture resumes before Whisper is loaded. Wait for SDK initialization and the
            // already queued pushes, so the idle assertion actually exercises Silero.
            executor.submit {}.get(30, TimeUnit.SECONDS)
            result.put("microphone_sdk_ready_ms", SystemClock.elapsedRealtime() - beforeResume)
            SystemClock.sleep(2200)
            executor.submit {}.get(30, TimeUnit.SECONDS)
            val microphoneStates = states.drop(baseline)
            result.put("microphone_resume_states", JSONArray(microphoneStates.map { it.name }))
                .put("microphone_observed_ms", SystemClock.elapsedRealtime() - beforeResume)
            report.writeText(result.toString(2))
            assertEquals("No automatic VAD submission after playback without a new utterance",
                listOf(AiliaVoiceAgentSample.State.LISTENING), microphoneStates)
        } finally {
            instrumentation.runOnMainSync { sample.stop() }
            assertTrue("Models were not released", stopped.await(30, TimeUnit.SECONDS))
            executor.shutdown()
        }
    }

    private fun waitForStartEnabled(scenario: ActivityScenario<MainActivity>, timeoutMs: Long = 5000) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            var enabled = false
            scenario.onActivity { enabled = it.findViewById<Button>(R.id.voiceAgentStart).isEnabled }
            if (enabled) return
            SystemClock.sleep(50)
        }
        fail("Start button did not become enabled after cancellation")
    }

    @Test(timeout = 240_000) fun speechBufferedRecognizesQuietInputAndResetsForTheNextQuestion() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("voiceAgentNative") == "true")
        val context = instrumentation.targetContext
        System.loadLibrary("ailia")
        axip.ailia.Ailia.SetTemporaryCachePath(context.cacheDir.absolutePath)
        val qnnFile = QnnSupport.speechQnnFileName(SpeechModelType.WHISPER_LARGE_V3_TURBO)
        assumeTrue(qnnFile != null && File(ModelDownloader.modelDirectory(context), qnnFile).exists())
        val cpu = axip.ailia.AiliaModel.getEnvironments().first {
            it.type == axip.ailia.AiliaEnvironment.TYPE_CPU && !QnnSupport.isQnnEnvironmentName(it.name)
        }
        val executor = Executors.newSingleThreadExecutor()
        try {
            executor.submit {
                val speech = AiliaSpeechSample(ModelDownloader.modelDirectory(context)).apply {
                    currentModelType = SpeechModelType.WHISPER_LARGE_V3_TURBO
                    useQnn = true
                    language = "ja"
                }
                try {
                    assertTrue(speech.downloadModel())
                    assertTrue(speech.initializeSpeech(cpu.id, liveMode = false))
                    val turn = speech.createBufferedTurn()
                    val reference = AudioUtil().loadRawAudio(context.resources.openRawResource(R.raw.reference_audio_girl))
                    val audio = SpeakerVerificationAudio.toMono16k(reference.audioData, reference.channels, reference.sampleRate)
                    val baseRms = voiceAgentAudioRms(audio)
                    val results = JSONArray()
                    for (gain in listOf(1f, .006f / baseRms)) {
                        repeat(15) { assertFalse("Reset must not leave the previous utterance buffered",
                            turn.push(FloatArray(1600), 1, 16000)) }
                        val input = FloatArray(audio.size) { audio[it] * gain }
                        var buffered = false
                        for (offset in input.indices step 1600) {
                            buffered = turn.push(input.copyOfRange(offset, minOf(offset + 1600, input.size)), 1, 16000)
                            if (buffered) break
                        }
                        repeat(20) { if (!buffered) buffered = turn.push(FloatArray(1600), 1, 16000) }
                        assertTrue("Silero must detect speech even below the old RMS threshold", buffered)
                        val text = voiceAgentTranscript(turn.finish {})
                        assertTrue("Speech should be transcribed: $text", text.isNotBlank())
                        results.put(JSONObject().put("rms", voiceAgentAudioRms(input).toDouble())
                            .put("recognized", text).put("buffered", buffered))
                    }
                    repeat(15) { assertFalse(turn.push(FloatArray(1600), 1, 16000)) }
                    assertTrue("Silence after reset must not repeat previous text", turn.finish {}.isEmpty())
                    File(context.getExternalFilesDir(null), "voice-agent-silero-buffered-test.json").writeText(results.toString(2))
                } finally { speech.releaseSpeech() }
            }.get(220, TimeUnit.SECONDS)
        } finally { executor.shutdown() }
    }

    private fun screenshot(name: String) {
        instrumentation.waitForIdleSync()
        SystemClock.sleep(350) // Let ActionBar hide/show and the next layout frame finish.
        val bitmap = instrumentation.uiAutomation.takeScreenshot() ?: return
        File(instrumentation.targetContext.getExternalFilesDir(null), name).outputStream().use {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        bitmap.recycle()
    }
}
