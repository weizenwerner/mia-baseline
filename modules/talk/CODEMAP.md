# CODE MAP — `modules/talk/260215_main.py`

## High-level overview (10–15 lines)

1. The script is a voice-loop assistant runtime (“Mia Talk”) that continuously listens to microphone input, detects speech with VAD, transcribes with Whisper, sends prompts to Ollama, and speaks responses via Piper + PipeWire playback.
2. It maintains long-lived process state via global flags/events (`STOP`, `CANCEL_TTS`, `SPEAKING`) and session memory persisted as JSON messages.
3. Audio capture is done in short `pw-record` chunks; these are frame-analyzed (RMS + `webrtcvad`) and assembled into one utterance WAV.
4. It supports wake-word gating (`MIA_REQUIRE_WAKE`, `MIA_WAKEWORDS`) and awake timeout windows.
5. It supports immediate barge-in/stop commands even during playback; stop handling cancels current TTS/playback and continues listening.
6. User/assistant turns are appended to session history and written atomically to disk.
7. LLM output is streamed from Ollama’s `/api/chat`; optional delta chunking feeds low-latency incremental TTS.
8. TTS synthesis is done by Piper into WAV files, then output through `pw-play`.
9. During speaking, input handling changes to stricter short-listening parameters to preferentially catch stop/exit commands.
10. Echo guard heuristics suppress likely feedback loops where assistant output is re-captured by microphone shortly after playback.
11. The process can shut down from signal handlers (`SIGINT`/`SIGTERM`), exit phrases, or fatal startup dependency errors.
12. Warmup optionally pre-loads the Ollama model via CLI before entering the main loop.

## Execution flow (process start → shutdown)

1. **Interpreter entrypoint**
   - `if __name__ == "__main__":` calls `main()` in a top-level try/except to log uncaught tracebacks.
2. **Global setup before `main()`**
   - Module import initializes globals/events, registers signal handlers, and defines utility/helpers.
3. **`main()` startup/config**
   - Reads env-based config (audio/VAD, wake/stop/exit, Whisper/Ollama/TTS/session).
   - Verifies `webrtcvad` import; may exit with status 2 on failure.
   - Optionally sets PipeWire default sink (`wpctl`) and optionally warms up Ollama (`ollama run`).
   - Loads/creates session JSON; injects initial system message if empty.
4. **Persistent components initialized**
   - `recent_assistant` deque for echo detection.
   - `TTSManager` object created (worker thread not started yet).
5. **Main loop (`while not STOP`)**
   - Adjusts awake status timeout.
   - Determines if currently speaking (`SPEAKING.is_set()`) and chooses stricter listen params when speaking.
   - Calls `record_vad_to_wav(...)` to block until a speech segment is captured or timeout/empty.
   - Runs `whisper_transcribe(...)` on captured WAV; deletes temp WAV in `finally`.
   - Filters noise transcripts.
   - Handles stop/barge-in (always active): `tts_mgr.stop()` + continue.
   - Handles exit phrases: break loop.
   - If speaking and transcript is non-stop: ignore and continue.
   - Wake-word logic: mark awake, enforce wake requirement, strip wake prefix.
   - Echo guard: for a short post-playback window, ignore high-overlap non-question transcripts.
   - Append user text to session messages and clear `CANCEL_TTS` before generating assistant response.
6. **LLM + TTS path**
   - If streaming TTS enabled:
     - Start TTS worker thread (`tts_mgr.start()`) if not running.
     - Create `StreamChunker` and pass `on_delta` callback into Ollama streaming call.
     - Each emitted chunk is queued to TTS worker; worker synthesizes and plays sequentially.
   - If streaming disabled:
     - Fetch full assistant text only (no queued speaking behavior here).
7. **Post-generation**
   - If canceled during generation, stop TTS and continue loop.
   - Fallback text if assistant output empty.
   - Append assistant response to session + echo history and persist session JSON.
8. **Shutdown sequence**
   - On loop exit, if streaming mode is enabled: `tts_mgr.stop(reason="shutdown")` to terminate worker/playback.
   - Log graceful shutdown and return.

### Threads, queues, subprocesses

- **Main thread**: runs signal handling target state updates, VAD capture/transcribe loop, Ollama stream consumption, state machine decisions.
- **TTS worker thread (`TTSManager._worker`)**: blocking queue consumer (`queue.Queue`), performs synth + playback serially.
- **Queue**: `TTSManager.q` (`maxsize=64`), carries text chunks and `None` sentinel for shutdown.
- **Subprocess calls**:
  - `pw-record` (audio capture; repeated chunk recording).
  - `whisper-cli` (STT, GPU-first then optional CPU fallback).
  - `ollama run` (optional warmup).
  - `pw-play` (audio playback via `Popen`, polled loop).
  - `wpctl status/set-default`, `pkill pw-play`, `command -v` checks.
- **HTTP stream**:
  - `requests.post(..., stream=True)` to Ollama `/api/chat`, line-wise JSON delta consumption.

## State machine / modes / flags

### Global flags/events

- `STOP` (bool)
  - Set by SIGINT/SIGTERM handler; also causes loops to exit.
  - Read in main loop, VAD loop, streaming loop, TTS worker/playback loops.
- `CANCEL_TTS` (`threading.Event`)
  - Set on stop/barge-in/shutdown.
  - Clears before a new assistant response begins.
  - Causes LLM stream read and TTS worker/playback to abort early.
- `SPEAKING` (`threading.Event`)
  - Set during piper synth and playback; cleared when those end/cancel.
  - Governs “speaking mode” input policy (only stop/exit effectively honored).
- `LAST_PLAY_END_TS` (float timestamp)
  - Updated when playback stops/ends; used to enable short-time echo guard window.

### Conversational modes

1. **Sleeping mode (`awake=False`)**
   - If `MIA_REQUIRE_WAKE=True`, normal utterances are ignored unless wakeword detected.
   - Transition to awake on wakeword match.
2. **Awake mode (`awake=True`)**
   - Maintained until `awake_timeout_sec` inactivity timeout.
   - Timeout transition: awake → sleeping.
3. **Speaking mode (`SPEAKING.is_set()=True`)**
   - Uses stricter recording thresholds/shorter windows.
   - Non-stop transcripts are ignored.
   - Stop command transitions to canceled output + continue listening.
4. **Generation active (implicit)**
   - During Ollama streaming; can be interrupted by `STOP` or `CANCEL_TTS`.
5. **Shutdown mode (implicit)**
   - Triggered by signal, exit phrase, or fatal exception path.

### Transition triggers (key examples)

- **Wake transition**: transcript contains configured wakeword.
- **Sleep timeout**: `time.time() > awake_until`.
- **Barge-in transition**: transcript matches stop command patterns/phrases.
- **Exit transition**: transcript contains configured exit phrase.
- **Echo suppression branch**: transcript overlaps recent assistant words within echo window and is not question-like.
- **Signal transition**: SIGINT/SIGTERM sets STOP + CANCEL_TTS, clears SPEAKING.

## I/O map

### Audio input

- Source: system microphone via `pw-record` CLI.
- Intermediate files:
  - Short chunks in `/tmp/mia_talk_chunks/chunk_<uuid>.wav` (deleted after read).
  - Utterance file `/tmp/mia_talk/mia_in_<uuid>.wav` (deleted after transcription attempt).
- Processing: WAV read + mono PCM extraction + RMS + VAD frame logic.

### Audio output

- TTS WAV output files in `/data/mia/data/audio/out/<uuid>.wav`.
- Playback via `pw-play <wav>` subprocess.
- Hard stop path kills `pw-play` processes via `pkill`.

### Files written

- Log append file: `/data/mia/logs/talk/talk.log`.
- Session JSON:
  - default: `/data/mia/memory/sessions/talk_default.json`
  - or new timestamped file in `/data/mia/memory/sessions/talk_*.json`
  - save is atomic via temp file + replace.

### Network calls

- Ollama chat streaming HTTP call:
  - `POST <MIA_OLLAMA_HOST>/api/chat` with `stream: true` and generation options.
- No other explicit external HTTP services in this file.

### Environment variables used

- **General/debug**: `MIA_DEBUG`
- **Audio sink**: `MIA_AUDIO_SINK_ID`
- **Ollama**:
  - `MIA_OLLAMA_HOST`, `MIA_LLM`, `MIA_KEEPALIVE`, `MIA_WARMUP`
  - `MIA_NUM_PREDICT`, `MIA_TEMPERATURE`, `MIA_TOP_P`, `MIA_REPEAT_PENALTY`
  - `MIA_OLLAMA_STALL_TIMEOUT_SEC`
- **Whisper**:
  - `MIA_WHISPER_CLI`, `MIA_WHISPER_MODEL`, `MIA_WHISPER_LANG`, `MIA_WHISPER_FORCE_CPU`
- **Audio/VAD**:
  - `MIA_RATE`, `MIA_FRAME_MS`, `MIA_VAD_MODE`, `MIA_SILENCE_LIMIT_SEC`, `MIA_MIN_SPEECH_SEC`
  - `MIA_TRIGGER_RATIO`, `MIA_RING_MS`, `MIA_MAX_SEC`, `MIA_RMS_THRESHOLD`
  - `MIA_CHUNK_SEC`
- **Speaking listen tuning**:
  - `MIA_SPEAK_LISTEN_MAX_SEC`, `MIA_SPEAK_SILENCE_LIMIT_SEC`, `MIA_SPEAK_MIN_SPEECH_SEC`, `MIA_SPEAK_RMS_THRESHOLD`
- **Wake/stop/exit**:
  - `MIA_REQUIRE_WAKE`, `MIA_WAKEWORDS`, `MIA_EXITWORDS`, `MIA_AWAKE_TIMEOUT_SEC`, `MIA_STOPWORDS`
- **TTS streaming**:
  - `MIA_TTS_STREAM`, `MIA_TTS_STREAM_MIN_CHARS`, `MIA_TTS_STREAM_MAX_CHARS`
- **Echo guard**:
  - `MIA_ECHO_OVERLAP`, `MIA_ECHO_HISTORY`, `MIA_ECHO_WINDOW_SEC`
- **Session**:
  - `MIA_SESSION_MODE`, `MIA_SESSION`
- **Piper**:
  - `PIPER_MODEL`

## Potential blocking points (exact functions/locations and why)

1. `safe_run(...)` → `subprocess.run(..., timeout=timeout_sec)`
   - Blocks until command returns or timeout.
2. `_record_chunk_wav(...)` → `subprocess.run(timeout pw-record ... )`
   - Blocks for chunk duration plus process overhead.
3. `record_vad_to_wav(...)` main `while` loop
   - Potentially long-running until trigger/silence/max duration; includes repeated capture + parsing loops.
4. `_run_whisper(...)` / `whisper_transcribe(...)`
   - Blocking `subprocess.run` for STT, possibly long on CPU fallback.
5. `ollama_warmup_cli(...)`
   - Blocking `subprocess.run("ollama run ...")` for model init.
6. `ollama_chat_http_stream(...)`
   - Blocking network call `requests.post(..., stream=True, timeout=600)` and iterative `iter_lines` consumption.
   - Can stall waiting for stream lines until stall timeout logic exits.
7. `piper_speak(...)`
   - Blocking `subprocess.run` for TTS synthesis.
8. `play_wav(...)`
   - Blocks in poll loop until `pw-play` exits or cancel/stop triggers termination.
9. `TTSManager.enqueue(...)`
   - `queue.put(..., timeout=1.0)` can block up to 1 second on full queue.
10. `TTSManager.stop(...)`
    - `queue.put(None, timeout=0.2)` and `thread.join(timeout=2.0)` block briefly during shutdown/stop.
11. `TTSManager._worker(...)`
    - `self.q.get()` blocks waiting for next chunk/sentinel.
12. `load_session(...)` / `save_session(...)`
    - Blocking disk I/O; usually short but synchronous.

## Timing-critical sections vs deferrable work

### Latency-sensitive / timing-critical

- **Barge-in stop path**
  - `is_stop_command(...)` decision + `tts_mgr.stop()` + `stop_playback()` should be fast to cut speech output quickly.
- **Speaking-mode listening loop**
  - Uses stricter and shorter VAD settings to catch stop/exit promptly during output.
- **Streaming TTS pipeline**
  - `ollama_chat_http_stream` deltas → `StreamChunker.push` → `TTSManager.enqueue` directly affects time-to-first-audio.
- **Playback cancel polling**
  - `play_wav` loop checks cancel/stop every 20ms; this is central to interruption responsiveness.

### Can be deferred / less latency-critical

- Session file persistence after each turn (important for durability, not immediate UX latency within an active utterance).
- Warmup (`MIA_WARMUP`) happens before first interaction; improves later latency but is optional upfront cost.
- Detailed logging writes and debug verbosity.
- Echo-history maintenance (`recent_assistant` append/overlap calculations) is lightweight and not hard real-time.

## Assumptions / unclear points

- The file itself does not define external service availability/SLAs; assumptions about microphone, PipeWire, Ollama server, and model presence are inferred from subprocess/network usage.
- “States” are partly explicit (`awake`, event flags) and partly implicit by control-flow branches (e.g., generation active, shutdown path).
- No explicit retry/backoff policy for Ollama HTTP beyond stall timeout and outer loop continuation.
- Non-streaming TTS mode currently logs assistant text but does not enqueue playback in this file (documented as observed behavior, not interpreted as bug).
