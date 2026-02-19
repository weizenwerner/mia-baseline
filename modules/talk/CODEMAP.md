# CODEMAP – `modules/talk/260215_main.py`

> Hinweis: In diesem Repository existiert aktuell keine Datei `modules/talk/main.py`. Diese CODEMAP beschreibt deshalb die vorhandene Laufzeitdatei `modules/talk/260215_main.py`.

## Kurzüberblick (10–15 Zeilen)

1. Das Programm implementiert einen kontinuierlichen Voice-Loop für „Mia Talk“.
2. Es nimmt Mikrofon-Audio über PipeWire (`pw-record`) in kurzen Chunks auf.
3. Ein VAD-Recorder (RMS + `webrtcvad`) erkennt Sprachbeginn/-ende und schreibt ein Utterance-WAV.
4. Dieses WAV wird mit `whisper-cli` zu Text transkribiert.
5. Optional ist Wakeword-Pflicht aktiv; ohne Wakeword werden Eingaben verworfen.
6. Stop-/Barge-in-Kommandos werden jederzeit erkannt, auch während laufender Ausgabe.
7. User-Texte gehen als Chat-Historie an Ollama (`/api/chat`, Streaming).
8. Antwort-Deltas können in Text-Chunks zerlegt und direkt an TTS übergeben werden.
9. TTS läuft über Piper und schreibt WAV-Dateien ins Output-Verzeichnis.
10. Audioausgabe erfolgt über `pw-play`; laufendes Playback kann sofort gestoppt werden.
11. Ein Echo-Guard filtert kurz nach Playback-Ende mögliche Rückkopplungs-Transkripte.
12. Sitzungsdaten werden als JSON geladen/fortgeschrieben (system/user/assistant messages).
13. Logs werden in eine Datei geschrieben und optional auf stdout ausgegeben.
14. Shutdown erfolgt über Signal (`SIGINT`/`SIGTERM`), Exit-Phrase oder Prozessfehler.

## Ablaufdiagramm (Textform)

### Start → Init → Hauptloop → Shutdown

1. **Prozessstart**
   - Modul lädt globale Flags/Events (`STOP`, `CANCEL_TTS`, `SPEAKING`) und Signalhandler.
2. **Init in `main()`**
   - Liest ENV-Konfiguration.
   - Prüft `webrtcvad`.
   - Optional: Default-Sink setzen (`wpctl set-default`).
   - Optional: Ollama-Warmup (`ollama run ...`).
   - Session-Datei laden/erzeugen, ggf. System-Prompt setzen.
3. **Laufende Komponenten initialisieren**
   - `recent_assistant` (Deque für Echo-Guard).
   - `TTSManager` (Queue + Worker-Thread, Start erst bei Bedarf).
4. **Hauptloop (`while not STOP`)**
   - Awake-Timeout prüfen.
   - Listening-Modus wählen (normal oder speaking-mode).
   - `record_vad_to_wav(...)` erstellt Input-Utterance.
   - `whisper_transcribe(...)` erzeugt Text.
   - Reihenfolge der Prüfungen:
     - Noise/leer ignorieren.
     - Stop/Barge-in: `tts_mgr.stop(...)`, weiter zuhören.
     - Exit: Loop verlassen.
     - Wenn `SPEAKING`: Nicht-Stop-Inputs ignorieren.
     - Wakeword/awake-Regeln anwenden.
     - Echo-Guard anwenden.
   - User-Text zur Session hinzufügen.
   - Ollama-Streaming starten.
   - Bei Streaming-TTS: Deltas → `StreamChunker` → `TTSManager.enqueue()`.
   - Assistant-Text zur Session hinzufügen und speichern.
5. **Shutdown**
   - Bei aktivem TTS-Streaming: `tts_mgr.stop(reason="shutdown")`.
   - Graceful-Shutdown-Log.

### Threads / Queues / Subprocesses

- **Main-Thread**: kompletter Dialog- und Zustandsfluss, VAD/Whisper/Ollama-Steuerung.
- **TTS-Worker-Thread** (`TTSManager._worker`): verarbeitet Textqueue seriell (clean → Piper → Playback).
- **Queue**: `queue.Queue(maxsize=64)` in `TTSManager`, inkl. `None`-Sentinel für Worker-Ende.
- **Subprocesses**:
  - `pw-record` (Audio-In),
  - `whisper-cli` (STT),
  - `ollama run` (Warmup),
  - `pw-play` (Audio-Out),
  - `wpctl`, `pkill`, `command -v`.

## State-/Event-Übersicht

### Globale Flags/Events

- **`STOP` (bool)**
  - Bedeutung: globaler Exit-Flag.
  - Trigger: `SIGINT`/`SIGTERM` via `on_signal(...)`.
  - Wirkung: beendet Hauptloop, VAD-Loop, Streaming-Loop, Worker-Pfade.

- **`CANCEL_TTS` (`threading.Event`)**
  - Bedeutung: laufende Ausgabe/Streaming abbrechen.
  - Trigger: Stop/Barge-in, Shutdown, Signalhandler.
  - Wirkung: beendet Ollama-Delta-Verarbeitung, Playback und Worker-Abarbeitung.

- **`SPEAKING` (`threading.Event`)**
  - Bedeutung: Mia gibt gerade Audio aus (Synthese oder Playback aktiv).
  - Trigger setzen: in `piper_speak(...)` und `play_wav(...)`.
  - Trigger löschen: bei Ende/Fehler/Stop von TTS/Playback.
  - Wirkung: aktiviert speaking-mode (striktere Hörparameter; Nicht-Stop wird ignoriert).

### Konversationszustände

- **sleeping (`awake=False`)**
  - Bei `MIA_REQUIRE_WAKE=True` werden normale Eingaben ignoriert.
  - Übergang zu awake durch Wakeword.

- **awake (`awake=True`)**
  - Aktiv bis `awake_until` überschritten wird.
  - Timeout-Übergang zurück zu sleeping.

- **speaking-mode (`SPEAKING=True`)**
  - Kürzere/strengere Aufnahmeparameter (`MIA_SPEAK_*`).
  - Primär für Stop/Exit-Erkennung während Ausgabe.

- **shutdown**
  - Trigger: Exit-Phrase, Signal, fataler Fehlerpfad.

### Trigger (praktisch)

- Wakeword erkannt → `awake=True`.
- Awake-Timeout erreicht → `awake=False`.
- Stop-Text erkannt (`is_stop_command`) → `CANCEL_TTS.set()` + Playback-Stopp.
- Exit-Phrase erkannt (`contains_any(..., exitwords)`) → Loop-Ende.

## I/O Map

### Audio In

- Aufnahme über `pw-record`.
- Temporäre Chunk-Dateien: `/tmp/mia_talk_chunks/chunk_<uuid>.wav`.
- Temporäre Utterance-Dateien: `/tmp/mia_talk/mia_in_<uuid>.wav`.

### Audio Out

- Piper-Ausgabedateien: `/data/mia/data/audio/out/<uuid>.wav`.
- Playback: `pw-play <wav>`.
- Sofortiger Stopp über `pkill -x pw-play` (User-scope).

### Sessions / Logs / Persistenz

- Session default: `/data/mia/memory/sessions/talk_default.json`.
- Session neu: `/data/mia/memory/sessions/talk_YYYYMMDD_HHMMSS_<id>.json`.
- Session-Save atomar via `.tmp` + `replace`.
- Logfile: `/data/mia/logs/talk/talk.log`.

### Netzwerk (Ollama HTTP)

- `POST {MIA_OLLAMA_HOST}/api/chat`
- Streaming-Antwort via `iter_lines(...)`.
- Payload enthält `model`, `messages`, `stream`, `keep_alive`, `options`.

## Blocker-Stellen (potenziell blockierend)

- `safe_run(...)` → `subprocess.run(..., timeout=...)`.
- `_record_chunk_wav(...)` → blockiert während `pw-record`-Chunklauf.
- `record_vad_to_wav(...)` → lange Schleife bis Trigger/Silence/Timeout.
- `_run_whisper(...)` / `whisper_transcribe(...)` → blockierender STT-Subprocess.
- `ollama_warmup_cli(...)` → blockierender `ollama run`.
- `ollama_chat_http_stream(...)` → blockierender HTTP-Stream (`requests.post`, `iter_lines`).
- `piper_speak(...)` → blockierender Piper-Subprocess.
- `play_wav(...)` → wartet bis `pw-play` endet oder abgebrochen wird.
- `TTSManager._worker(...)` → `self.q.get()` blockiert auf Queue-Input.
- `TTSManager.enqueue(...)` → `q.put(..., timeout=1.0)` kann kurz blockieren.
- `TTSManager.stop(...)` → `q.put(None, timeout=0.2)` + `join(timeout=2.0)`.

## Konfig-Übersicht (ENV-Variablen)

### Audio / VAD

- `MIA_RATE`
- `MIA_FRAME_MS`
- `MIA_VAD_MODE`
- `MIA_SILENCE_LIMIT_SEC`
- `MIA_MIN_SPEECH_SEC`
- `MIA_TRIGGER_RATIO`
- `MIA_RING_MS`
- `MIA_MAX_SEC`
- `MIA_RMS_THRESHOLD`
- `MIA_CHUNK_SEC`
- `MIA_AUDIO_SINK_ID`

### Speaking-Mode (während Ausgabe)

- `MIA_SPEAK_LISTEN_MAX_SEC`
- `MIA_SPEAK_SILENCE_LIMIT_SEC`
- `MIA_SPEAK_MIN_SPEECH_SEC`
- `MIA_SPEAK_RMS_THRESHOLD`

### Wake / Stop / Exit

- `MIA_REQUIRE_WAKE`
- `MIA_WAKEWORDS`
- `MIA_EXITWORDS`
- `MIA_AWAKE_TIMEOUT_SEC`
- `MIA_STOPWORDS`

### Whisper

- `MIA_WHISPER_CLI`
- `MIA_WHISPER_MODEL`
- `MIA_WHISPER_LANG`
- `MIA_WHISPER_FORCE_CPU`

### Ollama

- `MIA_OLLAMA_HOST`
- `MIA_LLM`
- `MIA_KEEPALIVE`
- `MIA_WARMUP`
- `MIA_NUM_PREDICT`
- `MIA_TEMPERATURE`
- `MIA_TOP_P`
- `MIA_REPEAT_PENALTY`
- `MIA_OLLAMA_STALL_TIMEOUT_SEC`

### TTS

- `MIA_TTS_STREAM`
- `MIA_TTS_STREAM_MIN_CHARS`
- `MIA_TTS_STREAM_MAX_CHARS`
- `PIPER_MODEL`

### Session

- `MIA_SESSION_MODE`
- `MIA_SESSION`

### Echo-Guard

- `MIA_ECHO_OVERLAP`
- `MIA_ECHO_HISTORY`
- `MIA_ECHO_WINDOW_SEC`

### Debug

- `MIA_DEBUG`
