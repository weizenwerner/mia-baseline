# Stand

- Commit: `e8813a3`
- Ziel-Datei: `modules/talk/main.py`
- Erstellt am: `2026-02-19 10:11:05 +0000`

> Hinweis: `modules/talk/main.py` ist im aktuellen Repository-Stand nicht vorhanden. Diese Dokumentation folgt der gewünschten Zielbenennung (`main.py`) und beschreibt den aktuellen Mia-Talk-Laufzeitfluss auf Basis der vorhandenen Talk-Implementierung.

# CODEMAP – `modules/talk/main.py`

## Kurzüberblick (10–15 Zeilen)

1. Mia-Talk ist ein kontinuierlicher Sprach-Loop für Aufnahme, Transkription, Antwortgenerierung und Audioausgabe.
2. Audio wird über PipeWire aufgenommen und in kurzen Chunks verarbeitet.
3. Voice Activity Detection (VAD) bestimmt Start/Ende eines Sprachsegments.
4. Das Segment wird per Whisper-CLI in Text umgewandelt.
5. Wakeword-Logik kann erzwingen, dass nur nach Aktivierung reagiert wird.
6. Stop/Barge-in-Kommandos greifen jederzeit, auch während laufender Ausgabe.
7. Nutzertexte werden mit Session-Kontext an Ollama (`/api/chat`) gestreamt.
8. Antwortdeltas können in sprechbare Chunks zerlegt werden.
9. TTS läuft über Piper und erzeugt WAV-Dateien im Output-Verzeichnis.
10. Playback erfolgt über `pw-play`, mit sofortigem Abbruchpfad.
11. Echo-Guard reduziert Fehltrigger durch Rückkopplung nach Assistant-Ausgabe.
12. Konversationen werden als Session-JSON fortgeschrieben.
13. Logs gehen in Datei und optional auf stdout.
14. Shutdown erfolgt per Signal, Exit-Phrase oder Fehlerpfad.

## Ablaufdiagramm (Textform)

### Start → Init → Hauptloop → Shutdown

1. **Start**
   - Prozess lädt globale Flags/Events (`STOP`, `CANCEL_TTS`, `SPEAKING`) und registriert Signalhandler.
2. **Init**
   - Umgebungsvariablen einlesen.
   - Abhängigkeiten prüfen (insb. `webrtcvad`).
   - Optional: Audio-Default-Sink setzen.
   - Optional: Ollama-Warmup ausführen.
   - Session laden/erstellen, ggf. Systemnachricht initialisieren.
3. **Hauptloop**
   - Awake-Timeout prüfen.
   - Listening-Parameter nach Modus wählen (normal/speaking-mode).
   - Audiosegment per VAD aufnehmen.
   - Segment mit Whisper transkribieren.
   - Reihenfolge bei Transcript:
     - leere/Noise-Transkripte ignorieren,
     - Stop/Barge-in sofort behandeln,
     - Exitphrase behandeln,
     - bei `SPEAKING` Nicht-Stop ignorieren,
     - Wake/Awake-Regeln anwenden,
     - Echo-Guard anwenden.
   - User-Nachricht in Session einfügen.
   - Ollama-Streaming starten; Deltas ggf. in TTS-Queue geben.
   - Assistant-Text loggen, speichern, Session persistieren.
4. **Shutdown**
   - Laufende TTS/Playback-Aktivität beenden.
   - Graceful-Shutdown loggen.

### Threads / Queues / Subprocesses

- **Main-Thread**: Aufnahme, STT, Steuerlogik, HTTP-Streaming, Session-Schreiben.
- **TTS-Worker-Thread**: konsumiert Textqueue und führt Synthese + Playback seriell aus.
- **Queue**: `queue.Queue(maxsize=64)`, `None` als Stop-Sentinel.
- **Subprocesses**:
  - `pw-record` (Audio-In),
  - `whisper-cli` (STT),
  - `ollama run` (Warmup),
  - `piper`/`piper-tts` (Synthese),
  - `pw-play` (Playback),
  - `wpctl`, `pkill`, `command -v` (Hilfsaufrufe).

## State-/Event-Übersicht

### Globale Events/Flags

- **`STOP`**
  - globaler Exit-Flag.
  - Trigger: `SIGINT`/`SIGTERM`.
  - Wirkung: Schleifen und Workerpfade fahren herunter.

- **`CANCEL_TTS`**
  - Abbruch laufender Ausgabe/Streaming-Verarbeitung.
  - Trigger: Stop/Barge-in, Shutdown, Signalpfad.
  - Wirkung: Delta-Verarbeitung, TTS-Queue-Abarbeitung und Playback werden beendet.

- **`SPEAKING`**
  - kennzeichnet aktive Synthese/Audioausgabe.
  - Trigger setzen/löschen entlang TTS- und Playback-Lebenszyklus.
  - Wirkung: speaking-mode mit strikter Input-Behandlung.

### Modus-/Zustandslogik

- **sleeping (`awake=False`)**: bei Wake-Pflicht werden normale Inputs ignoriert.
- **awake (`awake=True`)**: Reaktionsmodus bis Timeout.
- **speaking-mode (`SPEAKING=True`)**: Hörfenster priorisiert Stop/Exit.
- **shutdown**: ausgelöst durch Exit-Phrase, Signal oder fatalen Fehler.

### Trigger

- Wakeword erkannt → `awake=True`.
- Awake-Timeout erreicht → `awake=False`.
- Stopkommando erkannt → `CANCEL_TTS.set()` + Playback-Stopp.
- Exitphrase erkannt → Hauptloop-Ende.

## I/O Map

### Audio-In

- Mikrofonaufnahme über `pw-record`.
- Temporäre Chunk-Dateien: `/tmp/mia_talk_chunks/chunk_<uuid>.wav`.
- Temporäre Utterance-Dateien: `/tmp/mia_talk/mia_in_<uuid>.wav`.

### Audio-Out

- TTS-WAVs: `/data/mia/data/audio/out/<uuid>.wav`.
- Ausgabe per `pw-play`.
- Harte Unterbrechung per `pkill -x pw-play`.

### Dateien / Persistenz

- Logs: `/data/mia/logs/talk/talk.log`.
- Sessions:
  - default: `/data/mia/memory/sessions/talk_default.json`
  - neu: `/data/mia/memory/sessions/talk_YYYYMMDD_HHMMSS_<id>.json`
- Session-Write atomar über temporäre Datei + replace.

### Netzwerk

- Ollama Chat-Streaming: `POST {MIA_OLLAMA_HOST}/api/chat`.
- Streaming-Verarbeitung über lineweise JSON-Deltas.

## Blocker-Stellen (potenziell hängend)

- `safe_run(...)` (`subprocess.run` mit Timeout).
- `_record_chunk_wav(...)` (`pw-record`-Chunklauf).
- `record_vad_to_wav(...)` (wiederholte Chunkaufnahme + VAD-Loop).
- `_run_whisper(...)` / `whisper_transcribe(...)` (STT-Subprocess).
- `ollama_warmup_cli(...)` (`ollama run`).
- `ollama_chat_http_stream(...)` (`requests.post(..., stream=True)` + `iter_lines`).
- `piper_speak(...)` (TTS-Subprocess).
- `play_wav(...)` (Warten auf `pw-play`-Ende/Abbruch).
- `TTSManager._worker(...)` (`queue.get()` blockierend).
- `TTSManager.enqueue(...)` (`queue.put(..., timeout=1.0)`).
- `TTSManager.stop(...)` (`queue.put(None, ...)` + `join(...)`).

## Konfig-Übersicht (ENV, gruppiert)

### Audio / VAD

- `MIA_AUDIO_SINK_ID`
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

### Speaking-Mode

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
