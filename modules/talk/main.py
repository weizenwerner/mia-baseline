#!/usr/bin/env python3
"""
Mia Talk (Voice Loop)

Ziel:
- Mikrofon aufnehmen (PipeWire pw-record)
- VAD (webrtcvad) erkennt Sprachsegmente (Start/Ende)
- Whisper transkribiert Audio -> Text
- Wakeword-Logik (optional): Nur reagieren, wenn "Hey Mia" etc.
- Stop/Barge-in: während Mia spricht kann Nutzer "Stopp/Pause" sagen -> Playback sofort stoppen
- LLM (Ollama /api/chat) streamt Antwort
- TTS (Piper) erzeugt WAV
- Playback (pw-play) spielt WAV ab
- Echo-Guard: verhindert, dass Whisper Mias eigene Stimme als neuen User-Input interpretiert

WICHTIGES DESIGN:
- SPEAKING Event bedeutet: "Es läuft gerade Synth/Playback" (nicht: Worker-Thread existiert).
  -> Dadurch hängt der Dialog nicht im Modus "STOP/ENDE während Mia spricht".
"""

import os
import re
import sys
import json
import time
import uuid
import wave
import signal
import subprocess
import collections
import threading
import queue
import traceback
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Callable, Deque
from collections import deque

# DeprecationWarnings (z.B. audioop) unterdrücken
warnings.filterwarnings("ignore", category=DeprecationWarning)

# audioop: wird für RMS-Berechnung (Lautstärke) genutzt, ist aber in Python 3.13 deprecated.
# Wir versuchen es zu importieren. Falls nicht verfügbar -> audioop=None und VAD scheitert.
try:
    import audioop  # type: ignore
except Exception:
    audioop = None  # type: ignore

# requests: wird für HTTP Streaming zu Ollama benötigt
try:
    import requests  # type: ignore
except Exception:
    requests = None

# Globaler Stop-Schalter: sobald True -> Hauptloop beendet sich.
STOP = False

# CANCEL_TTS: wenn gesetzt -> TTS/Playback/Streaming abbrechen
CANCEL_TTS = threading.Event()

# SPEAKING: "Mia gibt gerade Audio aus" (TTS/Playback aktiv)
# (nicht: "TTS-Thread existiert")
SPEAKING = threading.Event()

# Stabiler Audio-Ausgabe-Modus (über gesamten Antwort-Turn inkl. Chunk-Lücken).
AUDIO_OUT_ACTIVE = threading.Event()
AUDIO_OUT_LAST_ACTIVE_TS: float = 0.0

# Optional: PID des aktuell laufenden pw-play Prozesses (nur informativ)
CURRENT_PLAY_PID: Optional[int] = None

# Timestamp wann Playback geendet hat.
# Wird für Echo-Guard verwendet: kurz nach Playback-Ende sind Echo-Fehltriggers wahrscheinlicher.
LAST_PLAY_END_TS: float = 0.0

AUDIO_ACTIVITY_GRACE_SEC = 0.4
STOP_ACTION_DEBOUNCE_SEC = 1.0
STOP_ACTION_LOCK = threading.Lock()
LAST_STOP_ACTION_TS: float = 0.0


def is_audio_output_active() -> bool:
    """Robuste Audio-Aktivitätsprüfung mit Failsafe + Grace."""
    if AUDIO_OUT_ACTIVE.is_set() or SPEAKING.is_set() or (CURRENT_PLAY_PID is not None):
        return True
    return (time.time() - LAST_PLAY_END_TS) < AUDIO_ACTIVITY_GRACE_SEC


def request_stop_output(tts_mgr: "TTSManager", reason: str, debug: bool) -> bool:
    """Debounced Stop, damit Stop-Routinen nicht mehrfach parallel laufen."""
    global LAST_STOP_ACTION_TS
    now = time.time()
    with STOP_ACTION_LOCK:
        if (now - LAST_STOP_ACTION_TS) < STOP_ACTION_DEBOUNCE_SEC:
            if debug:
                log(f"Stop debounced: reason={reason}", debug)
            return False
        LAST_STOP_ACTION_TS = now
    tts_mgr.stop(reason=reason)
    return True

STOP_COMMANDS: set = set()
DEBUG_ENABLED = False

# Serialisiert Mikrofonzugriffe (Main-VAD vs. Barge-in), um pw-record Konkurrenz zu vermeiden.
MIC_RECORD_LOCK = threading.Lock()

# Optionales Timing-Trace pro Turn
TRACE_TIMINGS = False
TRACE_LOCK = threading.Lock()
class TimingAccumulator:
    """Akkumulierte Timing-Werte pro Turn."""

    def __init__(self):
        self.tts_total_sec = 0.0
        self.play_total_sec = 0.0
        self.enqueued_chunks = 0
        self.finished_chunks = 0
        self.closed = False
        self.cancelled = False
        self.done_event = threading.Event()


TRACE_ACCUM: Dict[str, TimingAccumulator] = {}


def trace_begin_turn(turn_id: str) -> TimingAccumulator:
    """Initialisiert Timing-Akkus für einen Turn."""
    acc = TimingAccumulator()
    with TRACE_LOCK:
        TRACE_ACCUM[turn_id] = acc
    return acc


def trace_add(kind: str, turn_id: Optional[str], dt: float) -> None:
    """Addiert dt (Sekunden) auf einen Turn-Akku."""
    if not TRACE_TIMINGS or not turn_id:
        return
    with TRACE_LOCK:
        d = TRACE_ACCUM.get(turn_id)
        if d is None:
            return
        if kind == "tts":
            inc = max(0.0, float(dt))
            d.tts_total_sec += inc
            if TRACE_TIMINGS:
                log(f"TRACE tts_add turn={turn_id} +{inc:.2f}s total={d.tts_total_sec:.2f}s", debug=True)
        elif kind == "play":
            inc = max(0.0, float(dt))
            d.play_total_sec += inc
            if TRACE_TIMINGS:
                log(f"TRACE play_add turn={turn_id} +{inc:.2f}s total={d.play_total_sec:.2f}s", debug=True)




def trace_note_enqueued(turn_id: Optional[str]) -> None:
    """Markiert einen neu enqueueten Chunk für den Turn."""
    if not TRACE_TIMINGS or not turn_id:
        return
    with TRACE_LOCK:
        d = TRACE_ACCUM.get(turn_id)
        if d is None:
            return
        d.enqueued_chunks += 1


def _trace_maybe_done(turn_id: str, d: TimingAccumulator) -> None:
    if d.done_event.is_set():
        return
    if d.cancelled or (d.closed and d.finished_chunks >= d.enqueued_chunks):
        d.done_event.set()
        log(f"TRACE tts_turn_done turn={turn_id}", debug=TRACE_TIMINGS)


def trace_mark_chunk_done(turn_id: Optional[str]) -> None:
    """Markiert einen abgearbeiteten Chunk (inkl. Fehler/Skip) für den Turn."""
    if not TRACE_TIMINGS or not turn_id:
        return
    with TRACE_LOCK:
        d = TRACE_ACCUM.get(turn_id)
        if d is None:
            return
        d.finished_chunks += 1
        _trace_maybe_done(turn_id, d)


def trace_close_turn(turn_id: Optional[str]) -> None:
    """Signalisiert: es kommen keine weiteren Chunks mehr für diesen Turn."""
    if not TRACE_TIMINGS or not turn_id:
        return
    with TRACE_LOCK:
        d = TRACE_ACCUM.get(turn_id)
        if d is None:
            return
        d.closed = True
        _trace_maybe_done(turn_id, d)


def trace_cancel_turn(turn_id: Optional[str]) -> None:
    """Signalisiert Turn-Abbruch; partial sums werden freigegeben."""
    if not TRACE_TIMINGS or not turn_id:
        return
    with TRACE_LOCK:
        d = TRACE_ACCUM.get(turn_id)
        if d is None:
            return
        d.cancelled = True
        _trace_maybe_done(turn_id, d)


def trace_wait_done(turn_id: Optional[str], timeout_sec: float) -> None:
    """Wartet kurz auf Turn-Fertigsignal aus TTS-Worker."""
    if not TRACE_TIMINGS or not turn_id:
        return
    with TRACE_LOCK:
        d = TRACE_ACCUM.get(turn_id)
    if d is None:
        return
    d.done_event.wait(timeout=max(0.0, timeout_sec))

def trace_pop_turn(turn_id: str) -> Tuple[float, float]:
    """Liest+entfernt Timing-Akkus eines Turns."""
    if not TRACE_TIMINGS:
        return 0.0, 0.0
    with TRACE_LOCK:
        d = TRACE_ACCUM.pop(turn_id, None)
    if not d:
        return 0.0, 0.0
    return d.tts_total_sec, d.play_total_sec


def fmt_timing(v: Optional[float]) -> str:
    """Formatierer für Timing-Logwerte."""
    return "None" if v is None else f"{v:.2f}s"


def fmt_seconds(v: float) -> str:
    """Formatierer für immer numerische Sekundenwerte."""
    return f"{max(0.0, float(v)):.2f}s"


# Logfile & Audio-Out Verzeichnis
LOG_FILE = Path("/data/mia/logs/talk/talk.log")
AUDIO_OUT_DIR = Path("/data/mia/data/audio/out")

# Whisper.cpp braucht LD_LIBRARY_PATH für CUDA/ggml libs (dein Build-Pfad).
WHISPER_LD_PATH = (
    "/data/mia/hear/whisper.cpp/build-cuda/src:"
    "/data/mia/hear/whisper.cpp/build-cuda/ggml/src:"
    "/data/mia/hear/whisper.cpp/build-cuda/ggml/src/ggml-cuda"
)


def on_signal(sig, frame):
    """
    Signalhandler (SIGINT/SIGTERM):
    - STOP setzt den globalen Exit-Flag
    - CANCEL_TTS setzt Abbruch für laufende Audioausgabe
    - SPEAKING.clear() sorgt dafür, dass wir nicht "sprechend" hängen bleiben
    """
    global STOP
    STOP = True
    CANCEL_TTS.set()
    SPEAKING.clear()


# Signals aktivieren (Ctrl+C / Stop)
signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)


def ensure_dir(p: Path) -> None:
    """Erstellt ein Verzeichnis inkl. parents, wenn es nicht existiert."""
    p.mkdir(parents=True, exist_ok=True)


def env_str(name: str, default: str = "") -> str:
    """
    Liest Environment Variable als String.
    - Strippt whitespace
    - Wenn leer/nicht gesetzt -> default
    """
    v = os.environ.get(name, "")
    v = v.strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    """env_str -> int mit Fallback default."""
    try:
        return int(env_str(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    """env_str -> float mit Fallback default."""
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    """
    Liest Env als bool.
    Akzeptiert: 1/true/yes/y/on
    """
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _csv_set(env_name: str, default: str) -> set[str]:
    """CSV aus Env lesen, normalisieren und als Set zurückgeben."""
    raw = env_str(env_name, default)
    return {normalize_cmd(x) for x in raw.split(",") if normalize_cmd(x)}


def _csv_list(env_name: str, default: str) -> list[str]:
    """CSV aus Env lesen, normalisieren und als Liste (geordnet) zurückgeben."""
    raw = env_str(env_name, default)
    return [normalize_cmd(x) for x in raw.split(",") if normalize_cmd(x)]


def load_local_env_file(path: Path, debug: bool = False) -> None:
    """Lädt einfache KEY=VALUE Zeilen aus einer env-Datei in os.environ (nur wenn KEY noch nicht gesetzt)."""
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            if not key:
                continue
            val = v.strip()
            if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                val = val[1:-1]
            os.environ.setdefault(key, val)
    except Exception as e:
        if debug:
            log(f"Env load warning: {path} ({e})", debug)

def now_iso() -> str:
    """Zeitstempel (lokal) im ISO-ish Format für Logs."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def log(msg: str, debug: bool = False, also_stdout: bool = True) -> None:
    """
    Logging:
    - debug=True wird auf stdout nur bei aktivem DEBUG_ENABLED ausgegeben
    - Datei-Logging bleibt vollständig (auch Debug), um Diagnose nicht zu verlieren
    """
    should_stdout = bool(also_stdout and ((not debug) or DEBUG_ENABLED))
    if should_stdout:
        print(msg, flush=True)
    try:
        ensure_dir(LOG_FILE.parent)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] {msg}\n")
    except Exception:
        # Logging darf nicht den Prozess crashen
        pass


def norm_text(s: str) -> str:
    """
    Normalisiert Text:
    - lower
    - trim
    - mehrfach whitespace -> single spaces
    """
    return " ".join((s or "").strip().lower().split())


def normalize_cmd(text: str) -> str:
    """Normalisiert Kommandotext robust gegen Satzzeichen."""
    t = (text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def contains_any(text: str, csv_words: str) -> bool:
    """
    Prüft, ob einer der CSV-Phrasen in text vorkommt.
    - text & words normalisiert
    - substring match (w in t)
    """
    t = normalize_cmd(text)
    words = [normalize_cmd(w) for w in csv_words.split(",") if w.strip()]
    return any(w and (w in t) for w in words)


def strip_wake_prefix(text: str, wake_csv: str) -> str:
    """
    Entfernt ein Wakeword am Anfang:
    z.B. "Hey Mia, wie geht's?" -> "wie geht's?"
    Arbeitet grob token-basiert mit raw split().
    """
    raw = text.strip()
    t = norm_text(raw)
    for w in [norm_text(x) for x in wake_csv.split(",") if x.strip()]:
        if w and t.startswith(w):
            wtoks = w.split()
            toks = raw.split()
            if len(toks) >= len(wtoks):
                return " ".join(toks[len(wtoks):]).strip(" ,.:;!?")
    return raw


# ----------------- Session Handling -----------------

def session_path_default() -> Path:
    """Default Session Datei (falls kein 'new' Modus)."""
    return Path("/data/mia/memory/sessions/talk_default.json")


def load_session(path: Path) -> Dict[str, Any]:
    """
    Lädt Session JSON:
    Erwartung:
      {"messages":[...], "created_at":..., "updated_at":...}

    Wenn Datei kaputt/fehlt -> leere Session.
    """
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("messages"), list):
                return data
        except Exception:
            pass
    return {"created_at": now_iso(), "updated_at": now_iso(), "messages": []}


def save_session(path: Path, sess: Dict[str, Any]) -> None:
    """
    Speichert Session atomar:
    - schreibt tmp
    - ersetzt original
    """
    sess["updated_at"] = now_iso()
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(sess, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def new_session_path() -> Path:
    """Erzeugt neuen Dateinamen talk_YYYYMMDD_HHMMSS_xxxxxx.json"""
    base = Path("/data/mia/memory/sessions")
    ensure_dir(base)
    return base / f"talk_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"


# ----------------- Shell Helpers -----------------

def safe_run(cmd: List[str], timeout_sec: float, capture: bool = True) -> Tuple[int, str, str]:
    """
    Führt subprocess.run aus mit Timeout und optional capture_output.
    Gibt (rc, stdout, stderr) zurück.
    """
    try:
        r = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout_sec,
        )
        return r.returncode, (r.stdout or ""), (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "", f"TIMEOUT after {timeout_sec}s: {' '.join(cmd)}"
    except Exception as e:
        return 125, "", f"ERROR running {' '.join(cmd)}: {e}"


def stop_playback(debug: bool) -> None:
    """
    Stoppt Audioausgabe:
    - pkill pw-play (nur der User)
    - setzt LAST_PLAY_END_TS für Echo-Guard
    - SPEAKING.clear() (damit wir nicht im Speaking-Modus hängen)
    """
    global LAST_PLAY_END_TS
    subprocess.run(["bash", "-lc", "pkill -u $USER -x pw-play 2>/dev/null || true"], check=False)
    LAST_PLAY_END_TS = time.time()
    SPEAKING.clear()
    if debug:
        log("Playback: stopped (pkill pw-play)", debug)


def wpctl_has_node_id(node_id: str) -> bool:
    """
    Prüft, ob ein PipeWire Node/Sink ID in 'wpctl status' auftaucht.
    """
    rc, out, _ = safe_run(["wpctl", "status"], timeout_sec=2.0, capture=True)
    if rc != 0:
        return False
    return f" {node_id}. " in out or f" {node_id}." in out


def maybe_set_default_sink(debug: bool) -> None:
    """
    Optional setzt Default Audio Sink über wpctl set-default.
    - gesteuert über MIA_AUDIO_SINK_ID
    - wenn ID fehlt oder nicht existiert -> keine Änderung
    """
    sink_id = env_str("MIA_AUDIO_SINK_ID", "")
    if not sink_id:
        log("Audio: keeping current default sink.", debug)
        return
    if not wpctl_has_node_id(sink_id):
        log(f"Audio: sink id '{sink_id}' not found OR wpctl timed out -> keeping current default.", debug)
        return
    if debug:
        log(f"Audio: setting default sink -> {sink_id}", debug)
    safe_run(["wpctl", "set-default", sink_id], timeout_sec=2.0, capture=True)


def _have_cmd(cmd: str) -> bool:
    """Prüft, ob command im PATH existiert."""
    try:
        r = subprocess.run(["bash", "-lc", f"command -v {cmd}"], capture_output=True, text=True)
        return r.returncode == 0 and (r.stdout or "").strip() != ""
    except Exception:
        return False


def _which(cmd: str) -> Optional[str]:
    """Wie 'which': gibt Pfad zurück oder None."""
    rc, out, _ = safe_run(["bash", "-lc", f"command -v {cmd}"], timeout_sec=2.0, capture=True)
    if rc != 0:
        return None
    out = (out or "").strip()
    return out if out else None


# ----------------- STOP detection (robust) -----------------

def tokenize(text: str) -> List[str]:
    """
    Tokenizer für Stop-Erkennung:
    - lower/normalize
    - ersetzt Satzzeichen durch spaces
    - split -> tokens
    """
    t = norm_text(text)
    for ch in [",", ".", "!", "?", ":", ";", "—", "-", "(", ")", "[", "]", "{", "}", "\"", "'"]:
        t = t.replace(ch, " ")
    toks = [x for x in t.split() if x]
    return toks


def is_stop_command(text: str, stop_set: set) -> bool:
    """
    Robuste Stop/Barge-in Erkennung.

    Erkennt nur definierte Stop-Kommandos aus übergebenem stop_set.
    """
    nt = norm_text(text)
    toks = tokenize(nt)

    candidates = set(toks)
    for i in range(len(toks) - 1):
        candidates.add(f"{toks[i]} {toks[i + 1]}")
    return any(cmd in candidates for cmd in stop_set)


# ----------------- VAD recorder -----------------

def _record_chunk_wav(tmp_wav: Path, rate: int, seconds: float, debug: bool) -> bool:
    """
    Nimmt einen kurzen Audio-Chunk via pw-record auf:
    - pw-record schreibt WAV Datei
    - timeout sorgt für hartes Ende (falls pw-record hängt)
    """
    ensure_dir(tmp_wav.parent)
    cmd = [
        "timeout", "-k", "1s", f"{seconds:.2f}s",
        "pw-record",
        "--rate", str(rate),
        "--channels", "1",
        "--format", "s16",
        str(tmp_wav),
    ]
    with MIC_RECORD_LOCK:
        r = subprocess.run(cmd, capture_output=True, text=True)

    # wenn WAV existiert und > ~1KB -> ok
    if tmp_wav.exists() and tmp_wav.stat().st_size > 1000:
        return True

    # Debug: letzte stdout/stderr tail ausgeben
    if debug:
        err = (r.stderr or "").strip()
        out = (r.stdout or "").strip()
        if err or out:
            log(f"Audio chunk record failed: rc={r.returncode} stdout='{out[-200:]}' stderr='{err[-200:]}'", debug)
    return False


def _read_wav_pcm16_mono(path: Path) -> bytes:
    """
    Liest WAV und liefert PCM16 mono bytes.
    - Wenn WAV stereo, wird auf mono gedownmixed (einfach Kanal 0 extrahieren)
    """
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

        # perfektes Format: mono, 16-bit
        if ch == 1 and sw == 2:
            return frames

        # fallback: mehrkanalig -> nur jeden ch-ten sample nehmen (Kanal 0)
        if ch > 1:
            import array
            a = array.array("h", frames)
            a = a[0::ch]
            return a.tobytes()

        return frames


def record_vad_to_wav(
    output_wav: Path,
    rate: int,
    frame_ms: int,
    vad_mode: int,
    silence_limit_sec: float,
    min_speech_sec: float,
    trigger_ratio: float,
    ring_ms: int,
    max_sec: int,
    rms_threshold: int,
    debug: bool,
) -> bool:
    """
    Haupt-VAD Recorder:

    Ablauf:
    - Nimmt wiederholt kurze Chunks (z.B. 0.40s) auf
    - Zerlegt in Frames (frame_ms, z.B. 30ms)
    - Pro Frame:
        - RMS prüfen (Lautstärke)
        - wenn laut genug -> webrtcvad.is_speech
    - Ringbuffer (ring_ms) sammelt Frames vor Trigger:
        - wenn Anteil speech >= trigger_ratio -> "speech start"
    - Ab Trigger:
        - Frames sammeln bis silence länger als silence_limit_sec -> "speech end"
    - Wenn Aufnahme < min_speech_sec -> ignorieren
    - Schreibt am Ende output_wav (mono, 16kHz, PCM16)
    """
    import webrtcvad  # type: ignore

    if not _have_cmd("pw-record"):
        log("FATAL: pw-record not found. Install pipewire-utils.", debug)
        return False

    if audioop is None:
        log("FATAL: audioop missing/unavailable.", debug)
        return False

    vad = webrtcvad.Vad(vad_mode)

    # Bytes pro Frame (PCM16 -> 2 bytes pro sample)
    frame_size_bytes = int(rate * frame_ms / 1000) * 2

    # Ringbuffer Länge in Frames
    ring_maxlen = max(1, int(ring_ms / frame_ms))
    ring_buffer = collections.deque(maxlen=ring_maxlen)

    frames: List[bytes] = []
    triggered = False
    silence_counter = 0

    t0 = time.time()

    # Chunk-Länge für pw-record (kann über Env angepasst werden)
    chunk_sec = float(env_float("MIA_CHUNK_SEC", 0.40))

    # Temporäre Chunk-Dateien
    tmp_dir = Path("/tmp/mia_talk_chunks")
    ensure_dir(tmp_dir)

    while not STOP:
        # Globale max_sec: schützt gegen "ewig warten"
        if (time.time() - t0) > max_sec:
            if debug:
                log("VAD: max_sec reached", debug)
            break

        tmp_wav = tmp_dir / f"chunk_{uuid.uuid4().hex}.wav"
        ok = _record_chunk_wav(tmp_wav, rate=rate, seconds=chunk_sec, debug=debug)
        if not ok:
            time.sleep(0.05)
            continue

        pcm = _read_wav_pcm16_mono(tmp_wav)

        # Chunk-Datei löschen (wir haben bytes im Speicher)
        try:
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            pass

        idx = 0
        while idx + frame_size_bytes <= len(pcm):
            chunk = pcm[idx: idx + frame_size_bytes]
            idx += frame_size_bytes

            # RMS: verhindert VAD false positives bei sehr leisen Signalen
            rms = audioop.rms(chunk, 2)
            if rms < rms_threshold:
                is_speech = False
            else:
                is_speech = vad.is_speech(chunk, rate)

            if not triggered:
                # vor Trigger -> Ringbuffer füllen
                ring_buffer.append((chunk, is_speech))
                num_voiced = sum(1 for _, sp in ring_buffer if sp)

                # wenn genug voiced frames -> trigger
                if num_voiced >= trigger_ratio * ring_buffer.maxlen:
                    triggered = True
                    frames.extend([c for c, _ in ring_buffer])  # pre-roll übernehmen
                    ring_buffer.clear()
                    silence_counter = 0
                    if debug:
                        log("VAD: speech start", debug)
            else:
                # nach Trigger -> sammeln und Silence zählen
                frames.append(chunk)
                if not is_speech:
                    silence_counter += 1
                else:
                    silence_counter = 0

                # genug Stille -> Ende
                if (silence_counter * frame_ms / 1000.0) > silence_limit_sec:
                    if debug:
                        log("VAD: speech end (silence)", debug)
                    idx = len(pcm)
                    break

        # Wenn noch nicht getriggert -> weiter aufnehmen
        if not triggered:
            continue

        # Wenn getriggert und Silence überschritten -> raus
        if triggered and (silence_counter * frame_ms / 1000.0) > silence_limit_sec:
            break

    if not frames:
        return False

    # Dauer (in Sekunden) grob berechnen
    speech_sec = (len(frames) * frame_size_bytes) / (rate * 2.0)
    if speech_sec < min_speech_sec:
        if debug:
            log(f"VAD: too short ({speech_sec:.2f}s) ignored", debug)
        return False

    # WAV schreiben
    ensure_dir(output_wav.parent)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    return True


# ----------------- Whisper -----------------

def _run_whisper(cmd: List[str], env: Dict[str, str]) -> Tuple[int, str, str, float]:
    """
    Führt whisper-cli aus und misst Laufzeit.
    Returns: (rc, stdout, stderr, seconds)
    """
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    return r.returncode, (r.stdout or ""), (r.stderr or ""), dt


def whisper_transcribe(wav_path: Path, model_path: str, whisper_cli: str, lang: str, debug: bool) -> str:
    """
    Whisper.cpp Transkription:
    - setzt LD_LIBRARY_PATH für CUDA libs
    - versucht GPU first (sofern nicht MIA_WHISPER_FORCE_CPU)
    - falls GPU OOM -> retry CPU (-ng)
    - nimmt letzte stdout Zeile als Text (typisch whisper-cli output)
    """
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = WHISPER_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    base = [whisper_cli, "-m", model_path, "-f", str(wav_path), "-l", lang, "-nt"]

    force_cpu = env_bool("MIA_WHISPER_FORCE_CPU", False)
    tried_cpu = False

    if not force_cpu:
        rc, out, err, dt = _run_whisper(base, env)
        if rc == 0:
            lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
            text = lines[-1] if lines else ""
            if debug:
                log(f"Whisper ({Path(model_path).name}) in {dt:.2f}s -> '{text}'", debug)
            return text

        blob = (out + "\n" + err).lower()
        if ("cuda error" in blob and "out of memory" in blob) or ("ggml_cuda_init" in blob and "out of memory" in blob):
            tried_cpu = True
            if debug:
                log("Whisper GPU OOM -> retry with CPU (-ng)", debug)
        else:
            raise RuntimeError(f"whisper-cli failed rc={rc}\nstdout={out[-2000:]}\nstderr={err[-2000:]}")

    # CPU Modus
    cmd_cpu = base + ["-ng"]
    rc2, out2, err2, dt2 = _run_whisper(cmd_cpu, env)
    if rc2 != 0:
        raise RuntimeError(f"whisper-cli failed rc={rc2}\nstdout={out2[-2000:]}\nstderr={err2[-2000:]}")
    lines2 = [ln.strip() for ln in (out2 or "").splitlines() if ln.strip()]
    text2 = lines2[-1] if lines2 else ""
    if debug:
        mode = "CPU(-ng)" if (force_cpu or tried_cpu) else "CPU"
        log(f"Whisper {mode} ({Path(model_path).name}) in {dt2:.2f}s -> '{text2}'", debug)
    return text2


def is_noise_transcript(t: str) -> bool:
    """
    Filtert typische "kein Inhalt" Transkripte:
    - leer
    - Musik Marker
    - ... / …
    """
    x = (t or "").strip().lower()
    if not x:
        return True
    if x in ("[musik]", "* musik *", "musik", "[music]", "* music *", "...", "…"):
        return True
    return False


# ----------------- Ollama -----------------

def ollama_warmup_http(host: str, model: str, keepalive: str) -> Tuple[float, str]:
    """Echter Modell-Warmup über /api/chat (stream:false)."""
    if requests is None:
        raise RuntimeError("Python 'requests' not installed (needed for warmup HTTP).")

    url = host.rstrip("/") + "/api/chat"
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Sag nur: OK."}],
        "stream": False,
        "keep_alive": keepalive,
        "options": {
            "num_predict": 8,
            "temperature": env_float("MIA_TEMPERATURE", 0.2),
            "top_p": env_float("MIA_TOP_P", 0.9),
            "repeat_penalty": env_float("MIA_REPEAT_PENALTY", 1.1),
        },
    }
    r = requests.post(url, json=payload, timeout=45)
    if r.status_code >= 400:
        raise RuntimeError(f"warmup http failed {r.status_code}: {(r.text or '')[:300]}")
    txt = ""
    try:
        obj = r.json()
        msg = obj.get("message") if isinstance(obj, dict) else None
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str):
                txt = c.strip()
    except Exception:
        txt = (r.text or "").strip()
    return time.perf_counter() - t0, txt


def ollama_chat_http_stream(
    host: str,
    model: str,
    keepalive: str,
    messages: List[Dict[str, str]],
    debug: bool,
    on_delta: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Streaming Chat über Ollama /api/chat

    - POST {model, messages, stream:true, keep_alive, options:{...}}
    - iter_lines liefert JSON Zeilen (message.content deltas)
    - sammelt alles zu full text
    - optional: ruft on_delta(delta) pro content-Stück auf (für TTS-Streaming)
    - stall_timeout: wenn lange keine neuen Zeilen kommen -> abbrechen
    """
    if requests is None:
        raise RuntimeError("Python 'requests' not installed (needed for Ollama /api/chat).")

    url = host.rstrip("/") + "/api/chat"
    num_predict = env_int("MIA_NUM_PREDICT", 1200)
    temperature = env_float("MIA_TEMPERATURE", 0.2)
    top_p = env_float("MIA_TOP_P", 0.9)
    repeat_penalty = env_float("MIA_REPEAT_PENALTY", 1.1)

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": keepalive,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        },
    }

    if debug:
        log(f"Ollama /api/chat model={model} keep_alive={keepalive} num_predict={num_predict}", debug)

    full: List[str] = []
    chunk_count = 0

    # Wenn Ollama "hängt" (keine Daten) -> Abbruch nach X Sekunden
    stall_timeout_sec = env_int("MIA_OLLAMA_STALL_TIMEOUT_SEC", 30)
    last_line_ts = time.time()

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        if r.status_code >= 400:
            body = (r.text or "").strip()
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {body[:2000]}")

        for line in r.iter_lines(decode_unicode=True, chunk_size=1):
            now = time.time()
            if STOP or CANCEL_TTS.is_set():
                break

            # Keepalive: wenn line leer, prüfen ob stall
            if not line:
                if (now - last_line_ts) > stall_timeout_sec:
                    if debug:
                        log("Ollama stream stall timeout reached -> stopping read", debug)
                    break
                continue

            last_line_ts = now

            try:
                obj = json.loads(line)
            except Exception:
                continue

            msg = obj.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content:
                    full.append(content)
                    chunk_count += 1
                    if on_delta and not CANCEL_TTS.is_set() and not STOP:
                        on_delta(content)

            if obj.get("done") is True:
                break

    text = "".join(full).strip()
    if debug:
        log(f"Ollama stream done: chunks={chunk_count} chars={len(text)}", debug)
    return text


# ----------------- TTS + playback -----------------

def clean_for_tts(text: str) -> str:
    """
    Sanitizer für TTS:
    - entfernt Markdown Marker (** __ `)
    - normalisiert Zeilen
    - filtert auf erlaubte Zeichenmenge (verhindert "komische" Symbole)
    - behält \n als leichte Pausen
    """
    t = (text or "").strip()
    for pat in ("**", "__", "`"):
        t = t.replace(pat, "")
    t = t.replace("\r", "\n")
    t = "\n".join([ln.strip() for ln in t.split("\n") if ln.strip()])

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789äöüÄÖÜß .,;:!?-()'\"%/+=@\n")
    t = "".join(ch for ch in t if ch in allowed)

    t = " ".join(t.replace("\n", " \n ").split())
    t = t.replace(" \n ", "\n").strip()
    return t


def piper_speak(text: str, debug: bool, turn_id: Optional[str] = None) -> Path:
    global AUDIO_OUT_LAST_ACTIVE_TS
    """
    Piper TTS:
    - erzeugt WAV in AUDIO_OUT_DIR
    - setzt SPEAKING während Synth, weil in dieser Zeit "Mia spricht" (für UI/ignore logic)
    """
    ensure_dir(AUDIO_OUT_DIR)
    out_wav = AUDIO_OUT_DIR / f"{uuid.uuid4()}.wav"

    piper_bin = _which("piper-tts") or _which("piper") or "piper-tts"
    model = env_str("PIPER_MODEL", "")

    if not model or not os.path.exists(model):
        raise RuntimeError(f"Piper model not found: '{model}'")

    cmd = [piper_bin, "--model", model, "--output_file", str(out_wav)]
    if debug:
        log(f"Piper: bin={piper_bin} model={model}", debug)

    # Ab hier gilt: sie spricht / ist busy
    SPEAKING.set()
    AUDIO_OUT_LAST_ACTIVE_TS = time.time()

    t0 = time.perf_counter()
    r = subprocess.run(cmd, input=text, text=True, capture_output=True)
    dt = time.perf_counter() - t0
    trace_add("tts", turn_id, dt)

    if r.returncode != 0 or not out_wav.exists():
        SPEAKING.clear()
        raise RuntimeError(
            f"piper failed rc={r.returncode}\n"
            f"stdout={(r.stdout or '').strip()[-2000:]}\n"
            f"stderr={(r.stderr or '').strip()[-2000:]}\n"
        )

    if debug:
        log(f"Piper: synth time {dt:.2f}s -> {out_wav}", debug)
    return out_wav


def play_wav(wav: Path, debug: bool, turn_id: Optional[str] = None) -> None:
    """
    Playback via pw-play:
    - setzt SPEAKING für Dauer des Playback
    - checkt STOP/CANCEL_TTS in Schleife -> Barge-in kann abbrechen
    - aktualisiert LAST_PLAY_END_TS am Ende
    """
    global CURRENT_PLAY_PID, LAST_PLAY_END_TS, AUDIO_OUT_LAST_ACTIVE_TS
    if STOP or CANCEL_TTS.is_set():
        SPEAKING.clear()
        return

    cmd = ["pw-play", str(wav)]
    if debug:
        log(f"Playback: pw-play {wav}", debug)

    # Playback aktiv -> speaking
    SPEAKING.set()

    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    CURRENT_PLAY_PID = p.pid
    t0 = time.perf_counter()
    try:
        while p.poll() is None:
            if STOP or CANCEL_TTS.is_set():
                try:
                    p.terminate()
                except Exception:
                    pass
                stop_playback(debug)
                return
            time.sleep(0.02)
    finally:
        trace_add("play", turn_id, time.perf_counter() - t0)
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
        CURRENT_PLAY_PID = None
        LAST_PLAY_END_TS = time.time()
        AUDIO_OUT_LAST_ACTIVE_TS = LAST_PLAY_END_TS
        SPEAKING.clear()


class TTSManager:
    """
    TTSManager:
    - hat eine Queue für Textstücke
    - Worker-Thread wartet auf Queue
    - Für jedes Stück:
        - clean_for_tts
        - piper_speak
        - play_wav
    - stop(): bricht Playback ab + beendet Worker
    """

    def __init__(self, debug: bool):
        self.debug = debug
        self.q: "queue.Queue[Union[Tuple[str, Optional[str]], None]]" = queue.Queue(maxsize=64)
        self.th: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_queue_warn_ts = 0.0

    def start(self) -> None:
        """Startet Worker-Thread (falls nicht läuft)."""
        with self._lock:
            if self.th and self.th.is_alive():
                return
            self.th = threading.Thread(target=self._worker, daemon=True)
            self.th.start()

    def reset_for_new_turn(self) -> None:
        """Stellt TTS-Worker und Cancel-Status für einen neuen Turn sicher."""
        CANCEL_TTS.clear()
        self.start()

    def enqueue(self, text: str, turn_id: Optional[str] = None) -> None:
        """Legt Text in Queue (wenn nicht STOP/CANCEL)."""
        if STOP or CANCEL_TTS.is_set():
            return
        try:
            self.q.put((text, turn_id), timeout=1.0)
            trace_note_enqueued(turn_id)
        except Exception:
            now = time.time()
            if (now - self._last_queue_warn_ts) >= 0.33:
                self._last_queue_warn_ts = now
                log("TTS queue full -> dropping chunk", debug=True)
            if turn_id:
                trace_mark_chunk_done(turn_id)

    def stop(self, reason: str = "stop") -> None:
        """
        Stoppt TTS sofort:
        - CANCEL_TTS setzen
        - pw-play killen
        - None in Queue senden -> Worker beendet
        """
        CANCEL_TTS.set()
        stop_playback(self.debug)
        with self._lock:
            try:
                self.q.put(None, timeout=0.2)
            except Exception:
                pass
            if self.th:
                self.th.join(timeout=2.0)
            self.th = None
        SPEAKING.clear()
        AUDIO_OUT_ACTIVE.clear()
        if self.debug:
            log(f"TTS: stopped ({reason})", self.debug)

    def _drain_queue_fast(self) -> None:
        """Leert Queue schnell (ohne zu blockieren)."""
        try:
            while True:
                x = self.q.get_nowait()
                if x is None:
                    break
        except Exception:
            pass

    def _worker(self) -> None:
        """Worker-Schleife: nimmt Texte aus Queue und spricht sie."""
        global AUDIO_OUT_LAST_ACTIVE_TS
        try:
            while not STOP:
                item = self.q.get()
                if item is None:
                    break
                if CANCEL_TTS.is_set():
                    self._drain_queue_fast()
                    break

                txt_raw, item_turn_id = item
                txt = clean_for_tts(txt_raw)
                if not txt:
                    trace_mark_chunk_done(item_turn_id)
                    continue

                # stabiler Sprech-Modus über Chunk-Grenzen hinweg
                AUDIO_OUT_ACTIVE.set()
                AUDIO_OUT_LAST_ACTIVE_TS = time.time()

                wav: Optional[Path] = None
                try:
                    wav = piper_speak(txt, self.debug, turn_id=item_turn_id)
                    if STOP or CANCEL_TTS.is_set():
                        if self.debug:
                            log("TTS/Playback skipped: cancel/stop before playback", self.debug)
                        SPEAKING.clear()
                        continue
                    play_wav(wav, self.debug, turn_id=item_turn_id)
                except Exception as e:
                    SPEAKING.clear()
                    log(f"TTS/Playback ERROR: {type(e).__name__}: {e}", self.debug)
                finally:
                    if wav is not None:
                        try:
                            wav.unlink(missing_ok=True)
                        except Exception:
                            pass
                    trace_mark_chunk_done(item_turn_id)
                    AUDIO_OUT_LAST_ACTIVE_TS = time.time()

                # Nur clearen wenn wirklich idle + kurze Grace-Phase gegen Flackern.
                while (not STOP) and (not CANCEL_TTS.is_set()) and self.q.empty() and (CURRENT_PLAY_PID is None) and (not SPEAKING.is_set()):
                    if (time.time() - AUDIO_OUT_LAST_ACTIVE_TS) > AUDIO_ACTIVITY_GRACE_SEC:
                        AUDIO_OUT_ACTIVE.clear()
                        break
                    time.sleep(0.05)
        finally:
            SPEAKING.clear()
            AUDIO_OUT_ACTIVE.clear()


class StreamChunker:
    """
    StreamChunker:
    Zweck: LLM streamt deltas -> wir wollen sinnvoll in Sätzen/Chunks TTS-en.
    - sammelt in Buffer
    - trennt bei \n
    - oder bei Satzendzeichen (. ! ?)
    - min_chars / max_chars steuern, ab wann ein Chunk rausgeht
    """

    def __init__(
        self,
        emit: Callable[[str], None],
        min_chars: int,
        max_chars: int,
        adaptive: bool = False,
        phase1_sec: float = 1.2,
        phase2_sec: float = 3.5,
        first_min_chars: int = 25,
        first_max_wait_sec: float = 0.8,
    ):
        self.emit = emit
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buf = ""
        self.adaptive = adaptive
        self.phase1_sec = max(0.1, phase1_sec)
        self.phase2_sec = max(self.phase1_sec, phase2_sec)
        self.t0 = time.perf_counter()
        self.first_min_chars = max(1, int(first_min_chars))
        self.first_max_wait_sec = max(0.05, float(first_max_wait_sec))
        self.first_token_ts: Optional[float] = None
        self.first_chunk_emitted = False

    def _emit(self, chunk: str) -> None:
        c = (chunk or "").strip()
        if not c:
            return
        self.emit(c)
        self.first_chunk_emitted = True

    def _limits(self) -> Tuple[int, int]:
        if not self.adaptive:
            return self.min_chars, self.max_chars
        dt = time.perf_counter() - self.t0
        if dt < self.phase1_sec:
            return max(20, int(self.min_chars * 0.45)), max(60, int(self.max_chars * 0.55))
        if dt < self.phase2_sec:
            return max(35, int(self.min_chars * 0.70)), max(90, int(self.max_chars * 0.80))
        return self.min_chars, self.max_chars

    def push(self, delta: str):
        """Nimmt neue Textdeltas auf und emittiert ggf. fertige Chunks."""
        if not delta or STOP or CANCEL_TTS.is_set():
            return
        if self.first_token_ts is None:
            self.first_token_ts = time.perf_counter()
        self.buf += delta

        # 1) harte Trennung bei Zeilenumbrüchen
        while "\n" in self.buf:
            left, right = self.buf.split("\n", 1)
            self._emit(left)
            self.buf = right

        # Fast-first: frühes erstes Chunk ohne Satzende
        if not self.first_chunk_emitted and self.buf.strip():
            if len(self.buf) >= self.first_min_chars:
                self._emit(self.buf)
                self.buf = ""
                return
            if self.first_token_ts and (time.perf_counter() - self.first_token_ts) >= self.first_max_wait_sec:
                self._emit(self.buf)
                self.buf = ""
                return

        # 2) chunking bei Satzzeichen, sobald min_chars erreicht
        while True:
            min_chars, max_chars = self._limits()
            if len(self.buf) < min_chars:
                break
            cut = self.buf[: max_chars]
            idx = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
            if idx == -1:
                # kein Satzende gefunden:
                # wenn Buffer viel zu groß -> am letzten Space trennen
                if len(self.buf) > max_chars:
                    sp = cut.rfind(" ")
                    if sp > 0:
                        self._emit(self.buf[:sp])
                        self.buf = self.buf[sp:].lstrip()
                        continue
                break
            emit_upto = idx + 1
            self._emit(self.buf[:emit_upto])
            self.buf = self.buf[emit_upto:].lstrip()

    def flush(self):
        """Gibt Restbuffer am Ende aus."""
        tail = self.buf.strip()
        if tail:
            self._emit(tail)
        self.buf = ""


# ----------------- echo guard -----------------

def _word_set(s: str) -> set:
    """
    Extrahiert "wichtige" Wörter:
    - normalisiert
    - filtert stop words (und/der/die/...)
    - filtert sehr kurze Tokens
    """
    s = norm_text(s)
    toks = [t for t in s.replace("'", " ").replace('"', " ").split() if t]
    stop = {"und", "oder", "der", "die", "das", "ein", "eine", "ist", "sind", "zu", "im", "in", "am", "an", "auf", "für"}
    return {t for t in toks if t not in stop and len(t) >= 3}


def looks_like_echo(user_t: str, recent_assistant: Deque[str], min_overlap: float) -> bool:
    """
    Echo-Heuristik:
    - Vergleicht user_text mit den letzten Assistant-Ausgaben (recent_assistant)
    - berechnet overlap = |uw ∩ aw| / |uw|
    - wenn overlap >= min_overlap -> wahrscheinlich Echo
    """
    ut = norm_text(user_t)
    if not ut:
        return True
    uw = _word_set(ut)
    if not uw:
        return False

    best = 0.0
    for a in reversed(recent_assistant):
        aw = _word_set(a)
        if not aw:
            continue
        inter = len(uw & aw)
        denom = max(1, len(uw))
        score = inter / denom
        if score > best:
            best = score

    return best >= min_overlap


def is_phantom_utterance(
    text: str,
    state: str,
    phantom_exact: set[str],
    phantom_contains: list[str],
    phantom_max_words: int,
    phantom_max_chars: int,
    phantom_year_filter: bool,
    confirm_allowed: set[str],
) -> bool:
    """Konfigurierbarer Phantom/Parasit-Filter für Transkripte."""
    t = normalize_cmd(text)
    if not t:
        return True

    if state == "AWAIT_CONFIRMATION":
        if t in confirm_allowed:
            return False
        return False

    if t in phantom_exact:
        return True

    word_count = len(t.split())
    is_short = (word_count <= max(1, int(phantom_max_words))) or (len(t) <= max(1, int(phantom_max_chars)))
    if is_short:
        if any(p and (p in t) for p in phantom_contains):
            return True
        if phantom_year_filter and re.search(r"\b(19\d{2}|20\d{2})\b", t):
            return True

    return False


def is_question_like(text: str) -> bool:
    """
    Wenn User-Text wie eine Frage aussieht, wollen wir ihn eher nicht als Echo wegwerfen.
    Kriterien:
    - enthält "?"
    - beginnt mit Fragewörtern oder "kannst du..."
    """
    t = (text or "").strip()
    if "?" in t:
        return True
    nt = norm_text(t)
    for w in ("ist", "sind", "warum", "wieso", "wie", "was", "wer", "kannst", "könntest", "erklär", "erklaer"):
        if nt.startswith(w + " ") or nt == w:
            return True
    return False


class BargeInMonitor:
    """Barge-in Scanner während SPEAKING (ein Loop, kein Doppel-pw-record)."""

    def __init__(
        self,
        rate: int,
        whisper_cli: str,
        whisper_lang: str,
        barge_model: str,
        window_sec: float,
        interval_sec: float,
        is_stop_match: Callable[[str], bool],
        on_hit: Callable[[str], None],
        debug: bool,
    ):
        self.rate = rate
        self.whisper_cli = whisper_cli
        self.whisper_lang = whisper_lang
        self.barge_model = barge_model
        self.window_sec = max(0.6, float(window_sec))
        self.interval_sec = max(0.15, float(interval_sec))
        self.is_stop_match = is_stop_match
        self.on_hit = on_hit
        self.debug = debug

        self._stop = threading.Event()
        self._th_scan: Optional[threading.Thread] = None
        self._last_hit_ts = 0.0
        self._last_err_log_ts = 0.0

    def start(self) -> None:
        self._th_scan = threading.Thread(target=self._scan_loop, daemon=True)
        self._th_scan.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th_scan:
            self._th_scan.join(timeout=1.5)

    def _scan_loop(self) -> None:
        """
        Scannt Stop-Phrasen nur während SPEAKING:
        - nimmt pro Scan genau ein kurzes WAV auf (window_sec)
        - nutzt dieses WAV direkt für tiny-whisper

        Dadurch entfällt der frühere Doppelpfad (capture-loop + scan-loop)
        und die I/O Last sinkt deutlich.
        """
        tmp_dir = Path("/tmp/mia_barge_scan")
        ensure_dir(tmp_dir)

        while not STOP and not self._stop.is_set():
            if not is_audio_output_active():
                time.sleep(self.interval_sec)
                continue

            wav = tmp_dir / f"barge_{uuid.uuid4().hex}.wav"
            try:
                ok = _record_chunk_wav(
                    wav,
                    rate=self.rate,
                    seconds=self.window_sec,
                    debug=False,
                )
                if not ok:
                    time.sleep(self.interval_sec)
                    continue

                txt = whisper_transcribe(
                    wav,
                    self.barge_model,
                    self.whisper_cli,
                    self.whisper_lang,
                    debug=False,
                ).strip()

                nt = norm_text(txt)
                if txt and self.is_stop_match(txt):
                    now = time.time()
                    if (now - self._last_hit_ts) > 0.8:
                        self._last_hit_ts = now
                        if self.debug:
                            log(f"BARGE HIT text='{txt}' norm='{nt}'", self.debug)
                        self.on_hit(txt)
            except Exception as e:
                now = time.time()
                if (now - self._last_err_log_ts) >= 5.0:
                    self._last_err_log_ts = now
                    log(
                        f"BARGE ERROR (rate-limited): {e} "
                        f"(model={Path(self.barge_model).name} window={self.window_sec:.2f}s interval={self.interval_sec:.2f}s)",
                        self.debug,
                    )
            finally:
                try:
                    wav.unlink(missing_ok=True)
                except Exception:
                    pass

            time.sleep(self.interval_sec)


# ----------------- main -----------------

def main():
    """
    Hauptloop:
    - Setup/Config lesen
    - Session laden
    - optional Warmup
    - Loop:
        - record_vad_to_wav
        - whisper_transcribe
        - stop/exit prüfen
        - speaking-ignores prüfen
        - wake/awake prüfen
        - echo guard prüfen
        - Ollama streamen
        - TTS chunks enqueuen
        - Session speichern
    """
    global LAST_PLAY_END_TS, TRACE_TIMINGS, STOP_COMMANDS, DEBUG_ENABLED

    # Optionales env-file laden (für direkten Python-Start ohne run.sh)
    load_local_env_file(Path(__file__).with_name("config.env"), debug=False)

    debug = env_bool("MIA_DEBUG", False)
    DEBUG_ENABLED = debug
    trace_env_raw = os.environ.get("MIA_TRACE_TIMINGS", "")
    TRACE_TIMINGS = env_bool("MIA_TRACE_TIMINGS", False)

    phantom_exact = _csv_set("MIA_PHANTOM_EXACT", "danke,vielen dank,dankeschön,ok,okay,ja")
    phantom_contains = _csv_list("MIA_PHANTOM_CONTAINS", "zdf,ard,untertitel,untertitelung")
    phantom_max_words = env_int("MIA_PHANTOM_MAX_WORDS", 6)
    phantom_max_chars = env_int("MIA_PHANTOM_MAX_CHARS", 40)
    phantom_year_filter = env_bool("MIA_PHANTOM_YEAR_FILTER", True)
    confirm_allowed = _csv_set("MIA_CONFIRM_ALLOWED", "ja,nein,ok,okay,stimmt,genau,weiter")

    log(f"TraceTimings: enabled={TRACE_TIMINGS} env={trace_env_raw if trace_env_raw != '' else '<unset>'}", debug)
    log(f"Using python: {sys.executable}", debug)
    log(f"Python version: {sys.version.split()[0]}", debug)

    # Sicherstellen, dass webrtcvad verfügbar ist
    try:
        import webrtcvad  # noqa: F401
    except Exception as e:
        log(f"FATAL: webrtcvad missing. pip install webrtcvad (err={e})", debug)
        sys.exit(2)

    # Optional Default sink setzen
    maybe_set_default_sink(debug)

    # Ollama Konfig
    ollama_host = env_str("MIA_OLLAMA_HOST", "http://127.0.0.1:11434")
    model = env_str("MIA_LLM", "mia-talk-72b:latest")
    keepalive = env_str("MIA_KEEPALIVE", "-1m")
    warmup = env_bool("MIA_WARMUP", True)

    # Whisper Konfig
    whisper_cli = env_str("MIA_WHISPER_CLI", "/data/mia/hear/whisper.cpp/build-cuda/bin/whisper-cli")
    whisper_model = env_str("MIA_WHISPER_MODEL", "/data/mia/hear/whisper.cpp/models/ggml-large-v3.bin")
    whisper_lang = env_str("MIA_WHISPER_LANG", "de")

    # Barge-in (always-on) via tiny Whisper + Ringbuffer
    barge_whisper_model = env_str("MIA_BARGE_WHISPER_MODEL", "/data/mia/hear/whisper.cpp/models/ggml-tiny.bin")
    barge_window_sec = env_float("MIA_BARGE_WINDOW_SEC", 1.10)
    barge_interval_sec = env_float("MIA_BARGE_INTERVAL_SEC", 0.35)

    # Audio/VAD Parameter (normaler Listening-Mode)
    rate = env_int("MIA_RATE", 16000)
    frame_ms = env_int("MIA_FRAME_MS", 30)
    vad_mode = env_int("MIA_VAD_MODE", 2)
    silence_limit = env_float("MIA_SILENCE_LIMIT_SEC", 0.9)
    min_speech = env_float("MIA_MIN_SPEECH_SEC", 0.35)
    trigger_ratio = env_float("MIA_TRIGGER_RATIO", 0.70)
    ring_ms = env_int("MIA_RING_MS", 240)
    max_sec = env_int("MIA_MAX_SEC", 15)
    rms_threshold = env_int("MIA_RMS_THRESHOLD", 420)

    # Wake/Exit
    require_wake = env_bool("MIA_REQUIRE_WAKE", True)
    wakewords = env_str("MIA_WAKEWORDS", "hey mia,mia")
    exitwords = env_str("MIA_EXITWORDS", "mia ende,mir ende")
    awake_timeout_sec = env_int("MIA_AWAKE_TIMEOUT_SEC", 180)

    # Stop/Barge-in
    stopwords_raw = env_str("MIA_STOPWORDS", "stop,stopp,pause,mia stop,mia stopp,mia pause")
    stopwords_list = [normalize_cmd(x) for x in stopwords_raw.split(",") if normalize_cmd(x)]
    STOP_COMMANDS = set(stopwords_list) if stopwords_list else {
        normalize_cmd(x) for x in "stop,stopp,pause,mia stop,mia stopp,mia pause".split(",")
    }
    stopwords = stopwords_raw

    # TTS streaming: LLM deltas werden in Chunks gesprochen
    tts_stream = env_bool("MIA_TTS_STREAM", True)
    tts_min_chars = env_int("MIA_TTS_STREAM_MIN_CHARS", 80)
    tts_max_chars = env_int("MIA_TTS_STREAM_MAX_CHARS", 180)
    tts_adaptive_chunking = env_bool("MIA_TTS_ADAPTIVE_CHUNKING", True)
    tts_phase1_sec = env_float("MIA_TTS_PHASE1_SEC", 1.2)
    tts_phase2_sec = env_float("MIA_TTS_PHASE2_SEC", 3.5)
    tts_first_min_chars = env_int("MIA_TTS_FIRST_MIN_CHARS", 25)
    tts_first_max_wait_sec = env_float("MIA_TTS_FIRST_MAX_WAIT_SEC", 0.8)

    # Echo guard Parameter
    echo_overlap = env_float("MIA_ECHO_OVERLAP", 0.45)
    echo_history = env_int("MIA_ECHO_HISTORY", 12)
    echo_window_sec = env_float("MIA_ECHO_WINDOW_SEC", 2.5)

    # Session Modus: new vs default
    session_mode = env_str("MIA_SESSION_MODE", "").lower().strip()
    if session_mode == "new":
        sess_path = new_session_path()
        log(f"Session mode=new -> {sess_path}", debug)
    else:
        sess_path = Path(env_str("MIA_SESSION", "")) if env_str("MIA_SESSION", "") else session_path_default()

    ensure_dir(sess_path.parent)
    sess = load_session(sess_path)

    # Wenn Session leer: System Prompt setzen
    if not sess.get("messages"):
        sess["messages"] = [{
            "role": "system",
            "content": (
                "Du bist Mia, eine deutsche Sprach-Assistentin. "
                "Antworte NUR auf Deutsch. "
                "Keine Fremdsprachen, keine chinesischen Zeichen. "
                "Antworte klar. Wenn der Nutzer 'in Punkten' fragt: kurze Punkte."
            )
        }]
        save_session(sess_path, sess)

    # Start Logging
    log("Mia Talk starting...", debug)
    log(f"Session: {sess_path}", debug)
    log(f"Ollama: model={model} keepalive={keepalive} host={ollama_host}", debug)
    log(f"Whisper: model={Path(whisper_model).name} lang={whisper_lang}", debug)
    log(f"Wake: require_wake={require_wake} awake_timeout_sec={awake_timeout_sec}", debug)
    log(f"Logfile: {LOG_FILE}", debug)
    log(f"TTS stream: {tts_stream} min_chars={tts_min_chars} max_chars={tts_max_chars}", debug)
    log(
        f"TTS adaptive: enabled={tts_adaptive_chunking} p1={tts_phase1_sec:.2f}s p2={tts_phase2_sec:.2f}s",
        debug,
    )
    log(
        f"TTS first-chunk: min_chars={tts_first_min_chars} max_wait={tts_first_max_wait_sec:.2f}s",
        debug,
    )
    log(f"Stopwords: {stopwords}", debug)
    log(f"Echo guard: overlap>={echo_overlap:.2f} history={echo_history} window={echo_window_sec:.2f}s", debug)

    # Warmup
    if warmup:
        try:
            dt, wr = ollama_warmup_http(ollama_host, model, keepalive)
            log(f"Warmup: model chat done in {dt:.2f}s", debug)
            if wr:
                log(f"Warmup response: {wr[:40]}", debug)
        except Exception as e:
            log(f"Warmup WARN: {e}", debug)

    # temp input wavs
    tmp_dir = Path("/tmp/mia_talk")
    ensure_dir(tmp_dir)

    log("Ready. Exit phrase: 'Mia Ende'. Ctrl+C beendet ebenfalls.", debug)

    # Awake-Zustand:
    # - awake=False bedeutet: Wakeword nötig (wenn require_wake=True)
    awake = False
    awake_until = 0.0

    # Echo-Historie der Assistant-Ausgaben
    recent_assistant: Deque[str] = deque(maxlen=max(1, echo_history))

    # Minimal-State-Machine
    state = "IDLE"  # IDLE, LISTENING, RESPONDING, CANCELLED, AWAIT_CONFIRMATION
    last_user_text = ""
    last_assistant_full = ""
    last_cancelled_assistant_partial = ""
    last_turn_was_cancelled = False

    def set_state(new_state: str) -> None:
        nonlocal state
        if state != new_state:
            if debug:
                log(f"STATE: {state} -> {new_state}", debug)
            state = new_state

    # TTS Manager (Queue + Worker)
    tts_mgr = TTSManager(debug=debug)

    def _on_barge_hit(_txt: str) -> None:
        nonlocal last_turn_was_cancelled, last_cancelled_assistant_partial
        set_state("CANCELLED")
        last_turn_was_cancelled = True
        last_cancelled_assistant_partial = ""
        request_stop_output(tts_mgr, reason="barge", debug=debug)

    barge_mon = BargeInMonitor(
        rate=rate,
        whisper_cli=whisper_cli,
        whisper_lang=whisper_lang,
        barge_model=barge_whisper_model,
        window_sec=barge_window_sec,
        interval_sec=barge_interval_sec,
        is_stop_match=lambda txt: is_stop_command(txt, STOP_COMMANDS),
        on_hit=_on_barge_hit,
        debug=debug,
    )
    barge_mon.start()
    log(
        f"Whisper(barge): model={Path(barge_whisper_model).name} window={barge_window_sec:.2f}s interval={barge_interval_sec:.2f}s",
        debug,
    )

    speaking_wait_logged = False

    while not STOP:
        # Wake timeout prüfen
        if awake and time.time() > awake_until:
            awake = False
            if debug:
                log("Awake timeout: Mia schläft wieder (Wakeword nötig).", debug)

        # AUDIO_OUT_ACTIVE = stabiler Audio-Ausgabe-Modus
        speaking_now = is_audio_output_active()
        if speaking_now:
            if debug and not speaking_wait_logged:
                speaking_wait_logged = True
                log(">>> Mia spricht: Main-Loop wartet passiv (Barge-in only).", debug)
            while not STOP and is_audio_output_active():
                time.sleep(0.05)
            continue

        speaking_wait_logged = False
        log(">>> Ich höre zu (sprich jetzt)…", debug)

        # Aufnahme-Dateiname
        utt_id = str(uuid.uuid4())
        in_wav = tmp_dir / f"mia_in_{utt_id}.wav"

        set_state("LISTENING")

        # Audio aufnehmen bis VAD "speech segment" erkannt hat
        t0_record = time.perf_counter()
        ok = record_vad_to_wav(
            output_wav=in_wav,
            rate=rate,
            frame_ms=frame_ms,
            vad_mode=vad_mode,
            silence_limit_sec=silence_limit,
            min_speech_sec=min_speech,
            trigger_ratio=trigger_ratio,
            ring_ms=ring_ms,
            max_sec=max_sec,
            rms_threshold=rms_threshold,
            debug=debug,
        )
        t_record = time.perf_counter() - t0_record
        if STOP:
            break
        if not ok:
            # nichts erkannt -> weiter
            continue

        turn_id = str(uuid.uuid4())
        mode = "stream" if tts_stream else "nonstream"
        t_whisper: Optional[float] = None
        t_ollama_total: Optional[float] = None
        t_first_token: Optional[float] = None
        t_tts_total = 0.0
        t_play_total = 0.0

        if TRACE_TIMINGS:
            trace_begin_turn(turn_id)

        try:
            # Whisper transkribieren
            t0_whisper = time.perf_counter()
            try:
                t = whisper_transcribe(in_wav, whisper_model, whisper_cli, whisper_lang, debug).strip()
                t_whisper = time.perf_counter() - t0_whisper
                if TRACE_TIMINGS:
                    log(f"Trace: after_whisper turn={turn_id}", debug)
            except Exception as e:
                t_whisper = time.perf_counter() - t0_whisper
                log(f"Whisper ERROR: {e}", debug)
                continue
            finally:
                # input wav wegräumen
                try:
                    in_wav.unlink(missing_ok=True)
                except Exception:
                    pass

            # Noise/Leere Transkripte ignorieren
            if is_noise_transcript(t):
                continue

            # STOP funktioniert hier im Nicht-SPEAKING-Modus weiterhin.
            # Voice-Exit wird ebenfalls nur hier (nicht während SPEAKING) geprüft.
            if is_stop_command(t, STOP_COMMANDS):
                if debug:
                    log(f"Barge-in: stop erkannt -> '{t}'", debug)
                    log("Barge-in: cancel requested (stop)", debug)
                set_state("CANCELLED")
                last_turn_was_cancelled = True
                last_cancelled_assistant_partial = ""
                request_stop_output(tts_mgr, reason="stop", debug=debug)
                log("Assistant output cancelled -> continue listening.", debug)
                continue

            # Exit nur wenn nicht im Antwort-/Sprechmodus
            if contains_any(t, exitwords):
                if state == "RESPONDING" or SPEAKING.is_set() or is_audio_output_active():
                    if debug:
                        log("Exit ignored while speaking/responding", debug)
                    continue
                log(f"Exit erkannt: '{t}' -> shutdown.", debug)
                break

            # Wakeword prüfen
            woke_now = contains_any(t, wakewords)
            if woke_now:
                awake = True
                awake_until = time.time() + float(awake_timeout_sec)
                if debug:
                    log(f"Wake erkannt -> Mia ist wach für {awake_timeout_sec}s.", debug)

            # require_wake: wenn nicht awake -> ignorieren
            if require_wake and not awake:
                if debug:
                    log(f"Wake required -> ignored: '{t}'", debug)
                continue

            # Wenn awake: Timeout verlängern
            if awake:
                awake_until = time.time() + float(awake_timeout_sec)

            # Wakeprefix entfernen, damit prompt sauber ist
            user_text = strip_wake_prefix(t, wakewords) if woke_now else t
            user_text = user_text.strip()
            if not user_text:
                continue

            user_text_norm = norm_text(user_text)
            if user_text_norm in {"ja", "nein"} and state != "AWAIT_CONFIRMATION":
                if debug:
                    log(f"Ignored standalone confirmation outside confirmation state: '{user_text}'", debug)
                continue

            if user_text_norm in {"weiter", "bitte weiter", "mach weiter"}:
                if state == "CANCELLED" or last_turn_was_cancelled:
                    user_text = "Bitte fahre genau dort fort, wo du aufgehört hast. Wiederhole NICHT den Anfang. Mach weiter."
                    user_text_norm = norm_text(user_text)
                    last_turn_was_cancelled = False
                    if debug:
                        log("Continue command accepted after cancel -> forcing continuation prompt.", debug)
                else:
                    if debug:
                        log(f"Continue ignored without cancelled context: '{user_text}'", debug)
                    continue

            if is_phantom_utterance(
                user_text,
                state,
                phantom_exact,
                phantom_contains,
                phantom_max_words,
                phantom_max_chars,
                phantom_year_filter,
                confirm_allowed,
            ):
                if debug:
                    log(f"Phantom-filter -> ignored: '{user_text}'", debug)
                continue

            # Echo-Guard nur kurz nach Playback-Ende:
            # Wenn user_text stark mit recent_assistant überlappt, ist es vermutlich Mia's Echo.
            echo_recent = (time.time() - LAST_PLAY_END_TS) < echo_window_sec
            if echo_recent and (not is_question_like(user_text)) and looks_like_echo(user_text, recent_assistant, min_overlap=echo_overlap):
                if debug:
                    log(f"Echo-guard -> ignored: '{user_text}'", debug)
                continue

            last_user_text = user_text
            log(f"Du: {user_text}", debug)
            sess["messages"].append({"role": "user", "content": user_text})

            # neue Ausgabe -> TTS für neuen Turn resetten
            if tts_stream:
                tts_mgr.reset_for_new_turn()
                log(">>> Mia spricht…", debug)
            else:
                CANCEL_TTS.clear()

            assistant = ""
            set_state("RESPONDING")
            try:
                if requests is None:
                    raise RuntimeError("requests missing -> install requests in venv")

                t0_ollama = time.perf_counter()
                if tts_stream:
                    real_chunk_emitted = {"v": False}
                    first_chunk_logged = {"v": False}
                    chunks_enqueued = {"n": 0}

                    # Wenn ein Chunk fertig ist -> in TTS Queue
                    def emit_chunk(txt: str):
                        if STOP or CANCEL_TTS.is_set():
                            return
                        real_chunk_emitted["v"] = True
                        if debug and not first_chunk_logged["v"]:
                            first_chunk_logged["v"] = True
                            log(f"First real chunk emitted at t={time.perf_counter() - t0_ollama:.2f}s", debug)
                        tts_mgr.enqueue(txt, turn_id=turn_id)
                        chunks_enqueued["n"] += 1
                        recent_assistant.append(txt)

                    chunker = StreamChunker(
                        emit=emit_chunk,
                        min_chars=tts_min_chars,
                        max_chars=tts_max_chars,
                        adaptive=tts_adaptive_chunking,
                        phase1_sec=tts_phase1_sec,
                        phase2_sec=tts_phase2_sec,
                        first_min_chars=tts_first_min_chars,
                        first_max_wait_sec=tts_first_max_wait_sec,
                    )

                    # Bei jedem LLM delta -> chunker.push
                    def on_delta(delta: str):
                        nonlocal t_first_token
                        if STOP or CANCEL_TTS.is_set():
                            return
                        if t_first_token is None:
                            t_first_token = time.perf_counter() - t0_ollama
                        chunker.push(delta)

                    assistant = ollama_chat_http_stream(
                        ollama_host, model, keepalive, sess["messages"], debug, on_delta=on_delta
                    )
                    # Rest raus
                    chunker.flush()
                    if chunks_enqueued["n"] == 0 and (assistant or "").strip():
                        log("STREAM_TTS WARN: no chunks enqueued; enqueue full assistant as fallback", debug)
                        tts_mgr.enqueue(assistant, turn_id=turn_id)
                        chunks_enqueued["n"] += 1
                        real_chunk_emitted["v"] = True
                else:
                    # ohne streaming: erst komplette Antwort holen, dann ggf. sprechen (hier: nur log)
                    assistant = ollama_chat_http_stream(
                        ollama_host, model, keepalive, sess["messages"], debug, on_delta=None
                    )
                t_ollama_total = time.perf_counter() - t0_ollama

            except Exception as e:
                if "t0_ollama" in locals():
                    t_ollama_total = time.perf_counter() - t0_ollama
                log(f"Ollama ERROR: {e}", debug)
                assistant = ""

            # Wenn user stopp gesagt hat während Streaming -> abbrechen
            if CANCEL_TTS.is_set():
                set_state("CANCELLED")
                last_turn_was_cancelled = True
                last_cancelled_assistant_partial = assistant or ""
                trace_cancel_turn(turn_id)
                request_stop_output(tts_mgr, reason="cancel", debug=debug)
                log("Assistant output cancelled -> continue listening.", debug)
                continue

            # Falls LLM leer -> fallback
            assistant = (assistant or "").strip() or "Ich habe gerade keine Antwort. Bitte wiederhole das."
            log(f"Mia: {assistant}", debug)

            if tts_stream and (not real_chunk_emitted["v"]):
                tts_mgr.enqueue(assistant, turn_id=turn_id)

            # Volltext auch in echo history (zusätzlich zu chunk history)
            recent_assistant.append(assistant)

            # Session speichern
            sess["messages"].append({"role": "assistant", "content": assistant})
            save_session(sess_path, sess)
            last_assistant_full = assistant
            last_cancelled_assistant_partial = ""
            last_turn_was_cancelled = False
            set_state("IDLE")

            # WICHTIG:
            # Kein tts_mgr.finish()!
            # - Worker darf laufen (Queue wartet)
            # - SPEAKING wird NUR während Synth/Playback gesetzt/gelöscht
        finally:
            if TRACE_TIMINGS:
                trace_close_turn(turn_id)
                if CANCEL_TTS.is_set() or STOP:
                    trace_cancel_turn(turn_id)
                trace_wait_done(turn_id, timeout_sec=6.0)
                t_tts_total, t_play_total = trace_pop_turn(turn_id)
                log(
                    f"TIMING turn={turn_id} record={fmt_timing(t_record)} whisper={fmt_timing(t_whisper)} "
                    f"ollama_total={fmt_timing(t_ollama_total)} first_token={fmt_timing(t_first_token)} "
                    f"tts_total={fmt_seconds(t_tts_total)} play_total={fmt_seconds(t_play_total)} mode={mode}",
                    debug,
                )

    # Shutdown: laufende Audioausgabe stoppen
    if tts_stream:
        tts_mgr.stop(reason="shutdown")

    try:
        barge_mon.stop()
    except Exception:
        pass

    log("Mia Talk shutdown (graceful).", debug)


if __name__ == "__main__":
    # top-level guard: falls irgendwas ungefangen hochkommt -> traceback loggen
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        log(f"FATAL UNCAUGHT: {e}\n{tb}", debug=True, also_stdout=True)
        raise
