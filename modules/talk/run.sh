#!/usr/bin/env bash
# ^ Script läuft mit bash

set -euo pipefail
# -e: bei Fehler sofort abbrechen
# -u: nicht gesetzte Variablen sind Fehler
# -o pipefail: wenn Pipe-Teil fehlschlägt, gesamte Pipe fehlschlägt

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ^ HERE = Ordner, in dem dieses run.sh liegt

cd "$HERE"



# Make local package importable (mia_talk/*)
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

# ^ wechsle in den Ordner, damit relative Pfade stimmen

# Preserve launcher overrides (if launcher exports env vars)
PRESERVE_MIA_SESSION="${MIA_SESSION-}"
PRESERVE_MIA_SESSION_MODE="${MIA_SESSION_MODE-}"
PRESERVE_MIA_LLM="${MIA_LLM-}"
PRESERVE_MIA_KEEPALIVE="${MIA_KEEPALIVE-}"
PRESERVE_MIA_WARMUP="${MIA_WARMUP-}"
# ^ falls du ENV Variablen beim Start exportiert hast, sichern wir sie,
#   damit config.env sie nicht überschreibt

# Load module env (optional)
if [[ -f "$HERE/config.env" ]]; then
  set -a
  # ^ automatisch exportieren: jede Variable, die wir "source" laden, wird in ENV übernommen
  # shellcheck disable=SC1091
  source "$HERE/config.env"
  # ^ lädt Variablen aus config.env
  set +a
fi

# Restore launcher overrides (if set)
# ^ falls du beim Start Werte gesetzt hast, gewinnen die jetzt wieder
if [[ -n "${PRESERVE_MIA_SESSION}" ]]; then export MIA_SESSION="${PRESERVE_MIA_SESSION}"; fi
if [[ -n "${PRESERVE_MIA_SESSION_MODE}" ]]; then export MIA_SESSION_MODE="${PRESERVE_MIA_SESSION_MODE}"; fi
if [[ -n "${PRESERVE_MIA_LLM}" ]]; then export MIA_LLM="${PRESERVE_MIA_LLM}"; fi
if [[ -n "${PRESERVE_MIA_KEEPALIVE}" ]]; then export MIA_KEEPALIVE="${PRESERVE_MIA_KEEPALIVE}"; fi
if [[ -n "${PRESERVE_MIA_WARMUP}" ]]; then export MIA_WARMUP="${PRESERVE_MIA_WARMUP}"; fi

# whisper shared libs (GPU whisper.cpp build)
export LD_LIBRARY_PATH="/data/mia/hear/whisper.cpp/build-cuda/src:/data/mia/hear/whisper.cpp/build-cuda/ggml/src:/data/mia/hear/whisper.cpp/build-cuda/ggml/src/ggml-cuda:${LD_LIBRARY_PATH:-}"
# wichtig: damit whisper-cli die CUDA libs findet (sonst läuft es evtl. nicht / langsamer)

export PULSE_SOURCE=mia_aec_source
# optional auch sink, falls du willst:
# export PULSE_SINK=<dein_standard_sink_name>


# Ensure we use a suitable Python interpreter.
# Priorität:
# 1) explizit gesetztes MIA_PYTHON
# 2) /data/mia/venv/bin/python3
# 3) /data/mia/venv/bin/python
# 4) python3 aus PATH
PY="${MIA_PYTHON:-}"

if [[ -n "$PY" && ! -x "$PY" ]]; then
  echo "⚠️  Hinweis: MIA_PYTHON='$PY' ist nicht ausführbar – versuche Fallbacks."
  PY=""
fi

if [[ -z "$PY" && -x "/data/mia/venv/bin/python3" ]]; then
  PY="/data/mia/venv/bin/python3"
fi

if [[ -z "$PY" && -x "/data/mia/venv/bin/python" ]]; then
  PY="/data/mia/venv/bin/python"
fi

if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "❌ FATAL: Kein nutzbares Python gefunden."
  echo "   Erwartet: MIA_PYTHON oder /data/mia/venv/bin/python3 oder python3 im PATH."
  exit 1
fi

echo "[Mia Talk] Using Python: $PY"

# Dependency hint only (do not abort here; we want main.py logs/traceback)
if ! "$PY" -c "import webrtcvad" >/dev/null 2>&1; then
  echo "⚠️  Hinweis: webrtcvad fehlt in '$PY'."
  echo "   Der Start läuft weiter, um verwertbare Laufzeit-Logs zu erzeugen."
  echo "   Installation (empfohlen): $PY -m pip install webrtcvad"
fi

exec "$PY" -u "$HERE/main.py" "$@"
# ^ startet main.py
# -u: unbuffered output -> Logs kommen sofort, nicht verzögert
# "$@": reicht weitere Parameter durch (falls du später welche nutzt)
