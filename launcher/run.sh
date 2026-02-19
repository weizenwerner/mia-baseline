#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BOOT_LOG="${MIA_LAUNCHER_BOOT_LOG:-$HERE/launcher_boot.log}"
mkdir -p "$(dirname "$BOOT_LOG")"
exec > >(tee -a "$BOOT_LOG") 2>&1

echo "[Mia Launcher] Starting..."
echo "[Mia Launcher] Boot log: $BOOT_LOG"

# Tkinter launcher needs a graphical session. If none is present, try best-effort fallbacks.
PY_WRAPPER=()
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  # Common local desktop case: X server on :0 but DISPLAY not exported (e.g. ssh shell).
  if [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
    echo "[Mia Launcher] DISPLAY war leer, nutze Fallback DISPLAY=:0"
  fi

  if command -v xvfb-run >/dev/null 2>&1; then
    echo "[Mia Launcher] Kein DISPLAY gefunden – starte via xvfb-run (headless)."
    PY_WRAPPER=(xvfb-run -a)
  elif [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "⚠️  Kein GUI-Display gefunden (DISPLAY/WAYLAND_DISPLAY leer)."
    echo "   Tipp: lokal starten oder DISPLAY setzen (z. B. DISPLAY=:0)."
    echo "   Aktueller Start wird vermutlich an Tkinter scheitern."
  fi
fi

# Prefer venv if present (consistent with your talk module)
PYTHON_BIN="python3"
if [[ -x "/data/mia/venv/bin/python" ]]; then
  PYTHON_BIN="/data/mia/venv/bin/python"
fi

echo "[Mia Launcher] Using Python: $PYTHON_BIN"

# minimal deps
if ! "$PYTHON_BIN" -c "import json, os, sys, subprocess" >/dev/null 2>&1; then
  echo "❌ FATAL: Python seems broken."
  exit 1
fi

# Early Tk check for clearer diagnostics (before main.py traceback).
if ! "${PY_WRAPPER[@]}" "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import tkinter as tk
r = tk.Tk()
r.withdraw()
r.destroy()
PY
then
  echo "❌ FATAL: Tkinter kann kein GUI öffnen."
  echo "   Prüfe DISPLAY/WAYLAND_DISPLAY, X11/Wayland-Berechtigungen und ggf. XAUTHORITY."
  exit 1
fi

exec "${PY_WRAPPER[@]}" "$PYTHON_BIN" -u "$HERE/main.py" "$@"
