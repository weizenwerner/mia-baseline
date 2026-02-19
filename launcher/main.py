#!/usr/bin/env python3
import os
import sys
import re
import json
import time
import subprocess
import threading
import queue
import signal
from pathlib import Path
from typing import List, Dict, Optional, Any

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

DEFAULT_MODULES_DIR = Path(__file__).resolve().parents[1] / "modules"
MODULES_DIR = Path(
    os.getenv("MIA_MODULES_DIR", str(DEFAULT_MODULES_DIR if DEFAULT_MODULES_DIR.exists() else "/data/mia/modules"))
)
SESSIONS_DIR = Path("/data/mia/memory/sessions")
STATE_FILE = Path("/data/mia/launcher/launcher_state.json")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def parse_config_env_llm(config_path: Path) -> Optional[str]:
    if not config_path.exists():
        return None
    txt = read_text(config_path)
    m = re.search(r'^\s*MIA_LLM\s*=\s*("([^"]+)"|\'([^\']+)\'|([^\s#]+))', txt, flags=re.M)
    if not m:
        return None
    return (m.group(2) or m.group(3) or m.group(4) or "").strip() or None

def list_modules() -> List[Dict]:
    mods: List[Dict] = []
    if not MODULES_DIR.exists():
        return mods
    for d in sorted(MODULES_DIR.iterdir()):
        if not d.is_dir():
            continue
        run_sh = d / "run.sh"
        if run_sh.exists():
            default_llm = parse_config_env_llm(d / "config.env")
            mods.append({
                "name": d.name,
                "run_sh": run_sh,
                "default_llm": default_llm or "",
            })
    return mods

def session_preview(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        msgs = data.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                t = (m.get("content") or "").strip()
                if t:
                    t = " ".join(t.split())
                    return (t[:60] + "…") if len(t) > 60 else t
    except Exception:
        pass
    return ""

def list_sessions() -> List[Path]:
    ensure_dir(SESSIONS_DIR)
    return sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

def safe_session_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = "session"
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "", name)
    name = name.strip("_-") or "session"
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{name}_{ts}.json"

def create_session(title: str) -> Path:
    ensure_dir(SESSIONS_DIR)
    filename = safe_session_filename(title)
    p = SESSIONS_DIR / filename
    init = {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "title": (title or "").strip() or filename.replace(".json", ""),
        "messages": []
    }
    p.write_text(json.dumps(init, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def list_ollama_models() -> List[str]:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if r.returncode != 0:
            return []
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        if len(lines) < 2:
            return []
        models: List[str] = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []

def load_state() -> Dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_state(state: Dict) -> None:
    try:
        ensure_dir(STATE_FILE.parent)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

class LauncherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mia Launcher")
        self.geometry("980x720")

        self.modules = list_modules()
        self.models = list_ollama_models()
        self.sessions = list_sessions()

        self.state_data = load_state()

        self.proc: Optional[subprocess.Popen] = None
        self.proc_thread: Optional[threading.Thread] = None
        self.stop_reader = threading.Event()
        self.compact_talk_output = False
        self.talk_ready_seen = False
        self.talk_last_speaker: Optional[str] = None
        self.talk_filter_error_count = 0
        self.talk_is_awake = False

        # NEW: output queue to avoid UI freeze
        self.out_q: "queue.Queue[Any]" = queue.Queue()
        self._flush_job: Optional[str] = None

        self.talk_stream_start_idx: Optional[str] = None

        self._build_ui()
        self._load_defaults()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # start periodic flush loop
        self._schedule_flush()

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        sel = ttk.LabelFrame(root, text="Start-Auswahl", padding=12)
        sel.pack(fill="x")

        ttk.Label(sel, text="Modul").grid(row=0, column=0, sticky="w")
        self.module_var = tk.StringVar()
        self.module_cb = ttk.Combobox(sel, textvariable=self.module_var, state="readonly", width=45)
        self.module_cb.grid(row=0, column=1, sticky="we", padx=(8, 8))
        self.module_cb.bind("<<ComboboxSelected>>", self.on_module_changed)

        ttk.Label(sel, text="Session").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.session_var = tk.StringVar()
        self.session_cb = ttk.Combobox(sel, textvariable=self.session_var, state="readonly", width=45)
        self.session_cb.grid(row=1, column=1, sticky="we", padx=(8, 8), pady=(8, 0))

        self.btn_new_session = ttk.Button(sel, text="Neu…", command=self.new_session)
        self.btn_new_session.grid(row=1, column=2, sticky="w", pady=(8, 0))

        self.btn_open_sessions = ttk.Button(sel, text="Ordner öffnen", command=self.open_sessions_folder)
        self.btn_open_sessions.grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(sel, text="LLM").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.llm_var = tk.StringVar()
        self.llm_cb = ttk.Combobox(sel, textvariable=self.llm_var, state="readonly", width=45)
        self.llm_cb.grid(row=2, column=1, sticky="we", padx=(8, 8), pady=(8, 0))

        self.btn_refresh_llm = ttk.Button(sel, text="Refresh", command=self.refresh_llm)
        self.btn_refresh_llm.grid(row=2, column=2, sticky="w", pady=(8, 0))

        sel.columnconfigure(1, weight=1)

        ctrl = ttk.Frame(root)
        ctrl.pack(fill="x", pady=(12, 6))

        self.btn_start = ttk.Button(ctrl, text="Start", command=self.start)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Bereit.")
        ttk.Label(ctrl, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        outf = ttk.LabelFrame(root, text="Ausgabe", padding=8)
        outf.pack(fill="both", expand=True)

        self.out_text = tk.Text(outf, height=18, wrap="word")
        self.out_text.pack(side="left", fill="both", expand=True)

        out_sb = ttk.Scrollbar(outf, command=self.out_text.yview)
        out_sb.pack(side="right", fill="y")
        self.out_text.configure(yscrollcommand=out_sb.set)

        inf = ttk.LabelFrame(root, text="Eingabe", padding=8)
        inf.pack(fill="x", expand=False, pady=(10, 0))

        self.in_text = tk.Text(inf, height=6, wrap="word")
        self.in_text.pack(side="left", fill="both", expand=True)

        in_sb = ttk.Scrollbar(inf, command=self.in_text.yview)
        in_sb.pack(side="right", fill="y")
        self.in_text.configure(yscrollcommand=in_sb.set)

        btn_row = ttk.Frame(root)
        btn_row.pack(fill="x", pady=(6, 0))

        self.btn_send = ttk.Button(btn_row, text="Senden", command=self.send_text, state="disabled")
        self.btn_send.pack(side="left")

        hint = ttk.Label(btn_row, text="Tipp: Shift+Enter = neue Zeile")
        hint.pack(side="left", padx=(12, 0))

        self.in_text.bind("<Return>", self._on_enter_send)
        self.in_text.bind("<Shift-Return>", self._on_shift_enter_newline)

        self._refresh_module_list()
        self._refresh_session_list()
        self._refresh_llm_list()

    def _on_enter_send(self, _evt):
        self.send_text()
        return "break"

    def _on_shift_enter_newline(self, _evt):
        self.in_text.insert("insert", "\n")
        return "break"

    def _refresh_module_list(self):
        self.module_cb["values"] = [m["name"] for m in self.modules]

    def _refresh_session_list(self):
        self.sessions = list_sessions()
        vals = []
        for p in self.sessions[:80]:
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
            prev = session_preview(p)
            label = f"{p.name}  ({mtime})"
            if prev:
                label += f" — {prev}"
            vals.append(label)
        self.session_cb["values"] = vals

    def _refresh_llm_list(self):
        self.models = list_ollama_models()
        self.llm_cb["values"] = self.models

    def _load_defaults(self):
        last_mod = self.state_data.get("last_module", "")
        last_sess = self.state_data.get("last_session", "")
        last_llm = self.state_data.get("last_llm", "")

        mod_names = [m["name"] for m in self.modules]
        if last_mod in mod_names:
            self.module_var.set(last_mod)
        elif mod_names:
            self.module_var.set(mod_names[0])

        self._refresh_session_list()
        if last_sess:
            for i, p in enumerate(self.sessions[:80]):
                if p.name == last_sess:
                    self.session_cb.current(i)
                    self.session_var.set(self.session_cb["values"][i])
                    break
        else:
            if self.session_cb["values"]:
                self.session_cb.current(0)
                self.session_var.set(self.session_cb["values"][0])

        self._refresh_llm_list()
        mod = self.get_selected_module()
        default_llm = mod.get("default_llm", "") if mod else ""
        llm_pick = last_llm or default_llm

        if llm_pick and llm_pick in self.models:
            self.llm_var.set(llm_pick)
        elif default_llm and default_llm in self.models:
            self.llm_var.set(default_llm)
        elif self.models:
            self.llm_var.set(self.models[0])

        self._apply_module_io_mode()

    def _apply_module_io_mode(self):
        """
        Chat: stdin enabled + send button
        Talk/others: stdin disabled, input disabled
        """
        mod = self.get_selected_module()
        name = (mod["name"] if mod else "").strip().lower()
        is_chat = (name == "chat")
        self.compact_talk_output = (name == "talk")
        self.talk_ready_seen = False
        self.talk_last_speaker = None
        self.talk_filter_error_count = 0
        self.talk_is_awake = False
        self.talk_stream_start_idx = None

        self.btn_send.config(state=("normal" if (is_chat and self.proc and self.proc.poll() is None) else "disabled"))

        if is_chat:
            try:
                self.in_text.config(state="normal")
            except Exception:
                pass
        else:
            try:
                self.in_text.delete("1.0", "end")
            except Exception:
                pass
            try:
                self.in_text.config(state="disabled")
            except Exception:
                pass

    def get_selected_module(self) -> Optional[Dict]:
        name = self.module_var.get().strip()
        for m in self.modules:
            if m["name"] == name:
                return m
        return None

    def get_selected_session_path(self) -> Optional[Path]:
        label = self.session_var.get()
        if not label:
            return None
        values = list(self.session_cb["values"])
        if label in values:
            idx = values.index(label)
            if idx < len(self.sessions):
                return self.sessions[idx]
        fname = label.split("  (", 1)[0].strip()
        p = SESSIONS_DIR / fname
        return p if p.exists() else None

    def on_module_changed(self, _evt=None):
        mod = self.get_selected_module()
        if not mod:
            return
        default_llm = mod.get("default_llm", "")
        if default_llm and default_llm in self.models:
            self.llm_var.set(default_llm)
        self._apply_module_io_mode()

    def refresh_llm(self):
        self._refresh_llm_list()
        cur = self.llm_var.get().strip()
        if cur and cur in self.models:
            return
        mod = self.get_selected_module()
        default_llm = mod.get("default_llm", "") if mod else ""
        if default_llm and default_llm in self.models:
            self.llm_var.set(default_llm)
        elif self.models:
            self.llm_var.set(self.models[0])

    def new_session(self):
        title = simpledialog.askstring("Neue Session", "Name der Session (z.B. 'Heizung Neuwied'):", parent=self)
        if title is None:
            return
        p = create_session(title)
        self._refresh_session_list()
        for i, sp in enumerate(self.sessions[:80]):
            if sp.name == p.name:
                self.session_cb.current(i)
                self.session_var.set(self.session_cb["values"][i])
                break

    def open_sessions_folder(self):
        ensure_dir(SESSIONS_DIR)
        try:
            subprocess.Popen(["xdg-open", str(SESSIONS_DIR)])
        except Exception:
            messagebox.showinfo("Info", f"Sessions liegen hier: {SESSIONS_DIR}")

    def _schedule_flush(self):
        self._flush_output()
        self._flush_job = self.after(50, self._schedule_flush)  # 20 fps flush

    def _apply_stream_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("type")
        tid = event.get("tid")

        if kind == "begin":
            if self.talk_stream_tid != tid:
                self.talk_stream_tid = tid
                self.talk_stream_text = ""
                self.out_text.insert("end", "Mia: ")
                self.talk_stream_start_idx = self.out_text.index("end-1c")
                self.out_text.insert("end", "\n")
            return

        if kind == "delta":
            if tid != self.talk_stream_tid:
                self.talk_stream_tid = tid
                self.talk_stream_text = ""
                self.out_text.insert("end", "Mia: ")
                self.talk_stream_start_idx = self.out_text.index("end-1c")
                self.out_text.insert("end", "\n")
            delta = event.get("text", "")
            if not delta:
                return
            self.talk_stream_text += delta
            if self.talk_stream_start_idx:
                self.out_text.delete(self.talk_stream_start_idx, f"{self.talk_stream_start_idx} lineend")
                self.out_text.insert(self.talk_stream_start_idx, self.talk_stream_text)
            return

        if kind == "end" and tid == self.talk_stream_tid:
            self.talk_stream_tid = None
            self.talk_stream_text = ""
            self.talk_stream_start_idx = None

    def _flush_output(self):
        items = []
        try:
            for _ in range(300):
                items.append(self.out_q.get_nowait())
        except queue.Empty:
            pass

        if not items:
            return

        for item in items:
            msg = str(item)
            if self.compact_talk_output and msg.startswith("Mia:"):
                content = msg.split(":", 1)[1].strip()
                if self.talk_stream_start_idx is None:
                    self.out_text.insert("end", "Mia: ")
                    self.talk_stream_start_idx = self.out_text.index("end-1c")
                    self.out_text.insert("end", "\n")
                self.out_text.delete(self.talk_stream_start_idx, f"{self.talk_stream_start_idx} lineend")
                self.out_text.insert(self.talk_stream_start_idx, content)
                continue

            self.talk_stream_start_idx = None
            self.out_text.insert("end", msg)

        self.out_text.see("end")

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Läuft", "Ein Modul läuft bereits. Erst stoppen.")
            return

        mod = self.get_selected_module()
        if not mod:
            messagebox.showerror("Fehler", "Kein Modul ausgewählt.")
            return

        sess_path = self.get_selected_session_path()
        if not sess_path:
            messagebox.showerror("Fehler", "Keine Session ausgewählt.")
            return

        llm = self.llm_var.get().strip()
        run_sh = Path(mod["run_sh"])

        if not run_sh.exists():
            messagebox.showerror("Fehler", f"run.sh nicht gefunden: {run_sh}")
            return

        env = os.environ.copy()
        env["MIA_SESSION_MODE"] = "path"
        env["MIA_SESSION"] = str(sess_path)
        if llm:
            env["MIA_LLM"] = llm
        env["MIA_STARTED_BY"] = "launcher_gui"
        env["MIA_CHAT_NO_PROMPT"] = "1"
        if mod["name"].strip().lower() == "talk":
            env.setdefault("MIA_SESSION_RESET", "1")

        # persist state
        self.state_data["last_module"] = mod["name"]
        self.state_data["last_session"] = sess_path.name
        self.state_data["last_llm"] = llm
        save_state(self.state_data)

        name = mod["name"].strip().lower()
        is_chat = (name == "chat")
        self.compact_talk_output = (name == "talk")
        self.talk_ready_seen = False
        self.talk_last_speaker = None
        self.talk_filter_error_count = 0
        self.talk_is_awake = False
        self.talk_stream_start_idx = None

        try:
            try:
                run_sh.chmod(run_sh.stat().st_mode | 0o111)
            except Exception:
                pass

            self.out_q.put(
                f"\n--- START ---\nModul: {mod['name']}\nSession: {sess_path}\nLLM: {llm}\n"
                f"ModuleDir: {MODULES_DIR}\n------------\n"
            )
            self.status_var.set("Läuft…")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")

            if is_chat:
                self.btn_send.config(state="normal")
                try:
                    self.in_text.config(state="normal")
                except Exception:
                    pass
            else:
                self.btn_send.config(state="disabled")
                try:
                    self.in_text.delete("1.0", "end")
                    self.in_text.config(state="disabled")
                except Exception:
                    pass

            self.stop_reader.clear()

            self.proc = subprocess.Popen(
                [str(run_sh)],
                env=env,
                stdin=(subprocess.PIPE if is_chat else None),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            self.proc_thread = threading.Thread(target=self._reader_thread, daemon=True)
            self.proc_thread.start()

        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self.status_var.set("Bereit.")
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.btn_send.config(state="disabled")

    def _canonical_dialog_line(self, txt: str) -> str:
        try:
            clean = (txt or "").strip()
            low = clean.lower()

            if "wake erkannt" in low:
                self.talk_is_awake = True
                self.talk_last_speaker = "Mia"
                return "Mia: (wach)\n"
            if "awake timeout" in low or "schläft wieder" in low:
                self.talk_is_awake = False
                self.talk_last_speaker = "Mia"
                return "Mia: (schlafmodus)\n"

            msg = re.sub(r"^\[[^\]]+\]\s*", "", clean).strip()
            if msg.startswith(("TURN", "Ollama", "CHUNK", "TTS", "Piper", "PLAY", "VAD(", "Whisper", "Cleanup", "BargeIn", "Ready.")):
                return ""

            if msg.startswith("Du:"):
                if not self.talk_is_awake:
                    return ""
                content = msg.split(":", 1)[1].strip()
                self.talk_last_speaker = "Du"
                return f"Du: {content}\n" if content else "Du:\n"

            if msg.startswith("Mia:"):
                content = msg.split(":", 1)[1].strip()
                self.talk_last_speaker = "Mia"
                return f"Mia: {content}\n" if content else "Mia:\n"

            if msg.startswith(("-", "*", "•")) or re.match(r"^\d+[\.)]\s+", msg):
                if self.talk_last_speaker in {"Du", "Mia"}:
                    return f"{self.talk_last_speaker}: {msg}\n"

            return f"{msg}\n"
        except Exception:
            return ""
    def _filter_output_line(self, line: str) -> str:
        if not self.compact_talk_output:
            return line

        txt = (line or "").strip()
        if not txt:
            return line

        if not self.talk_ready_seen:
            if txt == "Ready." or txt.endswith(" Ready."):
                self.talk_ready_seen = True
                return ""
            return line

        return self._canonical_dialog_line(txt)

    def _reader_thread(self):
        assert self.proc is not None
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                if self.stop_reader.is_set():
                    break

                try:
                    filtered = self._filter_output_line(line)
                except Exception as e:
                    self.talk_filter_error_count += 1
                    if self.talk_filter_error_count <= 3:
                        self.out_q.put(f"\n[Launcher] Ausgabe-Filterfehler: {e}\n")
                    continue
                if not filtered:
                    continue
                self.out_q.put(filtered)
        except Exception as e:
            self.out_q.put(f"\n[Launcher] Reader-Fehler: {e}\n")
        finally:
            rc = self.proc.poll()
            if rc is None:
                if self.stop_reader.is_set():
                    rc = self._terminate_process(grace_sec=1.0)
                    self.after(0, self._on_process_end, rc)
                    return

                self.out_q.put("\n[Launcher] Hinweis: Ausgabestream beendet, warte auf Prozessende...\n")
                threading.Thread(target=self._wait_process_exit_after_eof, daemon=True).start()
                return

            self.after(0, self._on_process_end, rc)
    def _wait_process_exit_after_eof(self):
        if self.proc is None:
            return
        try:
            rc = self.proc.wait()
        except Exception:
            rc = -1
        self.after(0, self._on_process_end, rc)

    def _on_process_end(self, rc: int):
        self.out_q.put(f"\n--- ENDE (rc={rc}) ---\n")
        self.status_var.set("Bereit.")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_send.config(state="disabled")
        self.proc = None
        self._apply_module_io_mode()

    def send_text(self):
        mod = self.get_selected_module()
        name = (mod["name"] if mod else "").strip().lower()
        if name != "chat":
            return

        if self.proc is None or self.proc.poll() is not None:
            return
        if self.proc.stdin is None:
            return

        text = self.in_text.get("1.0", "end").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return

        try:
            self.in_text.delete("1.0", "end")
        except Exception:
            pass

        try:
            if "\n" in text:
                payload = "PASTE_BEGIN\n" + text + "\nPASTE_END\n"
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()
            else:
                self.proc.stdin.write(text + "\n")
                self.proc.stdin.flush()
        except Exception as e:
            try:
                self.in_text.insert("1.0", text)
            except Exception:
                pass
            messagebox.showerror("Fehler", f"Senden fehlgeschlagen: {e}")

    def _terminate_process(self, grace_sec: float = 1.5) -> int:
        """Terminate running module process (prefer process group) and return exit code."""
        if self.proc is None:
            return 0

        proc = self.proc
        rc = proc.poll()
        if rc is not None:
            return rc

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        deadline = time.time() + max(0.1, grace_sec)
        while time.time() < deadline:
            rc = proc.poll()
            if rc is not None:
                return rc
            time.sleep(0.05)

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        try:
            return proc.wait(timeout=1.0)
        except Exception:
            return -1

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        self.stop_reader.set()
        self.status_var.set("Stoppe…")
        self._terminate_process(grace_sec=1.5)

    def on_close(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("Beenden?", "Ein Modul läuft noch. Beenden und stoppen?"):
                return
            self.stop()
            time.sleep(0.3)
        self.destroy()

def main():
    if "--stdin-test" in sys.argv:
        last_role = None
        for raw in sys.stdin:
            line = (raw or "").strip()
            if not line:
                continue
            if line.startswith("Du:"):
                last_role = "Du"
                content = line.split(":", 1)[1].strip()
                print(f"Du: {content}" if content else "Du:")
                continue
            if line.startswith("Mia:"):
                last_role = "Mia"
                content = line.split(":", 1)[1].strip()
                print(f"Mia: {content}" if content else "Mia:")
                continue
            if (line.startswith(("-", "*", "•")) or re.match(r"^\d+[\.)]\s+", line)) and last_role in {"Du", "Mia"}:
                print(f"{last_role}: {line}")
            else:
                pass
        return
    ensure_dir(SESSIONS_DIR)
    app = LauncherGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
