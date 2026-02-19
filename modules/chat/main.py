#!/usr/bin/env python3
"""
Mia Chat (Launcher+Terminal REPL)

- Funktioniert im Terminal (input prompt)
- Funktioniert im Launcher (stdin=PIPE) via stdin.isatty() -> no prompt
- Console minimal (quiet/normal/debug), Logfile immer voll
- Session JSON inkl. summary
- Globale Summary JSONL
"""

import os
import sys
import json
import time
import uuid
import signal
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None

STOP = False


def on_signal(sig, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)


def env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name, "")
    v = v.strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


LEVELS = {"quiet": 0, "normal": 1, "debug": 2}


class MiaLogger:
    def __init__(self):
        self.console_mode = env_str("MIA_CONSOLE", "normal").lower().strip()
        if self.console_mode not in LEVELS:
            self.console_mode = "normal"

        self.console_state_lines = env_bool("MIA_CONSOLE_STATE_LINES", True)
        self.console_show_dialog = env_bool("MIA_CONSOLE_SHOW_DIALOG", True)
        self.console_max_chars = env_int("MIA_CONSOLE_MAX_CHARS", 400)

        log_dir = Path(env_str("MIA_LOG_DIR", "/data/mia/memory/logs"))
        log_file = env_str("MIA_LOG_FILE", "").strip()
        self.log_file = Path(log_file) if log_file else (log_dir / "chat" / "chat.log")
        ensure_dir(self.log_file.parent)

    def _write_file(self, line: str) -> None:
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] {line}\n")
        except Exception:
            pass

    def _print_console(self, msg: str) -> None:
        print(msg, flush=True)

    def _console_allowed(self, category: str) -> bool:
        category = (category or "INFO").upper()
        if category == "ERROR":
            return True
        if category == "BANNER":
            return True
        if category == "STATE":
            return bool(self.console_state_lines)
        if category == "INFO":
            return LEVELS[self.console_mode] >= LEVELS["normal"]
        if category == "DEBUG":
            return LEVELS[self.console_mode] >= LEVELS["debug"]
        return LEVELS[self.console_mode] >= LEVELS["normal"]

    def log(self, msg: str, category: str = "INFO") -> None:
        # file always full
        self._write_file(f"{category}: {msg}")
        # console filtered
        if self._console_allowed(category):
            self._print_console(msg)

    def banner(self, msg: str) -> None:
        self.log(msg, category="BANNER")

    def console_dialog_line(self, prefix: str, text: str) -> None:
        if not self.console_show_dialog:
            return
        t = (text or "").strip()
        if not t:
            return

        # optional truncation (applies to total chars)
        maxc = int(self.console_max_chars or 0)
        if maxc > 0 and len(t) > maxc:
            t = t[:maxc].rstrip() + " …"

        lines = t.splitlines() or [t]
        # first line with prefix, following lines indented
        self._print_console(f"{prefix} {lines[0]}")
        for ln in lines[1:]:
            self._print_console(f"     {ln}")


def session_dir_default() -> Path:
    return Path(env_str("MIA_SESSION_DIR", "/data/mia/memory/sessions"))


def session_path_default() -> Path:
    return session_dir_default() / "chat_default.json"


def new_session_path() -> Path:
    base = session_dir_default()
    ensure_dir(base)
    return base / f"chat_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"


def load_session(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("messages"), list):
                if "summary" not in data or not isinstance(data.get("summary"), dict):
                    data["summary"] = {"last": "", "updated_at": ""}
                return data
        except Exception:
            pass
    return {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
        "summary": {"last": "", "updated_at": ""},
    }


def save_session(path: Path, sess: Dict[str, Any]) -> None:
    sess["updated_at"] = now_iso()
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(sess, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def ensure_system_prompt(sess: Dict[str, Any]) -> None:
    msgs = sess.get("messages")
    if not isinstance(msgs, list):
        sess["messages"] = []
        msgs = sess["messages"]
    if not msgs:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "Du bist Mia, eine deutsche Assistentin.\n"
                    "Antworte NUR auf Deutsch.\n"
                    "Stil: pragmatisch, sachlich, lösungsorientiert.\n"
                    "Keine Smalltalk-Floskeln, keine Empathie-Sätze, kein 'Wie geht es dir'.\n"
                    "Direkt zur Sache, keine Einleitungen.\n"
                    "Wenn der Nutzer mehr Details will: technisch genauer, strukturiert.\n"
                    "Wenn der Nutzer 'in Punkten' fragt: kurze Bulletpoints.\n"
                ),
            }
        )


def pick_session_path(logger: MiaLogger) -> Path:
    # Hard rule: if MIA_SESSION points to an existing file -> always use it
    raw = env_str("MIA_SESSION", "").strip()
    if raw:
        p = Path(raw)
        if p.exists():
            logger.log(f"Session selected via MIA_SESSION -> {p}", category="DEBUG")
            return p

    mode = env_str("MIA_SESSION_MODE", "new").lower().strip()
    if mode == "new":
        p = new_session_path()
        logger.log(f"Session mode=new -> {p}", category="DEBUG")
        return p
    if mode == "path":
        return session_path_default()
    return session_path_default()


def ollama_chat_http_stream(
    host: str,
    model: str,
    keepalive: str,
    messages: List[Dict[str, str]],
    logger: MiaLogger,
    on_delta: Optional[Callable[[str], None]] = None,
) -> str:
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

    logger.log(
        f"Ollama /api/chat model={model} keep_alive={keepalive} num_predict={num_predict}",
        category="DEBUG",
    )

    full: List[str] = []
    stall_timeout_sec = env_int("MIA_OLLAMA_STALL_TIMEOUT_SEC", 30)
    last_line_ts = time.time()

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        if r.status_code >= 400:
            body = (r.text or "").strip()
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {body[:2000]}")

        for line in r.iter_lines(decode_unicode=True, chunk_size=1):
            if STOP:
                break

            now = time.time()
            if not line:
                if (now - last_line_ts) > stall_timeout_sec:
                    logger.log("Ollama stream stall timeout reached -> stopping read", category="DEBUG")
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
                    if on_delta:
                        on_delta(content)

            if obj.get("done") is True:
                break

    return "".join(full).strip()


def ollama_check_model(host: str, model: str) -> Tuple[bool, str]:
    if requests is None:
        return False, "requests missing"
    try:
        url = host.rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=5)
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}"
        data = r.json()
        models = data.get("models", [])
        names = []
        for m in models:
            n = m.get("name")
            if isinstance(n, str):
                names.append(n)
        if model in names:
            return True, "ok"
        return False, "model not found"
    except Exception as e:
        return False, str(e)


def global_summary_log_path() -> Path:
    override = env_str("MIA_GLOBAL_SUMMARY_LOG", "").strip()
    if override:
        return Path(override)
    base = Path(env_str("MIA_LOG_DIR", "/data/mia/memory/logs"))
    return base / "session_summaries.jsonl"


def append_global_summary(session_path: Path, summary: str, logger: MiaLogger) -> None:
    p = global_summary_log_path()
    ensure_dir(p.parent)
    rec = {"ts": now_iso(), "session": str(session_path), "summary": (summary or "").strip()}
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.log(f"Global summary log write failed: {e}", category="DEBUG")


def build_summary_prompt(messages: List[Dict[str, str]], max_bullets: int) -> List[Dict[str, str]]:
    sys_prompt = (
        "Du fasst Gespräche kurz zusammen.\n"
        f"- Maximal {max_bullets} Bulletpoints\n"
        "- Zusätzlich (falls sinnvoll): 'Offene Punkte' maximal 3 Bulletpoints\n"
        "- Keine erfundenen Fakten.\n"
        "- Antworte NUR mit der Zusammenfassung (ohne Vorwort)."
    )
    tail = messages[-40:] if len(messages) > 40 else messages[:]
    return [{"role": "system", "content": sys_prompt}] + tail


def generate_session_summary(
    sess: Dict[str, Any],
    ollama_host: str,
    model: str,
    keepalive: str,
    logger: MiaLogger,
) -> str:
    if not env_bool("MIA_SUMMARY", True):
        return ""
    max_bullets = env_int("MIA_SUMMARY_MAX_BULLETS", 8)
    sum_model = env_str("MIA_SUMMARY_MODEL", "").strip() or model

    msgs = sess.get("messages") or []
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ""

    prompt_msgs = build_summary_prompt(msgs, max_bullets=max_bullets)
    text = ollama_chat_http_stream(ollama_host, sum_model, keepalive, prompt_msgs, logger, on_delta=None)
    return (text or "").strip()


def normalize_csv_words(s: str) -> List[str]:
    return [x.strip().lower() for x in (s or "").split(",") if x.strip()]


def is_command(text: str, csv_words: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    words = normalize_csv_words(csv_words)
    return any(t == w for w in words)


def print_start_banner(logger: MiaLogger, sess_path: Path, sess: Dict[str, Any], model: str, host: str) -> None:
    logger.banner("Mia-Chat bereit.")
    logger.banner("Befehle: help | clear | exit")
    logger.banner(f"Session: {sess_path}")
    logger.banner(f"Ollama: {host} | Model: {model}")

    last_sum = ((sess.get("summary", {}) or {}).get("last", "") or "").strip()
    if last_sum:
        logger.banner("Letzte Zusammenfassung:")
        for ln in last_sum.splitlines():
            logger.banner(ln)


def main() -> int:
    global STOP

    logger = MiaLogger()

    if requests is None:
        logger.log("FATAL: requests missing. Install in venv: pip install requests", category="ERROR")
        return 2

    # Ollama config
    ollama_host = env_str("MIA_OLLAMA_HOST", "http://127.0.0.1:11434")
    model = env_str("MIA_LLM", "mia-talk-72b:latest")
    keepalive = env_str("MIA_KEEPALIVE", "-1m")

    # Session
    sess_path = pick_session_path(logger)
    ensure_dir(sess_path.parent)
    sess = load_session(sess_path)
    ensure_system_prompt(sess)
    save_session(sess_path, sess)

    print_start_banner(logger, sess_path, sess, model=model, host=ollama_host)

    ok, reason = ollama_check_model(ollama_host, model)
    if ok:
        logger.banner("LLM Status: OK (Modell gefunden).")
    else:
        logger.log(f"LLM Status: NICHT OK ({reason}). Prüfe Modellname / ollama list.", category="ERROR")

    # input mode
    no_prompt = env_bool("MIA_CHAT_NO_PROMPT", False) or (not sys.stdin.isatty())

    exitwords = env_str("MIA_EXITWORDS", "exit,quit,ende,mia ende").strip()
    helpwords = env_str("MIA_HELPWORDS", "help,?,hilfe").strip()
    clearwords = env_str("MIA_CLEARWORDS", "clear,cls,reset").strip()

    while not STOP:
        logger.log("Listening...", category="STATE")

        try:
            if no_prompt:
                line = sys.stdin.readline()
                if line == "":
                    break

                user_in = line.rstrip("\n")

                # --- Block Paste Mode ---
                if user_in.strip() == "PASTE_BEGIN":
                    buf: List[str] = []
                    while True:
                        l2 = sys.stdin.readline()
                        if l2 == "":
                            # EOF while pasting -> treat as end
                            break
                        l2 = l2.rstrip("\n")
                        if l2.strip() == "PASTE_END":
                            break
                        buf.append(l2)
                    user_in = "\n".join(buf).strip()
                else:
                    user_in = user_in.strip()

            else:
                user_in = input("> ").strip()

        except EOFError:
            break
        except KeyboardInterrupt:
            STOP = True
            break

        if not user_in:
            continue


        if is_command(user_in, exitwords):
            break

        if is_command(user_in, helpwords):
            logger.banner("help  -> zeigt Hilfe")
            logger.banner("clear -> Session zurücksetzen (Datei bleibt)")
            logger.banner("exit  -> beendet")
            continue

        if is_command(user_in, clearwords):
            sess["messages"] = []
            ensure_system_prompt(sess)
            save_session(sess_path, sess)
            logger.banner("(Session zurückgesetzt)")
            continue

        logger.console_dialog_line("Du:", user_in)
        logger.log(f"Du: {user_in}", category="DEBUG")
        sess["messages"].append({"role": "user", "content": user_in})
        save_session(sess_path, sess)

        logger.log("Thinking...", category="STATE")

        assistant = ""
        try:
            assistant = ollama_chat_http_stream(
                ollama_host, model, keepalive, sess["messages"], logger, on_delta=None
            )
        except Exception as e:
            logger.log(f"Ollama ERROR: {e}", category="ERROR")
            assistant = ""

        assistant = (assistant or "").strip() or "Ich habe gerade keine Antwort. Bitte wiederhole das."
        logger.console_dialog_line("Mia:", assistant)
        logger.log(f"Mia: {assistant}", category="DEBUG")

        sess["messages"].append({"role": "assistant", "content": assistant})
        save_session(sess_path, sess)

    # exit summary
    try:
        if env_bool("MIA_SUMMARY", True):
            s = generate_session_summary(sess, ollama_host, model, keepalive, logger)
            if s.strip():
                sess["summary"] = {"last": s, "updated_at": now_iso()}
                save_session(sess_path, sess)
                append_global_summary(sess_path, s, logger)
                logger.banner("Session-Zusammenfassung gespeichert.")
    except Exception as e:
        logger.log(f"Summary ERROR on exit: {e}", category="DEBUG")

    logger.banner("Mia-Chat beendet.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        tb = traceback.format_exc()
        try:
            print(f"FATAL UNCAUGHT: {e}\n{tb}", flush=True)
        except Exception:
            pass
        try:
            log_dir = Path(env_str("MIA_LOG_DIR", "/data/mia/memory/logs"))
            p = log_dir / "chat" / "chat.log"
            ensure_dir(p.parent)
            with p.open("a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] FATAL UNCAUGHT: {e}\n{tb}\n")
        except Exception:
            pass
        raise
