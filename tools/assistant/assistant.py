# tools/assistant/assistant.py
import os
import sys
import re
import json
import time
import wave
import queue
import random
import pathlib
import datetime
import tempfile
import threading
import subprocess
import importlib.util

import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
import requests

# =========================
# Defaults (env can override)
# =========================
DEFAULT_LLAMA_URL    = "http://127.0.0.1:8080"
DEFAULT_OLLAMA_HOST  = "http://192.168.1.60:11435"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
LLAMA_URL    = os.getenv("LLAMA_URL", "").strip().rstrip("/")

if OLLAMA_HOST:
    BACKEND = "ollama"
    CHAT_URL = f"{OLLAMA_HOST}/api/chat"
elif LLAMA_URL:
    BACKEND = "llamacpp"
    CHAT_URL = f"{LLAMA_URL or DEFAULT_LLAMA_URL}/completion"
else:
    BACKEND = "llamacpp"
    CHAT_URL = f"{DEFAULT_LLAMA_URL}/completion"

MODE_DEFAULT = os.getenv("MODE", "casual").lower()  # casual|tutor|auto
VOICE        = os.getenv("VOICE", "amy")

# ---- VAD & timing ----
FS = 16000
FRAME_MS         = 20
BLOCK_FRAMES     = int(FS * FRAME_MS / 1000)  # 320
VAD_LEVEL        = int(os.getenv("VAD_LEVEL", "2"))            # 0..3 (higher = more aggressive)
SILENCE_AFTER_MS = int(os.getenv("SILENCE_AFTER_MS", "2000"))  # end utterance after this much silence
MIN_UTTER_MS     = int(os.getenv("MIN_UTTER_MS", "350"))
MAX_UTTER_MS     = int(os.getenv("MAX_UTTER_MS", "12000"))
PRE_ROLL_MS      = 120

# Mute mic during/after TTS
POST_TTS_IGNORE_MS = int(os.getenv("POST_TTS_IGNORE_MS", "300"))
TTS_ACTIVE = threading.Event()
_post_tts_ignore_until = 0.0

# Session idle → reconfirm after N seconds
SESSION_IDLE_SEC = int(os.getenv("SESSION_IDLE_SEC", "90"))

# Piper in Docker
PIPER_CONT  = os.getenv("PIPER_CONT", "piper-tts")
ESPEAK_DATA = "/usr/share/espeak-ng-data"

# =========================
# Logging (JSONL)
# =========================
LOG_DIR = os.path.expanduser("~/.sophia/logs")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
SESSION_LOG = os.path.join(LOG_DIR, f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

def jlog(kind, **payload):
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "kind": kind, **payload}
    try:
        with open(SESSION_LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    if kind in ("heard", "said", "error"):
        txt = payload.get("text", "")
        if txt:
            print(f"[{kind}] {txt}")

# =========================
# Router prompt & parsing
# =========================
DEFAULT_SYSTEM_PROMPT = (
    "You are Sophia, a kind, playful companion for kids. "
    "Speak simply (short sentences), warm and encouraging.\n"
    "Safety: be age-appropriate and avoid sensitive topics. If unsure, gently redirect.\n"
    "Dialogue management:\n"
    "- If the user gives name/age/city/pronouns, extract and call tool save_person.\n"
    "- 'be quiet' → set_quiet(on=true); 'hey sophia'/'i'm back' → set_quiet(on=false).\n"
    "- Choose MODE: 'casual' (1–3 short sentences) vs 'tutor' (short answer + a few facts).\n"
    "- If the user asks to be taught ('please teach me/tutor me/explain…'), prefer mode='tutor'.\n"
    "Style: Reply in kid-friendly plain text. Do NOT include stage directions or formatting "
    "like *smiles*, [laughs], parentheses-as-asides, markdown, or code fences.\n"
    "Return ONLY a single fenced JSON block: "
    "```json {\"speak\":\"...\",\"mode\":\"casual|tutor\",\"actions\":[...] } ```"
)

TOOL_JSON_RX     = re.compile(r"```(?:json|jsonc|json5)?\s*(\{.*?\})\s*```", re.I | re.S)
CODEBLOCK_ANY_RX = re.compile(r"```(?:\w+)?\s*([\s\S]*?)\s*```", re.S)

def _extract_json_object(text: str) -> dict | None:
    m = TOOL_JSON_RX.search(text or "")
    blob = m.group(1) if m else None
    if blob is None:
        m2 = CODEBLOCK_ANY_RX.search(text or "")
        if m2:
            inner = m2.group(1).strip()
            if inner.startswith("{") and inner.endswith("}"):
                blob = inner
    if blob is None and text:
        try:
            start = text.index("{")
            end = text.rindex("}")
            blob = text[start:end+1]
        except ValueError:
            blob = None
    if blob:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

# =========================
# Stage-direction scrub + natural years
# =========================
CODEFENCE_RX       = re.compile(r"```[\s\S]*?```", re.S)
INLINE_CODE_RX     = re.compile(r"`([^`]{1,120})`")
ASTERISK_EMPH_RX   = re.compile(r"\*([^\*\n]{1,120})\*")
LEADING_STAGE_RX   = re.compile(r"^\s*(?:\*[^*]{1,60}\*|\[[^\]]{1,60}\]|\([^)]{1,60}\))\s*[,:\-–—]*\s*", re.U)
MID_STAGE_WORDS_RX = re.compile(r"\s*(\[(smiles|laughs|shrugs|sighs|grins|chuckles|gasps|waves)\]|\((smiles|laughs|shrugs|sighs|grins|chuckles|gasps|waves)\))\s*", re.I)

def prepare_tts_text(s: str) -> str:
    if not s:
        return s
    s = CODEFENCE_RX.sub("", s)
    s = INLINE_CODE_RX.sub(r"\1", s)
    s = LEADING_STAGE_RX.sub("", s)
    s = ASTERISK_EMPH_RX.sub(r"\1", s)
    s = MID_STAGE_WORDS_RX.sub(" ", s)
    # 1100–1999 → "14 92" style (fourteen ninety two, eighteen twelve, etc.)
    s = re.sub(r"\b(1[1-9])(\d{2})\b", r"\1 \2", s)
    # 1901–1909 → "19 oh 5"
    s = re.sub(r"\b(19)0([1-9])\b", r"\1 oh \2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# Reconfirmation lines
# =========================
RECONFIRM_OPENERS = [
    "Sorry, are you {name}? If not, what's your name?",
    "Oh, are you {name}? If not, what's your name?",
    "Is that you, {name}? If not, tell me your name.",
]
RECONFIRM_RETRY = [
    "Is that you, {name}? Or tell me your name.",
    "Are you {name}? If not, please say your name.",
]
RECONFIRM_YES_ACK = [
    "Great, welcome back! What would you like to do?",
    "Awesome—good to hear you again! What shall we do?",
]
RECONFIRM_START_NEW = [
    "Okay! Let’s get to know each other.",
    "No problem—let’s start fresh.",
]

def pick_line(pool, name):
    return random.choice(pool).format(name=name)

# =========================
# TTS with hard mute
# =========================
def say(text: str, voice: str = VOICE):
    """Speak via Piper and hard-mute mic during/after output."""
    global _post_tts_ignore_until
    model = f"/opt/voices/en_US-{voice}-medium.onnx"
    cfg   = f"/opt/voices/en_US-{voice}-medium.onnx.json"
    TTS_ACTIVE.set()
    try:
        p1 = subprocess.Popen(
            ["sudo", "docker", "exec", "-i", PIPER_CONT, "sh", "-lc",
             f"/opt/piper/build/piper -q -m {model} -c {cfg} --espeak_data {ESPEAK_DATA} -f /dev/stdout"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        p2 = subprocess.Popen(["aplay"], stdin=p1.stdout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p1.stdin:
            p1.stdin.write((text.strip() + "\n").encode("utf-8"))
            p1.stdin.close()
        p2.communicate()
    finally:
        TTS_ACTIVE.clear()
        _post_tts_ignore_until = time.monotonic() + (POST_TTS_IGNORE_MS / 1000.0)

def say_and_log(text: str):
    clean = prepare_tts_text(text)
    jlog("said", text=clean)
    try:
        say(clean)
    except Exception as e:
        jlog("error", where="say", text=str(e))

# =========================
# ASR
# =========================
def build_asr():
    try:
        m = WhisperModel("small.en", device="cuda", compute_type="float16")
        print("[asr] using CUDA float16")
        return m
    except Exception as e:
        print(f"[asr] GPU unavailable ({e}); using CPU int8")
        return WhisperModel("small.en", device="cpu", compute_type="int8")

# =========================
# Skills loader (new_person)
# =========================
def load_new_person_skill():
    try:
        from skills.new_person import NewPersonSkill, load_people, save_people
        return NewPersonSkill, load_people, save_people
    except ImportError:
        here = pathlib.Path(__file__).parent
        skill_path = here / "skills" / "new_person.py"
        spec = importlib.util.spec_from_file_location("skills.new_person_fallback", str(skill_path))
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)                 # type: ignore
        return mod.NewPersonSkill, mod.load_people, mod.save_people

NewPersonSkill, load_people, save_people = load_new_person_skill()

# =========================
# Intent helpers
# =========================
QUIET_PATTERNS = re.compile(r"\b(stop talking|be quiet|quiet please|hush|shush|shh?)\b", re.I)
WAKE_PATTERNS  = re.compile(r"\b(hey\s*sophia|sophia|wake up|okay\s*sophia|robot|i'?m back)\b", re.I)
TEACH_TUTOR_RX = re.compile(r"\b(please\s+(teach|tutor)\s+me|teach\s+me|explain|help me learn)\b", re.I)
YES_RX = re.compile(r"\b(yes|yeah|yep|it'?s me|that'?s me|i am|i'm)\b", re.I)
NO_RX  = re.compile(r"\b(no|nope|not me|i'?m not|that'?s not me)\b", re.I)

def decide_mode(user_text: str, fallback="casual") -> str:
    if MODE_DEFAULT in ("casual", "tutor"):
        return MODE_DEFAULT
    if TEACH_TUTOR_RX.search(user_text or ""):
        return "tutor"
    lower = (user_text or "").lower().strip()
    if lower.startswith(("why", "how", "explain", "teach", "please")):
        return "tutor"
    return "casual"

# =========================
# LLM router
# =========================
def ask_llm_router(user_text: str, ctx) -> dict:
    system = os.getenv("SOPHIA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    mode_hint = decide_mode(user_text, "casual")

    if BACKEND == "ollama":
        req_payload = {
            "model": OLLAMA_MODEL or DEFAULT_OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_text},
            ],
            "stream": False,
            "options": {"num_predict": 360 if mode_hint == "tutor" else 240,
                        "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1},
            "keep_alive": "30m",
        }
        try:
            r = requests.post(f"{OLLAMA_HOST or DEFAULT_OLLAMA_HOST}/api/chat", json=req_payload, timeout=60)
            raw = (r.json().get("message", {}) or {}).get("content", "") if r.ok else ""
        except Exception as e:
            raw = f'{{"speak":"(LLM error: {e})","mode":"casual","actions":[]}}'
    else:
        prompt = f"{system}\n\nUser: {user_text}\nAssistant:"
        req_payload = {"prompt": prompt,
                       "n_predict": 360 if mode_hint == "tutor" else 240,
                       "temperature": 0.6}
        try:
            r = requests.post(CHAT_URL, json=req_payload, timeout=60)
            raw = r.json().get("content", "") if r.ok else ""
        except Exception as e:
            raw = f'{{"speak":"(LLM error: {e})","mode":"casual","actions":[]}}'

    jlog("router_raw", backend=BACKEND, request=req_payload, raw=raw[:1000])

    obj = _extract_json_object(raw)
    if not obj:
        speak = re.sub(r"```[\s\S]*?```", "", raw).strip() or "Okay."
        plan = {"speak": speak[:600], "mode": mode_hint, "actions": []}
    else:
        obj.setdefault("mode", mode_hint)
        if not isinstance(obj.get("actions"), list):
            obj["actions"] = obj.get("actions") if obj.get("actions") else []
        plan = obj

    jlog("router_plan", plan=plan)
    return plan

def normalize_skill_result(result):
    if isinstance(result, tuple):
        if len(result) >= 2:
            return str(result[0]), bool(result[1])
        elif len(result) == 1:
            return str(result[0]), False
        else:
            return "Okay.", True
    return str(result), False

# ---- normalize actions returned by LLM (robust to strings, json-in-string, etc.)
KNOWN_TOOL_NAMES = {"set_quiet", "save_person", "ask_sophia_info"}

def normalize_actions(actions):
    """Return a clean list of {'name': str, 'args': dict} items; skip junk."""
    out = []
    if not actions:
        return out
    if not isinstance(actions, list):
        actions = [actions]

    for i, a in enumerate(actions):
        if isinstance(a, dict):
            name = str(a.get("name", "")).strip()
            args = a.get("args") if isinstance(a.get("args"), dict) else {}
            if name:
                out.append({"name": name, "args": args})
            else:
                jlog("router_action_skip", idx=i, reason="empty-name", value=a)
            continue

        if isinstance(a, str):
            s = a.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict) and parsed.get("name"):
                        args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
                        out.append({"name": str(parsed["name"]).strip(), "args": args})
                        continue
                except Exception:
                    pass
            low = s.lower()
            if low in KNOWN_TOOL_NAMES:
                out.append({"name": low, "args": {}})
                continue

        jlog("router_action_skip", idx=i, reason="unsupported-type", value=a)
    return out

# =========================
# Main loop
# =========================
def main():
    asr = build_asr()
    vad = webrtcvad.Vad(VAD_LEVEL)

    ctx = {
        "active_skill": None,
        "people": load_people(),
        "current_person": None,
        "last_heard_ts": time.time(),
        "expecting_reconfirm": False,
        "reconfirm_name": None,
        "reconfirm_tries": 0,
        "quiet_mode": False,
        "forced_mode": None,
        "handoff_text": None,
    }

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        now = time.monotonic()
        if TTS_ACTIVE.is_set() or now < _post_tts_ignore_until:
            return  # hard mute while speaking or grace period
        q.put(bytes(indata))

    silence_need = max(1, int(SILENCE_AFTER_MS / FRAME_MS))
    min_voiced   = max(1, int(MIN_UTTER_MS / FRAME_MS))
    max_frames   = max(1, int(MAX_UTTER_MS / FRAME_MS))
    pre_frames   = max(0, int(PRE_ROLL_MS / FRAME_MS))

    state = "idle"
    buf = []
    preroll = []
    voiced = silence = total = 0

    print("[loop] listening. Say 'stop talking' to pause; 'hey sophia' to resume. Ctrl+C to exit.")
    jlog("session_start", backend=BACKEND, model=OLLAMA_MODEL if BACKEND == "ollama" else "llamacpp")

    try:
        say_and_log("ready")
    except Exception:
        pass

    with sd.InputStream(samplerate=FS, channels=1, dtype='int16',
                        blocksize=BLOCK_FRAMES, callback=callback):
        while True:
            data = q.get()
            is_speech = vad.is_speech(data, FS)

            if state == "idle":
                preroll.append(data)
                if len(preroll) > pre_frames:
                    preroll.pop(0)

            if state in ("idle", "collecting"):
                if is_speech:
                    if state == "idle":
                        state = "collecting"
                        buf = preroll[:]
                        voiced = silence = total = 0
                    buf.append(data)
                    voiced += 1
                    total += 1
                else:
                    if state == "collecting":
                        buf.append(data)
                        silence += 1
                        total += 1

                if state == "collecting":
                    too_long = total >= max_frames
                    long_enough = voiced >= min_voiced
                    ended = long_enough and silence >= silence_need
                    if ended or too_long:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
                            path = t.name
                        with wave.open(path, "wb") as w:
                            w.setnchannels(1)
                            w.setsampwidth(2)
                            w.setframerate(FS)
                            for fr in buf:
                                w.writeframes(fr)
                        state = "idle"
                        buf.clear()
                        preroll.clear()
                        voiced = silence = total = 0

                        # ASR
                        try:
                            t0 = time.perf_counter()
                            segs, _ = asr.transcribe(path, beam_size=1, vad_filter=False)
                            text = "".join(s.text for s in segs).strip()
                            t_asr = time.perf_counter() - t0
                        except RuntimeError as e:
                            if "cuDNN" in str(e):
                                jlog("warn", what="asr_cudnn", msg=str(e))
                                cpu = WhisperModel("small.en", device="cpu", compute_type="int8")
                                segs, _ = cpu.transcribe(path, beam_size=1, vad_filter=False)
                                text = "".join(s.text for s in segs).strip()
                                t_asr = -1.0
                            else:
                                jlog("error", where="asr", text=str(e))
                                text = ""
                                t_asr = -1.0
                        if not text:
                            continue
                        jlog("heard", text=text)

                        # quiet/wake
                        if ctx["quiet_mode"]:
                            if WAKE_PATTERNS.search(text):
                                ctx["quiet_mode"] = False
                                print("[mode] quiet OFF")
                                say_and_log("Hi! I'm listening again.")
                            continue
                        if QUIET_PATTERNS.search(text):
                            ctx["quiet_mode"] = True
                            print("[mode] quiet ON")
                            say_and_log("Okay, I'll be quiet. Say 'Hey Sophia' when you want me again.")
                            continue

                        # idle reconfirm
                        now = time.time()
                        idle = (now - ctx.get("last_heard_ts", now)) > SESSION_IDLE_SEC
                        ctx["last_heard_ts"] = now
                        if idle and ctx.get("current_person") and not ctx.get("expecting_reconfirm"):
                            last_name = ctx["current_person"].get("name") or "friend"
                            say_and_log(pick_line(RECONFIRM_OPENERS, last_name))
                            ctx["expecting_reconfirm"] = True
                            ctx["reconfirm_name"] = last_name
                            ctx["reconfirm_tries"] = 0
                            continue
                        if ctx.get("expecting_reconfirm"):
                            low = text.lower()
                            if YES_RX.search(low):
                                say_and_log(random.choice(RECONFIRM_YES_ACK))
                                ctx["expecting_reconfirm"] = False
                                continue
                            if NO_RX.search(low):
                                ctx["active_skill"] = NewPersonSkill()
                                say_and_log(random.choice(RECONFIRM_START_NEW))
                                time.sleep(0.2)
                                say_and_log(ctx["active_skill"].start("", ctx))
                                ctx["expecting_reconfirm"] = False
                                continue
                            try:
                                nm = NewPersonSkill().extract_name(text)
                            except Exception:
                                nm = None
                            if nm:
                                ctx["active_skill"] = NewPersonSkill()
                                say_and_log(ctx["active_skill"].start(text, ctx))
                                ctx["expecting_reconfirm"] = False
                                continue
                            if ctx["reconfirm_tries"] == 0:
                                ctx["reconfirm_tries"] = 1
                                say_and_log(pick_line(RECONFIRM_RETRY, ctx.get("reconfirm_name") or "friend"))
                                continue
                            ctx["active_skill"] = NewPersonSkill()
                            say_and_log(ctx["active_skill"].start("", ctx))
                            ctx["expecting_reconfirm"] = False
                            continue

                        # skill flow
                        if ctx["active_skill"]:
                            try:
                                result = ctx["active_skill"].step(text, ctx)
                                reply, done = normalize_skill_result(result)
                            except Exception as e:
                                jlog("error", where="skill_step", text=str(e))
                                reply, done = "Sorry, something went wrong.", True

                            if reply:
                                say_and_log(reply)
                            if done:
                                ctx["active_skill"] = None
                                if ctx.get("handoff_text"):
                                    text_for_llm = ctx.pop("handoff_text")
                                else:
                                    continue
                            else:
                                continue
                        else:
                            # Try to start new_person if greeting/name-like
                            probe = NewPersonSkill()
                            if probe.match(text, ctx) > 0:
                                ctx["active_skill"] = probe
                                say_and_log(probe.start(text, ctx))
                                continue
                            text_for_llm = text

                        # LLM Router
                        t1 = time.perf_counter()
                        plan = ask_llm_router(text_for_llm, ctx)

                        # Defensive plan normalization and actions handling
                        if not isinstance(plan, dict):
                            plan = {"speak": str(plan), "mode": decide_mode(text_for_llm, "casual"), "actions": []}

                        actions = normalize_actions(plan.get("actions"))

                        for a in actions:
                            name = (a.get("name") or "").lower()
                            args = a.get("args") or {}

                            if name == "set_quiet":
                                ctx["quiet_mode"] = bool(args.get("on", False))

                            elif name == "save_person":
                                try:
                                    db = load_people()
                                    me = ctx.get("current_person")
                                    if me and isinstance(db, list):
                                        found = False
                                        me_name = (me.get("name") or "").strip().lower()
                                        if me_name:
                                            for p in db:
                                                if (p.get("name", "").strip().lower() == me_name):
                                                    p.update(me)
                                                    found = True
                                                    break
                                        if not found:
                                            db.append(me)
                                        save_people(db)
                                except Exception as e:
                                    jlog("error", where="save_person", text=str(e))

                        # Build reply/mode safely
                        reply = plan.get("speak")
                        if not isinstance(reply, str):
                            try:
                                reply = json.dumps(reply)[:600]
                            except Exception:
                                reply = "Okay."
                        reply = (reply or "Okay.").strip()

                        chosen_mode = ctx.get("forced_mode") or plan.get("mode") or decide_mode(text_for_llm, "casual")
                        ctx["forced_mode"] = None

                        t_llm = time.perf_counter() - t1
                        jlog("router", mode=chosen_mode, actions=actions)

                        say_and_log(reply)
                        total_t = (t_asr if t_asr >= 0 else 0.0) + t_llm
                        t_asr_str = f"{t_asr:.2f}" if t_asr >= 0 else "cpu"
                        print(f"[timings] asr={t_asr_str}s llm={t_llm:.2f}s total≈{total_t:.2f}s")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[loop] stopped.")
        sys.exit(0)
