import os, sys, time, tempfile, wave, subprocess, numpy as np, sounddevice as sd, requests
from faster_whisper import WhisperModel

# ---------- Config ----------
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://192.168.1.60:11435").rstrip("/")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MODE_DEFAULT  = os.getenv("MODE", "auto")  # auto|casual|tutor
VOICE         = os.getenv("VOICE", "amy")
ESPEAK_DATA   = "/usr/share/espeak-ng-data"
TIMEOUT_S     = int(os.getenv("OLLAMA_TIMEOUT", "60"))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BEEP_START  = os.path.join(CURRENT_DIR, "assets", "bip.wav")
BEEP_END    = os.path.join(CURRENT_DIR, "assets", "bip2.wav")

def tnow() -> float:
    return time.perf_counter()

# ---------- TTS ----------
def say(text: str, voice: str = VOICE):
    model = f"/opt/voices/en_US-{voice}-medium.onnx"
    cfg   = f"/opt/voices/en_US-{voice}-medium.onnx.json"
    p1 = subprocess.Popen(
        ["sudo","docker","exec","-i","piper-tts","sh","-lc",
         f"/opt/piper/build/piper -q -m {model} -c {cfg} --espeak_data {ESPEAK_DATA} -f /dev/stdout"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    p2 = subprocess.Popen(["aplay"], stdin=p1.stdout)
    if p1.stdin:
        p1.stdin.write(text.encode("utf-8")); p1.stdin.close()
    p2.communicate()

def play_wav(path: str):
    if os.path.isfile(path):
        subprocess.run(["aplay", path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------- Record ----------
def record_wav(path, seconds=5.0, fs=16000, cue=True):
    if cue:
        try: say("ready", voice=VOICE)
        except Exception: pass
    play_wav(BEEP_START)
    audio = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(fs); f.writeframes(audio.tobytes())
    play_wav(BEEP_END)

# ---------- ASR ----------
def build_model():
    try:
        m = WhisperModel("small.en", device="cuda", compute_type="float16")
        print("[asr] using CUDA float16")
        return m
    except Exception as e:
        print(f"[asr] CUDA init failed ({e}); using CPU int8")
        return WhisperModel("small.en", device="cpu", compute_type="int8")

def asr(path, model):
    use_vad = os.getenv("ASK_VAD", "1") != "0"
    segs, _ = model.transcribe(path, beam_size=1, vad_filter=use_vad)
    return "".join(s.text for s in segs).strip()

# ---------- Modes & router ----------
PROMPTS = {
    "casual": "You are brief and friendly. Reply in 1-2 sentences, plain language. Avoid extra detail.",
    "tutor":  "You are a patient tutor for kids. Explain clearly with a short summary first, then 2-3 key facts.",
}
LIMITS = {"casual": 80, "tutor": 300}

def pick_mode_auto(q: str) -> str:
    t = (q or "").lower()
    if any(t.startswith(x) for x in ("why", "how", "explain", "teach")): return "tutor"
    if "joke" in t or "funny" in t: return "casual"
    # more intents later (weather, math, etc.)
    return "casual"

# ---------- LLM (Ollama /api/chat) ----------
def ask_ollama_chat(prompt_text: str, mode: str) -> str:
    system_prompt = PROMPTS[mode]
    num_predict   = LIMITS[mode]
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt_text}
        ],
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        },
        "keep_alive": "30m"
    }
    r = requests.post(OLLAMA_HOST + "/api/chat", json=payload, timeout=TIMEOUT_S)
    if r.ok:
        j = r.json()
        # formats: {message:{content:...}} or {content:...}
        if isinstance(j, dict):
            if "message" in j and isinstance(j["message"], dict) and "content" in j["message"]:
                return str(j["message"]["content"]).strip()
            if "content" in j:
                return str(j["content"]).strip()
        return r.text.strip()
    return f"(Ollama chat HTTP {r.status_code}: {r.text[:200]})"

# ---------- Main ----------
if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    mode_arg = (sys.argv[2] if len(sys.argv) > 2 else MODE_DEFAULT).lower()
    model = build_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        path = t.name

    t0 = tnow()
    record_wav(path, seconds=seconds, cue=True); t_rec = tnow()

    q = asr(path, model); t_asr = tnow()
    print("[heard]", q if q else "(silence)")
    if not q:
        try: say("Sorry, I didn’t catch that.", voice=VOICE)
        except: pass
        print(f"[timings] rec={t_rec-t0:.2f}s asr={t_asr-t_rec:.2f}s llm=0.00s tts=0.00s total={t_asr-t0:.2f}s")
        sys.exit(0)

    mode = pick_mode_auto(q) if mode_arg == "auto" else ("tutor" if mode_arg=="tutor" else "casual")
    print(f"[mode] {mode}")

    a = ask_ollama_chat(q, mode); t_llm = tnow()
    print("[reply]", a)

    t_tts_start = tnow()
    if a and not a.startswith("("):
        try: say(a, voice=VOICE)
        except Exception as e:
            print(f"[tts] error: {e}")
    t_tts = tnow()

    print(f"[timings] rec={t_rec-t0:.2f}s asr={t_asr-t_rec:.2f}s llm={t_llm-t_asr:.2f}s tts={t_tts-t_tts_start:.2f}s total={t_tts-t0:.2f}s")
