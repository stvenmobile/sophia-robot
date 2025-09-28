import os, sys, tempfile, wave, subprocess, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel
import requests

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080/completion")

def record_wav(path, seconds=5, fs=16000):
    audio = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    with wave.open(path,"wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(fs); f.writeframes(audio.tobytes())

def asr(path):
    model = WhisperModel("small.en", device="cuda", compute_type="float16")
    segs, _ = model.transcribe(path, beam_size=1, vad_filter=True)
    return "".join(s.text for s in segs).strip()

def ask_llama(text):
    r = requests.post(LLAMA_URL, json={"prompt": text, "max_tokens": 256, "temperature": 0.6}, timeout=30)
    return r.json().get("content") if r.ok else f"(LLM {r.status_code})"

def say(text, voice="amy"):
    model = f"/opt/voices/en_US-{voice}-medium.onnx"
    cfg   = f"/opt/voices/en_US-{voice}-medium.onnx.json"
    p1 = subprocess.Popen(
        ["sudo","docker","exec","-i","piper-tts","sh","-lc",
         f"/opt/piper/build/piper -m {model} -c {cfg} --espeak_data /usr/share/espeak-ng-data -f /dev/stdout"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["aplay"], stdin=p1.stdout)
    p1.stdin.write(text.encode("utf-8")); p1.stdin.close(); p2.communicate()

if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv)>1 else 5.0
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t: path=t.name
    record_wav(path, seconds=secs)
    q = asr(path); print("[heard]", q)
    a = ask_llama(q); print("[reply]", a)
    if a: say(a, voice="amy")
