"""
Sophia Assistant (faster-whisper edition, Jetson-friendly)

Speech -> Text (faster-whisper) -> Tiny RAG -> Local LLaMA -> Text -> Speech (Piper in Docker)

Quick install (venv):
  pip install --upgrade pip
  pip install faster-whisper requests sounddevice numpy==1.24.4 faiss-cpu sentence-transformers

Run examples:
  python assistant_fw.py                      # defaults: small.en, cuda->float16, Amy, 5s record
  python assistant_fw.py --model medium.en    # higher accuracy
  python assistant_fw.py --voice kristin
  python assistant_fw.py --seconds 4
  python assistant_fw.py --device cpu --compute-type int8  # CPU fallback

Notes:
- Requires a running Docker container named "piper-tts" with voices mounted at /opt/voices
- Expects espeak-ng data in the container at /usr/share/espeak-ng-data (per your Dockerfile tweak)
- LLaMA server URL can be overridden via env var LLAMA_URL
"""

import os
import wave
import argparse
import tempfile
import subprocess
import numpy as np
import requests
import sounddevice as sd
import faiss  # faiss-cpu
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

# -----------------------------
# Config (env overrides)
# -----------------------------
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080/completion")
PIPER_CONTAINER = os.getenv("PIPER_CONTAINER", "piper-tts")
DEFAULT_VOICE = os.getenv("PIPER_DEFAULT_VOICE", "amy").lower()
ESPEAK_DATA_DIR = "/usr/share/espeak-ng-data"  # ensured by Dockerfile

INITIAL_PROMPT = (
    "You're Sophia, an AI assistant specialized in robotics, embedded systems (Jetson/ROS2), "
    "and practical maker workflows. Be concise, friendly, and direct. If unsure, say so and "
    "suggest a next step."
)

# Beep assets (optional)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BEEP_START = os.path.join(CURRENT_DIR, "assets", "bip.wav")
BEEP_END = os.path.join(CURRENT_DIR, "assets", "bip2.wav")

# Piper voices inside the container
VOICE_MAP = {
    "amy": ("/opt/voices/en_US-amy-medium.onnx", "/opt/voices/en_US-amy-medium.onnx.json"),
    "kristin": ("/opt/voices/en_US-kristin-medium.onnx", "/opt/voices/en_US-kristin-medium.onnx.json"),
}

# -----------------------------
# Tiny RAG setup
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class VectorDatabase:
    def __init__(self, dim: int = 384):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add_documents(self, docs):
        if not docs:
            return
        embeddings = embedding_model.encode(docs)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(docs)

    def search(self, query: str, top_k: int = 3):
        if not self.documents:
            return []
        q = embedding_model.encode([query])[0].astype(np.float32)
        _, idx = self.index.search(np.array([q]), top_k)
        return [self.documents[i] for i in idx[0] if 0 <= i < len(self.documents)]

docs = [
    "The Jetson Orin Nano runs CUDA-accelerated CTranslate2 for faster-whisper.",
    "Sophia uses faster-whisper for transcription and Piper (Dockerized) for speech.",
    "Keep responses concise and practical for robotics workflows.",
]
db = VectorDatabase()
db.add_documents(docs)

# -----------------------------
# Audio helpers
# -----------------------------
def play_sound(path: str):
    if os.path.isfile(path):
        try:
            subprocess.run(["aplay", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def record_audio(filename: str, duration: float = 5.0, fs: int = 16000):
    play_sound(BEEP_START)
    print(f"[rec] recording {duration:.1f}s @ {fs} Hz …")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    play_sound(BEEP_END)
    print("[rec] done")

# -----------------------------
# ASR (faster-whisper)
# -----------------------------
def build_whisper_model(name: str, device_pref: str, compute_type: str) -> WhisperModel:
    """
    device_pref: "cuda"|"cpu"|"auto"
    compute_type: e.g., "float16" (Jetson GPU), "int8_float16", "int8", etc.
    """
    if device_pref == "cpu":
        return WhisperModel(name, device="cpu", compute_type=("int8" if compute_type == "float16" else compute_type))
    if device_pref == "cuda":
        try:
            return WhisperModel(name, device="cuda", compute_type=compute_type)
        except Exception as e:
            print(f"[asr] CUDA init failed: {e}; falling back to CPU int8")
            return WhisperModel(name, device="cpu", compute_type="int8")
    # auto
    try:
        return WhisperModel(name, device="cuda", compute_type=compute_type)
    except Exception as e:
        print(f"[asr] CUDA init failed: {e}; using CPU int8")
        return WhisperModel(name, device="cpu", compute_type="int8")

def transcribe_audio_fw(model: WhisperModel, filename: str, beam_size: int = 1) -> str:
    try:
        segments, info = model.transcribe(filename, beam_size=beam_size, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
        return text
    except Exception as e:
        print(f"[asr] error: {e}")
        return ""

# -----------------------------
# LLM / RAG
# -----------------------------
def ask_llama(query: str, context: str) -> str:
    prompt = f"{INITIAL_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    data = {"prompt": prompt, "max_tokens": 512, "temperature": 0.6}
    try:
        r = requests.post(LLAMA_URL, json=data, headers={"Content-Type": "application/json"}, timeout=30)
        if r.status_code != 200:
            return f"(LLM error {r.status_code})"
        j = r.json()
        if isinstance(j, dict):
            if "content" in j and isinstance(j["content"], str):
                return j["content"].strip()
            if "choices" in j and j["choices"]:
                c0 = j["choices"][0]
                if isinstance(c0, dict):
                    if "text" in c0:
                        return str(c0["text"]).strip()
                    if "message" in c0 and isinstance(c0["message"], dict) and "content" in c0["message"]:
                        return str(c0["message"]["content"]).strip()
        return r.text.strip()
    except Exception as e:
        print(f"[llm] error: {e}")
        return "Sorry, I had trouble contacting the local model."

def rag_ask(query: str) -> str:
    ctx = " ".join(db.search(query)) if query else ""
    return ask_llama(query, ctx)

# -----------------------------
# TTS via Dockerized Piper
# -----------------------------
def text_to_speech(text: str, voice: str = DEFAULT_VOICE):
    voice = (voice or DEFAULT_VOICE).lower()
    model_path, config_path = VOICE_MAP.get(voice, VOICE_MAP["amy"])
    cmd = [
        "sudo", "docker", "exec", "-i", PIPER_CONTAINER,
        "sh", "-lc",
        f"/opt/piper/build/piper -m {model_path} -c {config_path} --espeak_data {ESPEAK_DATA_DIR} -f /dev/stdout"
    ]
    try:
        p1 = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["aplay"], stdin=p1.stdout)
        if p1.stdin:
            p1.stdin.write(text.encode("utf-8"))
            p1.stdin.close()
        p2.communicate()
    except FileNotFoundError:
        print("[tts] aplay not found on host; please install 'alsa-utils'.")
    except Exception as e:
        print(f"[tts] error: {e}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Sophia assistant (faster-whisper)")
    ap.add_argument("--model", default=os.getenv("FW_MODEL", "small.en"),
                    help="faster-whisper model name (e.g., tiny.en, small.en, medium.en)")
    ap.add_argument("--device", default=os.getenv("FW_DEVICE", "auto"),
                    choices=["auto", "cuda", "cpu"], help="prefer CUDA if available (auto)")
    ap.add_argument("--compute-type", default=os.getenv("FW_COMPUTE_TYPE", "float16"),
                    help="faster-whisper compute_type (e.g., float16, int8_float16, int8)")
    ap.add_argument("--beam-size", type=int, default=int(os.getenv("FW_BEAM_SIZE", "1")))
    ap.add_argument("--seconds", type=float, default=float(os.getenv("SOPHIA_RECORD_SECONDS", "5.0")))
    ap.add_argument("--voice", default=DEFAULT_VOICE, choices=list(VOICE_MAP.keys()))
    args = ap.parse_args()

    print(f"[sophia-fw] model={args.model}, device={args.device}, compute_type={args.compute_type}, "
          f"beam_size={args.beam_size}, voice={args.voice}")

    # Build ASR model
    fw_model = build_whisper_model(args.model, args.device, args.compute_type)

    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            record_audio(wav_path, duration=args.seconds)
            text = transcribe_audio_fw(fw_model, wav_path, beam_size=args.beam_size)
            print(f"[heard] {text}")
            if not text:
                continue

            # Exit intents
            if text.strip().lower() in {"quit", "exit", "stop", "cancel"}:
                text_to_speech("Okay, stopping now.", voice=args.voice)
                break

            reply = rag_ask(text)
            print(f"[reply] {reply}")
            if reply:
                text_to_speech(reply, voice=args.voice)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()
