"""
Sophia Assistant (Jetson Orin Nano)
Speech -> Text (Whisper) -> RAG -> Local LLaMA -> Text -> Speech (Piper in Docker)

Requirements (pip):
  openai-whisper, requests, sounddevice, numpy==1.24.4, faiss-cpu, sentence-transformers, torch

Docker:
  Container named "piper-tts" running the Piper build, with voices mounted at /opt/voices
  and espeak-ng data available at /usr/share/espeak-ng-data.

Environment overrides (optional):
  LLAMA_URL           (default: http://127.0.0.1:8080/completion)
  PIPER_CONTAINER     (default: piper-tts)
  PIPER_DEFAULT_VOICE (default: amy)
"""

import os
import wave
import tempfile
import subprocess
import numpy as np
import requests
import sounddevice as sd
import torch
import faiss  # faiss-cpu
from sentence_transformers import SentenceTransformer
import whisper

# -----------------------------
# Config
# -----------------------------
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080/completion")
PIPER_CONTAINER = os.getenv("PIPER_CONTAINER", "piper-tts")
DEFAULT_VOICE = os.getenv("PIPER_DEFAULT_VOICE", "amy").lower()

INITIAL_PROMPT = (
    "You're Sophia, an AI assistant specialized in robotics, embedded systems (Jetson/ROS2), "
    "and practical maker workflows. Be concise, friendly, and direct. If unsure, say so and "
    "suggest a next step."
)

# Paths for beep sounds
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BEEP_START = os.path.join(CURRENT_DIR, "assets", "bip.wav")
BEEP_END = os.path.join(CURRENT_DIR, "assets", "bip2.wav")

# Piper voices inside the container
VOICE_MAP = {
    "amy": ("/opt/voices/en_US-amy-medium.onnx",
            "/opt/voices/en_US-amy-medium.onnx.json"),
    "kristin": ("/opt/voices/en_US-kristin-medium.onnx",
                "/opt/voices/en_US-kristin-medium.onnx.json"),
}

ESPEAK_DATA_DIR = "/usr/share/espeak-ng-data"  # ensured by Dockerfile

# -----------------------------
# Models (load once)
# -----------------------------
# Embeddings (384-dim)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Torch device (Jetson should report CUDA available if NVIDIA wheel installed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Whisper ASR model
whisper_model = whisper.load_model("base").to(device)

# -----------------------------
# Tiny RAG (demo)
# -----------------------------
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
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.documents[i] for i in indices[0] if 0 <= i < len(self.documents)]

docs = [
    "The Jetson Orin Nano runs CUDA-accelerated PyTorch builds for efficient on-device ASR/TTS.",
    "Sophia uses Whisper for transcription and Piper for speech synthesis.",
    "Keep responses concise and practical for robotics workflows.",
]
db = VectorDatabase()
db.add_documents(docs)

# -----------------------------
# Audio helpers
# -----------------------------
def play_sound(path: str):
    """Play a short WAV via aplay if file exists."""
    if os.path.isfile(path):
        try:
            subprocess.run(["aplay", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def record_audio(filename: str, duration: float = 5.0, fs: int = 16000):
    """Record mono PCM16 audio to filename."""
    play_sound(BEEP_START)
    print(f"[rec] recording {duration:.1f}s @ {fs} Hz …")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    play_sound(BEEP_END)
    print("[rec] done")

# -----------------------------
# ASR / LLM / RAG
# -----------------------------
def transcribe_audio(filename: str) -> str:
    try:
        out = whisper_model.transcribe(filename, language="en")
        return (out.get("text") or "").strip()
    except Exception as e:
        print(f"[asr] error: {e}")
        return ""

def ask_llama(query: str, context: str) -> str:
    """Post a prompt to local LLaMA server; handle a few common response formats."""
    prompt = f"{INITIAL_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    data = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.6,
    }
    try:
        r = requests.post(LLAMA_URL, json=data, headers={"Content-Type": "application/json"}, timeout=30)
        if r.status_code != 200:
            return f"(LLM error {r.status_code})"
        j = r.json()
        # Try common fields
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
        # Fallback to raw text
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
    """Speak text using Piper inside the Docker container, streaming to aplay."""
    voice = (voice or DEFAULT_VOICE).lower()
    model_path, config_path = VOICE_MAP.get(voice, VOICE_MAP["amy"])
    cmd = [
        "sudo", "docker", "exec", "-i", PIPER_CONTAINER,
        "sh", "-lc",
        f"/opt/piper/build/piper -m {model_path} -c {config_path} "
        f"--espeak_data {ESPEAK_DATA_DIR} -f /dev/stdout"
    ]
    try:
        p1 = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["aplay"], stdin=p1.stdout)
        # Feed text
        if p1.stdin:
            p1.stdin.write(text.encode("utf-8"))
            p1.stdin.close()
        # Wait for playback to finish
        p2.communicate()
    except FileNotFoundError:
        print("[tts] aplay not found on host; please install 'alsa-utils'.")
    except Exception as e:
        print(f"[tts] error: {e}")

# -----------------------------
# Main loop
# -----------------------------
def main(record_seconds: float = 5.0, voice: str = DEFAULT_VOICE):
    print(f"[sophia] device={device}, whisper=base, voice={voice}")
    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            record_audio(wav_path, duration=record_seconds)
            text = transcribe_audio(wav_path)
            print(f"[heard] {text}")
            if not text:
                continue

            # Simple exit intents
            if text.strip().lower() in {"quit", "exit", "stop", "cancel"}:
                text_to_speech("Okay, stopping now.", voice=voice)
                break

            reply = rag_ask(text)
            print(f"[reply] {reply}")
            if reply:
                text_to_speech(reply, voice=voice)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Optional: read basic args from env
    try:
        secs = float(os.getenv("SOPHIA_RECORD_SECONDS", "5.0"))
    except Exception:
        secs = 5.0
    main(record_seconds=secs, voice=DEFAULT_VOICE)
