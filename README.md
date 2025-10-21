# 🤖 Sophia — A Voice Assistant Robot for Kids

[![Jetson Orin Nano](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-green)](#)
[![Status](https://img.shields.io/badge/Status-Prototype-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Sophia is a friendly **AI voice companion for kids**, designed to answer questions, tutor, and tell stories — fully **local** on a Jetson Orin Nano for privacy and low latency.

---

## ✨ What works today (Phase 1 MVP)

- **TTS**: Piper running in Docker with two voices (Amy default, Kristin optional)
- **ASR**:
  - `assistant.py` → OpenAI Whisper (baseline)
  - `assistant_fw.py` → faster-whisper (GPU-optimized) with CLI flags
- **Tiny RAG → Local LLM**: sends the transcript + small context to a local LLM endpoint (`LLAMA_URL`)
- **Quick single-turn test**: `ask_once.py` (record → transcribe → LLM → speak)
- **Wake word (optional)**: `wakeword_assistant.py` using Porcupine (“computer”) for hands-free

---

## 🗂 Repo layout (relevant bits)
```text
tools/
└── assistant/
    ├── assets/
    │   ├── bip.wav
    │   └── bip2.wav
    ├── assistant.py            # Whisper baseline
    ├── assistant_fw.py         # faster-whisper + CLI
    ├── ask_once.py             # one-shot Q→A test
    ├── wakeword_assistant.py   # wake-word ("computer") → Q→A loop
    └── docker/
        └── piper/
            └── Dockerfile      # Piper build (with espeak-ng data fix)
```


---

## 🧱 Prerequisites

- Jetson Orin Nano with JetPack (CUDA/cuDNN installed)
- Python 3.10+
- ALSA utils on host: `sudo apt-get install -y alsa-utils`
- Docker installed and working
- A local LLM server reachable via `LLAMA_URL` (examples: llama.cpp server, an adapter around Ollama, etc.)

---

## 🔊 Piper TTS (Dockerized)

1) **Download voices + configs (host):**
```bash
mkdir -p ~/.local/share/piper/voices
# Amy
curl -L -o ~/.local/share/piper/voices/en_US-amy-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
curl -L -o ~/.local/share/piper/voices/en_US-amy-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"

# Kristin
curl -L -o ~/.local/share/piper/voices/en_US-kristin-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium/en_US-kristin-medium.onnx"
curl -L -o ~/.local/share/piper/voices/en_US-kristin-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium/en_US-kristin-medium.onnx.json"

Build & run the container:

```bash

cd tools/assistant/docker/piper
sudo docker build -t piper-tts-jetson .
sudo docker rm -f piper-tts 2>/dev/null || true
sudo docker run --name piper-tts -d \
  -v ~/.local/share/piper/voices:/opt/voices:ro \
  piper-tts-jetson sleep infinity
sudo docker update --restart unless-stopped piper-tts

```
The Dockerfile includes an espeak-ng data fix so phontab is found reliably.

Sanity check:

```bash
sudo docker exec piper-tts bash -lc \
  'echo "Hello from Amy." | /opt/piper/build/piper \
     -m /opt/voices/en_US-amy-medium.onnx \
     -c /opt/voices/en_US-amy-medium.onnx.json \
     --espeak_data /usr/share/espeak-ng-data \
     -f /dev/stdout' | aplay
```

## 🗣️ Python environment

```bash
cd tools/assistant
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# baseline requirements (Whisper path)
pip install -r requirements.txt
# faster-whisper path (optional)
pip install faster-whisper
```

## 🚀 Run it
One-shot Q→A test (fastest path):

```bash
source tools/assistant/.venv/bin/activate
python tools/assistant/ask_once.py 5
```

Continuous assistant (Whisper baseline):

```bash
python tools/assistant/assistant.py
faster-whisper variant (recommended on GPU):
```

```bash
# CPU fallback (works now)
python tools/assistant/assistant_fw.py --device cpu --compute-type int8

# After enabling CUDA in CTranslate2 (see below)
python tools/assistant/assistant_fw.py --device cuda --compute-type float16
```

Wake-word loop (Porcupine’s “computer”):


```bash
pip install pvporcupine pvrecorder
python tools/assistant/wakeword_assistant.py
```

say: "computer" → ask a question → get a spoken answer

⚡ Enable GPU for faster-whisper (CTranslate2 with CUDA)

If assistant_fw.py complains about CPU-only CTranslate2, build it with CUDA:

```bash
# deps
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev

# in your venv
source tools/assistant/.venv/bin/activate
pip uninstall -y ctranslate2

# build & install (compute 8.7 for Orin)
cd ~
git clone --depth 1 --branch v4.6.0 https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DCUDA_ARCH_LIST="8.7"
cmake --build build -j"$(nproc)"
pip install ./python

# verify
python - <<'PY'
import ctranslate2; print(ctranslate2.get_build_info())
PY
# look for: "cuda": true
```

🔌 LLM endpoint
Set LLAMA_URL to your local model server. Examples:

llama.cpp server exposing a /completion-style endpoint on http://127.0.0.1:8080

a small adapter that translates to/from Ollama’s API

You can export it before running:

```bash
export LLAMA_URL="http://127.0.0.1:8080/completion"
```

## 🧭 Roadmap (kid-friendly upgrades)
- Better wake-word: custom “Hey Sophia” (Porcupine custom / openWakeWord)

- VAD endpointing & barge-in: stop speaking if the child starts talking

- Tutor mode: style guide + guardrails for age-appropriate answers

- LED / face animations: visual cues for wake/listen/speak

- Memory (short-term): simple conversation state, reset on “new topic”

## 🔒 Safety & design
Local-first (no cloud) • Age-appropriate answers • Positive tone

Avoid adult/violent/scary topics by design prompts and filtering

Parents can review/update system prompts

## 📄 License
MIT — see LICENSE.
