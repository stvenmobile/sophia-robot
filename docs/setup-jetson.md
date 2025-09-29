### `docs/setup-jetson.md`

```markdown
# Jetson Orin Nano Setup

These steps set up **GPU ASR (faster-whisper via CTranslate2 + CUDA + cuDNN)**, **Piper TTS in Docker**, and an **LLM endpoint** (Ollama or local llama.cpp).

> Tested with CUDA 12.6 and cuDNN 9 on JetPack. Power mode: `MAXN_Super` recommended.

---

## 0) Pre-reqs (Jetson)

- JetPack with CUDA 12.x (`nvcc --version` → 12.x)
- cuDNN dev libs for the same CUDA (9.x on JetPack 6):
  ```bash
  dpkg -l | grep -i cudnn
  # expect libcudnn9-cuda-12 and libcudnn9-dev-cuda-12
Audio I/O working (arecord -l, aplay -l)

Optional performance:

bash
Copy code
sudo nvpmodel -m 0     # MAXN (if available)
sudo jetson_clocks     # max clocks during benchmarking
1) Piper TTS (Docker)
Voices (host)
bash
Copy code
mkdir -p ~/.local/share/piper/voices
# Amy
curl -L -o ~/.local/share/piper/voices/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -L -o ~/.local/share/piper/voices/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
# Kristin (optional)
curl -L -o ~/.local/share/piper/voices/en_US-kristin-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium/en_US-kristin-medium.onnx
curl -L -o ~/.local/share/piper/voices/en_US-kristin-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium/en_US-kristin-medium.onnx.json
Build & run container
bash
Copy code
cd tools/assistant/docker/piper
sudo docker build -t piper-tts-jetson .
sudo docker rm -f piper-tts 2>/dev/null || true
sudo docker run --name piper-tts -d \
  -v ~/.local/share/piper/voices:/opt/voices:ro \
  piper-tts-jetson sleep infinity
sudo docker update --restart unless-stopped piper-tts
Sanity check
bash
Copy code
sudo docker exec piper-tts bash -lc \
  'echo "Hello from Amy." | /opt/piper/build/piper \
     -m /opt/voices/en_US-amy-medium.onnx \
     -c /opt/voices/en_US-amy-medium.onnx.json \
     --espeak_data /usr/share/espeak-ng-data \
     -f /dev/stdout' | aplay
Note: Bad voice attribute: option is a harmless espeak-ng warning.

2) Python venv & deps
bash
Copy code
cd tools/assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt faster-whisper
sudo apt-get install -y libopenblas-dev
3) Build CTranslate2 with CUDA + cuDNN
This is required for GPU ASR (Conv1D uses cuDNN).

CUDA env
bash
Copy code
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
sudo ln -sfn "$CUDA_HOME" /usr/local/cuda
nvcc --version    # expect 12.6.x
Build & install CT2
bash
Copy code
cd ~
git clone --recursive --branch v4.6.0 https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON -DWITH_CUDNN=ON \
  -DCUDNN_INCLUDE_DIR=/usr/include \
  -DCUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so \
  -DOPENMP_RUNTIME=COMP -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_DNNL=OFF \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCUDA_ARCH_LIST="8.7"
cmake --build build -j"$(nproc)"
sudo cmake --install build
sudo ldconfig
Install Python bindings into venv
bash
Copy code
source ~/projects/sophia-robot/tools/assistant/.venv/bin/activate
pip uninstall -y ctranslate2 2>/dev/null || true
pip install ./python --no-build-isolation --no-deps --force-reinstall
Verify links (optional but useful)
bash
Copy code
python - <<'PY'
import inspect, ctranslate2, ctranslate2._ext as ext
print(inspect.getfile(ext))
PY

ldd /usr/local/lib/libctranslate2.so | egrep -i 'cudnn|cublas|cudart' || true
# Expect libcudnn.so, libcublas.so, libcublasLt.so, libcudart.so
Runtime test (definitive)
bash
Copy code
python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("tiny.en", device="cuda", compute_type="float16")
print("✅ faster-whisper CUDA+cuDNN load: OK")
PY
If you ever installed a local CT2 into ~/ct2-local, ensure your runtime resolves /usr/local/lib/libctranslate2.so (remove ~/ct2-local from LD_LIBRARY_PATH, or rename it).

4) LLM endpoint
A) Remote Ollama
bash
Copy code
export OLLAMA_HOST="http://<host>:11435"
export OLLAMA_MODEL="llama3.2:3b"   # or phi3.5:3.8b, etc.
# sanity:
curl -s $OLLAMA_HOST/api/version && echo
curl -s $OLLAMA_HOST/api/tags | jq .
B) Local llama.cpp (no network hop)
bash
Copy code
sudo apt-get install -y build-essential cmake
cd ~ && git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
LLAMA_CUBLAS=1 make -j
# place a small GGUF (e.g., 3B Q4_K_M) at ~/models/model.gguf
./server -m ~/models/model.gguf -ngl 99 --port 8080
# point assistant at it:
unset OLLAMA_HOST OLLAMA_MODEL
export LLAMA_URL="http://127.0.0.1:8080/completion"
5) Run the assistant (one-shot)
bash
Copy code
cd ~/projects/sophia-robot/tools/assistant
source .venv/bin/activate
# concise by default:
MODE=casual python ask_once.py 7
# teaching style:
MODE=tutor python ask_once.py 7
You’ll see timing like:

ini
Copy code
[timings] rec=7.00s asr=0.35s llm=1.10s tts=0.40s total=8.85s
FAQ / Common issues
Conv1D requires cuDNN
Rebuild CTranslate2 with -DWITH_CUDNN=ON and install; ensure Python resolves /usr/local/lib/libctranslate2.so.

Ollama “404 model not found”
Pull the tag on the Ollama host or pick one from /api/tags.

First word clipped
ASK_VAD=0 python ask_once.py 7

TTS warning: “Bad voice attribute: option”
Harmless; can ignore.

Cold LLM start is slow
Ollama uses keep_alive: "30m" in requests; you can also prefer a smaller model (e.g., phi3.5:3.8b).

Next steps
Wake word (“Hey Sophia”) + barge-in loop

Intent router (weather/math/jokes/story tools)

Local LLM defaults on Jetson

Kid-safe guardrails + parent prompts


---
