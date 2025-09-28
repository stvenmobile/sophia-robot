# 🤖 Sophia — A Voice Assistant Robot for Kids  

[![Jetson Orin Nano](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-green)](#)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Status](https://img.shields.io/badge/Status-Prototype-orange)](#)  

Sophia is a friendly **AI-powered voice companion for children**, designed to answer questions, tutor in various subjects, and tell engaging stories — all in a safe, local, and interactive way.  

Built on the **Jetson Orin Nano Developer Kit (8 GB, 6-core)**, Sophia runs completely offline with a local LLM and speech pipeline, ensuring **privacy, low-latency, and safe interactions**.  

---

## 📖 Table of Contents

- [🤖 Sophia — A Voice Assistant Robot for Kids](#-sophia--a-voice-assistant-robot-for-kids)
  - [📖 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🛠 Tech Stack](#-tech-stack)
  - [🚀 Roadmap](#-roadmap)
  - [🔒 Safety \& Design Principles](#-safety--design-principles)
  - [📦 Getting Started](#-getting-started)
  - [🤔 Why “Sophia”?](#-why-sophia)

---

## ✨ Features

- **Voice interaction loop**  
  - Wake-word detection (*"Hey Sophia"*)  
  - Real-time speech-to-text (STT)  
  - Local LLM for safe, age-appropriate responses  
  - Text-to-speech (TTS) with a **female voice**  

- **Kid-friendly tutoring**  
  - Short explanations for homework questions  
  - Mini-lessons in math, science, history, and more  
  - Encouraging and positive responses  

- **Storytelling & fun**  
  - Tell bedtime stories and fun facts  
  - Answer “why” questions with curiosity and patience  

- **Robot-ready design**  
  - *Stationary*: Desktop robot with expressive arms/gestures  
  - *Mobile*: Expandable to ROS 2 robots with navigation and motion planning  

- **Local-first**  
  - No cloud dependency  
  - Runs fully offline on Jetson Orin Nano  

---

## 🛠 Tech Stack

**Hardware**  
- NVIDIA Jetson Orin Nano Dev Kit (8 GB)  
- USB microphone or WM8960 audio HAT  
- Speaker (3.5mm or I²S DAC)  

**Software**  
- Wake-word → [OpenWakeWord](https://github.com/dscripka/openwakeword) / [Porcupine](https://picovoice.ai/platform/porcupine/)  
- VAD → [Silero VAD](https://github.com/snakers4/silero-vad)  
- STT → [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CUDA optimized)  
- LLM → [Ollama](https://ollama.com/) with *Phi-3-mini* or *Qwen2.5-3B* (quantized)  
- TTS → [Piper](https://github.com/rhasspy/piper) for natural female voice  
- Optional → ROS 2 (Humble/Jazzy) for motion, planning, and expressive gestures  

---

## 🚀 Roadmap

- [ ] **Phase 1 — Voice Loop MVP**  
  - Wake-word + STT + LLM + TTS on Orin Nano  
  - Test latency and conversation flow  

- [ ] **Phase 2 — Tutoring Mode**  
  - Subject-specific prompts and kid-friendly system messages  
  - Content filters for safe conversation  

- [ ] **Phase 3 — Expressive Robot**  
  - Servo arms or LED expressions tied to dialogue  
  - Multi-modal interaction  

- [ ] **Phase 4 — Mobile Expansion (optional)**  
  - ROS 2 base for navigation and movement  
  - Dual-computer split: Orin Nano for voice, secondary SBC for mobility  

---

## 🔒 Safety & Design Principles

- **Age-appropriate**: Short, simple, positive answers  
- **Local-first**: No internet required  
- **Filtered**: Avoid adult, violent, scary, or sensitive topics  
- **Friendly**: Sophia speaks with warmth, patience, and encouragement  

---

## 📦 Getting Started

> Setup instructions coming soon once Phase 1 voice loop is stable.  

**Prerequisites**  
- JetPack 6.2 on Jetson Orin Nano  
- Python 3.8+  
- CUDA + cuDNN (installed with JetPack)  

**Quick start (planned):**
```bash
git clone https://github.com/YOURUSERNAME/sophia-robot.git
cd sophia-robot
pip3 install -r requirements.txt
python3 sophia.py

Say: “Hey Sophia” and start talking!
```

## 🤔 Why “Sophia”?

- Sophia comes from the Greek word for wisdom.
- The name conveys guidance, warmth, and intelligence, fitting for a learning companion.
- While “Sophia” has been used elsewhere in robotics, this project focuses on child-safe tutoring and companionship.

📸 Demo (coming soon)

Screenshots, photos, and video demos will be added as development progresses.

📜 License

This project is licensed under the MIT License
.
