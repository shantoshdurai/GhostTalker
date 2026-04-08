---
title: GhostTalker
emoji: 👻
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Zero-shot voice cloning powered by F5-TTS
---

# 👻 GhostTalker

GhostTalker is a zero-shot voice cloning app powered by **F5-TTS**. Upload a 3–10 second audio clip of any voice, type what you want it to say, and generate natural-sounding speech in that voice — no training required.

## ✨ Features

- **Zero-Shot Cloning** — No training. Just a short reference audio clip.
- **Auto Speech-to-Text** — Leave the transcript blank and it detects it automatically.
- **GPU Accelerated** — Runs on CUDA (RTX 2050 locally, T4 on HuggingFace Spaces).
- **Simple UI** — Record from microphone or upload a file. Generate and download.

## 🚀 Local Setup

**Prerequisites:** Python 3.10+, NVIDIA GPU recommended

```bash
# Clone and launch
git clone https://github.com/shantoshdurai/GhostTalker
cd GhostTalker
LAUNCH_GHOSTTALKER.bat   # Windows — sets up venv, installs deps, starts app
```

Open `http://127.0.0.1:7860` in your browser.

## 🎙️ How to Use

1. **Upload or record** a 3–10 second audio clip of the voice you want to clone
2. **Type the transcript** of what is said in that clip (or leave blank for auto-detection)
3. **Type the text** you want the cloned voice to speak (keep under 200 characters)
4. Click **Generate Cloned Voice**

## 💡 Tips

- Clean audio with no background noise gives the best results
- Typing the reference transcript manually improves cloning accuracy
- For long speech, split into multiple short runs and combine the audio files

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Voice Cloning | F5-TTS 1.1.18 (Flow Matching + DiT) |
| Speech-to-Text | OpenAI Whisper Base |
| UI | Gradio 6.11 |
| GPU Backend | PyTorch 2.5.1 + CUDA 12.1 |

## 📄 License

The F5-TTS model is licensed under **CC-BY-NC** (non-commercial use). The application code is open source.
