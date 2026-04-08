#  GhostTalker

**Live Demo → [huggingface.co/spaces/Santoshp123/GhostTalker](https://huggingface.co/spaces/Santoshp123/GhostTalker)**

GhostTalker is a zero-shot voice cloning app powered by **F5-TTS**. Upload a 3–10 second audio clip of any voice, type what you want it to say, and it generates natural-sounding speech in that voice — no training required.

---

## ✨ Features

- **Zero-Shot Cloning** — No model training. Just a short reference audio clip.
- **Auto Speech-to-Text** — Leave the transcript blank; Whisper detects it automatically.
- **Works on CPU & GPU** — Runs locally on CUDA (RTX 2050 tested) or on HuggingFace Spaces CPU.
- **Simple UI** — Record from microphone or upload a file. Generate and download.

---

## 🚀 Local Setup (Windows)

**Prerequisites:** Python 3.10+, NVIDIA GPU recommended (CPU works but is slow)

```bash
git clone https://github.com/shantoshdurai/GhostTalker
cd GhostTalker
LAUNCH_GHOSTTALKER.bat
```

The launcher will:
- Create a Python virtual environment (`venv/`)
- Install PyTorch with CUDA 12.1 support
- Install all dependencies from `requirements.txt`
- Open `http://127.0.0.1:7860` in your browser automatically

---

## 🎙️ How to Use

1. **Record or upload** a 3–10 second clip of the voice you want to clone
2. **Reference Transcript** — type what is said in the clip, or leave blank for auto-detection
3. **Target Text** — type what you want the cloned voice to say (keep under 200 characters)
4. Click **Generate Cloned Voice** and wait for the audio to appear

---

## 💡 Tips for Best Results

- Use clean audio with no background noise or music
- Typing the reference transcript yourself improves cloning accuracy over auto-detect
- Keep generated text under **200 characters** per run — split longer speech into multiple clips
- CPU generation takes ~60 seconds; GPU (CUDA) takes ~7 seconds

---

## 🛠️ Tech Stack

| Component | Library / Model |
|---|---|
| Voice Cloning Model | [F5-TTS 1.1.18](https://github.com/SWivid/F5-TTS) — Flow Matching + DiT |
| Vocoder | Vocos (charactr/vocos-mel-24khz) |
| Speech-to-Text | OpenAI Whisper Base |
| UI Framework | Gradio 6.11 |
| GPU Backend | PyTorch 2.5.1 + CUDA 12.1 (local) |
| CPU Backend | PyTorch 2.11 (HuggingFace Spaces) |

---

## 🧠 How It Works

1. Reference audio is transcribed automatically using Whisper (if no transcript provided)
2. F5-TTS uses the reference audio + transcript to learn the voice style in zero-shot
3. The model generates mel spectrogram features via flow matching (DiT architecture)
4. Vocos vocoder converts the mel spectrogram to a waveform

---

## 📋 Requirements

```
f5-tts
gradio
torch / torchaudio (CUDA 12.1)
openai-whisper
pydub
soundfile
vocos
```

Full list in [`requirements.txt`](requirements.txt). System package `ffmpeg` required (bundled via `imageio-ffmpeg` on Windows; installed via `packages.txt` on HuggingFace Spaces).

---

## 📄 License

The F5-TTS model weights are licensed under **CC-BY-NC-4.0** (non-commercial use only).  
Application code in this repo is open source under the same license.
