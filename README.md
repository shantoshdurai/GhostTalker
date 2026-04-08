# 👻 GhostTalker

GhostTalker is a powerful, zero-shot voice cloning application based on the state-of-the-art **F5-TTS** engine. It allows you to clone any voice using just a 3-10 second audio sample and generate natural-sounding speech from text.

## ✨ Features
- **Zero-Shot Cloning**: No training required. Just upload a sample.
- **High Quality**: Powered by F5-TTS with ConvNeXt V2 and Flow Matching.
- **Easy UI**: Simple Gradio web interface for uploading, generating, and playing back audio.
- **Optimized for RTX**: Pre-configured to use CUDA 12.1 for NVIDIA GPUs (like the RTX 2050).

## 🚀 How to Run

### Prerequisites
- **Python 3.10 or 3.11** installed.
- **FFmpeg** installed on your system (required for audio processing).
- **NVIDIA GPU** (optional but recommended for faster generation).

### Quick Start
1. Double-click `LAUNCH_GHOSTTALKER.bat`.
2. Wait for the environment to be set up and dependencies to install.
3. Once the local URL appears (e.g., `http://127.0.0.1:7860`), open it in your browser.

## 🎙️ Usage Tips
- **Sample Length**: Use clean audio snippets between 3 and 10 seconds.
- **Reference Text**: If you provide the exact transcript of the reference audio, the cloning accuracy improves significantly.
- **Noise**: Background noise in the reference audio will negatively impact the output quality.

## 📄 License
This project uses the **F5-TTS** model which is licensed under CC-BY-NC. The code in this repository is provided for educational and research purposes.
