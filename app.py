import os
import shutil
import imageio_ffmpeg

# imageio-ffmpeg bundles ffmpeg but names it "ffmpeg-win64-vX.exe" on Windows.
# Copy it as "ffmpeg.exe" so transformers/pydub subprocess calls can find it.
_ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
_ffmpeg_dir = os.path.dirname(_ffmpeg_src)
_ffmpeg_exe = os.path.join(_ffmpeg_dir, "ffmpeg.exe")
if not os.path.exists(_ffmpeg_exe):
    shutil.copy(_ffmpeg_src, _ffmpeg_exe)
os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
import json
import soundfile as sf
from cached_path import cached_path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

# Constants & Configuration
DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Starting GhostTalker on device: {device}")

# Global Model Loading
_vocoder = None
_f5tts_model = None

def get_vocoder():
    global _vocoder
    if _vocoder is None:
        print("[*] Loading Vocoder...")
        _vocoder = load_vocoder(device=device)
    return _vocoder

def get_f5tts():
    global _f5tts_model
    if _f5tts_model is None:
        print("[*] Loading F5-TTS Model...")
        ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
        vocab_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[1]))
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
        _f5tts_model = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path, device=device)
    return _f5tts_model

def clone_voice(ref_audio, ref_text, gen_text, progress=gr.Progress()):
    if not ref_audio:
        raise gr.Error("Please upload or record a reference audio file.")
    if not ref_text.strip():
        raise gr.Error("Reference Transcript is required. Type exactly what is said in your audio clip.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text you want the voice to speak.")
    if len(gen_text.strip()) > 200:
        raise gr.Error(f"Generated text is too long ({len(gen_text.strip())} chars). Keep it under 200 characters per generation. Split long text into multiple runs.")

    progress(0, desc="Pre-processing audio...")
    # Load models lazily to save memory at startup if needed, but here we preload for speed
    model = get_f5tts()
    vocoder = get_vocoder()

    # Preprocess reference audio (clipping to 12s max, auto-transcribing if text is missing)
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    
    progress(0.3, desc="Generating speech...")
    # Inference process
    final_wave, final_sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        model,
        vocoder,
        device=device,
    )
    
    progress(0.9, desc="Exporting result...")
    # Save the generated audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name
    
    sf.write(output_path, final_wave, final_sample_rate)
    
    return output_path, f"Reference used: \"{ref_text}\""

# UI Construction
with gr.Blocks(title="GhostTalker") as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
            <h1 style="font-size: 2.5em; margin-bottom: 10px;">👻 GhostTalker</h1>
            <p style="font-size: 1.2em;">Advanced Zero-Shot Voice Cloning based on F5-TTS</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Step 1: Reference Audio")
            audio_input = gr.Audio(
                label="Upload or Record (3-10 seconds recommended)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            text_ref = gr.Textbox(
                label="Reference Transcript (Required)",
                placeholder="Type exactly what is said in the audio clip...",
                lines=2
            )
            
            gr.Markdown("### ✍️ Step 2: Target Text")
            text_gen = gr.Textbox(
                label="What should they say? (max ~200 characters)",
                placeholder="Keep it short — under 200 characters per run for best results...",
                lines=4
            )
            
            btn_generate = gr.Button("🚀 Generate Cloned Voice", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 🎧 Result")
            audio_output = gr.Audio(label="Generated Audio", interactive=False)
            status_info = gr.Textbox(label="Status Log", interactive=False)
            
            gr.Markdown("""
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f4f8; border-left: 5px solid #667eea; border-radius: 5px;">
                    <h4 style="margin-top: 0;">💡 Tips for best results:</h4>
                    <ul style="margin-bottom: 0;">
                        <li>Use high-quality audio with no background noise.</li>
                        <li>Keep samples between 3 and 10 seconds.</li>
                        <li>Always type the reference transcript — it's required and improves accuracy.</li>
                        <li>Keep generated text under 200 characters. For long speech, run it in multiple shorter parts.</li>
                    </ul>
                </div>
            """)

    btn_generate.click(
        fn=clone_voice,
        inputs=[audio_input, text_ref, text_gen],
        outputs=[audio_output, status_info]
    )

if __name__ == "__main__":
    demo.launch(show_error=True, theme=gr.themes.Default(primary_hue="purple", secondary_hue="indigo"))
