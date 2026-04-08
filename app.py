import os
import sys
import shutil
import types

# ── torchcodec crash guard ────────────────────────────────────────────────────
# f5-tts installs torchcodec which tries to dlopen libnppic.so.13 (CUDA NPP)
# at import time — crashes on CPU-only HF Spaces (Python 3.13 / PyTorch 2.11).
#
# Fix: inject stub modules into sys.modules BEFORE f5-tts / datasets are
# imported. Python returns the stub on `import torchcodec` without executing
# the real native loader.
#
# Python 3.13 importlib.util.find_spec() raises ValueError when __spec__ is
# None, so we must supply a proper ModuleSpec with loader=None.
import importlib.machinery as _ilm

def _stub(name: str):
    m = types.ModuleType(name)
    m.__path__ = []          # mark as package so sub-imports don't fail
    m.__spec__ = _ilm.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = m
    return m

for _tc_mod in [
    "torchcodec",
    "torchcodec.decoders",
    "torchcodec.decoders._core",
    "torchcodec._core",
]:
    if _tc_mod not in sys.modules:
        _stub(_tc_mod)

del _stub, _tc_mod, _ilm
# ─────────────────────────────────────────────────────────────────────────────

# On Windows, imageio-ffmpeg bundles ffmpeg as "ffmpeg-win64-vX.exe".
# Copy it as "ffmpeg.exe" so pydub can find it.
# On Linux (HuggingFace Spaces), ffmpeg is installed via packages.txt — skip this.
if os.name == "nt":
    import imageio_ffmpeg
    _ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_src)
    _ffmpeg_exe = os.path.join(_ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(_ffmpeg_exe):
        shutil.copy(_ffmpeg_src, _ffmpeg_exe)
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# ZeroGPU support for HuggingFace Spaces (no-op locally)
try:
    import spaces
    HF_SPACES = True
except ImportError:
    spaces = None
    HF_SPACES = False

import torch
import gradio as gr
import tempfile
import json
import soundfile as sf
from pydub import AudioSegment
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Starting GhostTalker on device: {device}")
if device == "cpu":
    print("[!] No CUDA GPU detected — running on CPU. Generation will take ~60 seconds per request.")
else:
    import torch
    print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

# Preload ALL models at startup — nothing downloads during a request
print("[*] Loading F5-TTS model...")
ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
vocab_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[1]))
model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
_f5tts_model = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path, device=device)

print("[*] Loading Vocoder...")
_vocoder = load_vocoder(device=device)

print("[*] Loading Whisper ASR model...")
import whisper as _whisper_pkg
_whisper_model = _whisper_pkg.load_model("base", device=device)

print("[*] All models ready.")


def get_vocoder():
    return _vocoder


def get_f5tts():
    return _f5tts_model


def auto_transcribe(audio_path: str) -> str:
    """
    Transcribe audio using openai-whisper directly.
    openai-whisper has no torchcodec dependency and works on CPU-only environments.
    pydub converts to 16kHz mono WAV first so whisper gets clean input.
    """
    print("[*] Auto-transcribing reference audio...")

    # Convert to 16kHz mono WAV via pydub (uses bundled ffmpeg)
    aseg = AudioSegment.from_file(audio_path)
    aseg = aseg.set_channels(1).set_frame_rate(16000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    aseg.export(tmp_wav, format="wav")

    result = _whisper_model.transcribe(tmp_wav)
    os.unlink(tmp_wav)

    transcript = result["text"].strip()
    print(f"[*] Transcribed: {transcript}")
    return transcript


def _run_inference(ref_audio_processed, ref_text_processed, gen_text):
    model = get_f5tts()
    vocoder = get_vocoder()
    # nfe_step=16 on CPU for speed, 32 on GPU for quality
    nfe = 32 if device == "cuda" else 16
    final_wave, final_sample_rate, _ = infer_process(
        ref_audio_processed,
        ref_text_processed,
        gen_text,
        model,
        vocoder,
        nfe_step=nfe,
        device=device,
    )
    return final_wave, final_sample_rate

# Wrap with ZeroGPU decorator on HuggingFace Spaces, no-op locally
if HF_SPACES:
    _run_inference = spaces.GPU(duration=120)(_run_inference)


def clone_voice(ref_audio, ref_text, gen_text, progress=gr.Progress()):
    if not ref_audio:
        raise gr.Error("Please upload or record a reference audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text you want the voice to speak.")
    if len(gen_text.strip()) > 200:
        raise gr.Error(
            f"Text too long ({len(gen_text.strip())} chars). Keep it under 200 characters. "
            "Split longer speech into multiple runs."
        )

    # Auto-transcribe if no reference text provided
    if not ref_text.strip():
        progress(0.05, desc="Transcribing reference audio...")
        ref_text = auto_transcribe(ref_audio)

    progress(0.1, desc="Pre-processing audio...")
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio, ref_text)

    progress(0.3, desc="Generating speech...")
    final_wave, final_sample_rate = _run_inference(ref_audio_processed, ref_text_processed, gen_text)

    progress(0.9, desc="Exporting result...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name
    sf.write(output_path, final_wave, final_sample_rate)

    return output_path, f"Reference used: \"{ref_text_processed}\""


# UI
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
            with gr.Tabs():
                with gr.Tab("🎤 Record"):
                    mic_input = gr.Audio(
                        label="Record 3-10 seconds of the voice to clone",
                        type="filepath",
                        sources=["microphone"],
                    )
                with gr.Tab("📁 Upload"):
                    upload_input = gr.Audio(
                        label="Upload an audio file (WAV, MP3, etc.)",
                        type="filepath",
                        sources=["upload"],
                    )
            audio_input = gr.Audio(
                label="Active Reference Audio",
                type="filepath",
                visible=False,
            )
            # Sync whichever tab is used into audio_input
            mic_input.change(fn=lambda x: x, inputs=mic_input, outputs=audio_input)
            upload_input.change(fn=lambda x: x, inputs=upload_input, outputs=audio_input)

            text_ref = gr.Textbox(
                label="Reference Transcript (Optional — leave blank to auto-detect)",
                placeholder="Type what is said in the clip, or leave blank for auto speech-to-text...",
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
                        <li>Keep reference samples between 3 and 10 seconds.</li>
                        <li>Typing the transcript yourself gives better cloning accuracy than auto-detect.</li>
                        <li>Keep generated text under 200 characters. Split longer speech into multiple runs.</li>
                    </ul>
                </div>
            """)

    btn_generate.click(
        fn=clone_voice,
        inputs=[audio_input, text_ref, text_gen],
        outputs=[audio_output, status_info]
    )

if __name__ == "__main__":
    demo.queue(max_size=5, default_concurrency_limit=1)
    demo.launch(
        show_error=True,
        share=True,
        theme=gr.themes.Default(primary_hue="purple", secondary_hue="indigo"),
    )
