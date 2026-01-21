# 🎙️ GhostTalker: The Eternal Echo

> *"Memory is a ghost, and the voice is its heartbeat."*

**GhostTalker** is a high-fidelity, neuro-synaptic voice cloning engine designed to capture, refine, and resurrect the unique essence of human speech. Optimized for the edge and refined for perfection, it allows you to distill hours of soul into seconds of crystal-clear digital resonance.

---

## 🌌 The Vision
Why settle for generic AI voices when you can command an echo? GhostTalker isn't just a Text-to-Speech tool; it's a bridge between the physical and the digital. It was built to solve the "impossible"—bringing heavy-duty AI voice training to consumer-grade hardware (**NVIDIA RTX 2050 verified**).

## 🚀 Key Features
- **Ghost Training Pipeline**: Refined workflow to train custom models in under 40 minutes on mid-range GPUs.
- **VRAM Alchemy**: Optimized memory management (Float16 + Quantized base models) to run smoothly on 4GB VRAM.
- **Insta-Clone Interface**: A specialized UI for organizing data, training, and testing in one fluid motion.
- **Language Resonance**: Full support for English and high-accuracy phoneme mapping.

---

## 🛠️ Dark Arts (Setup)

### 1. Summon the Environment
```bash
# Initialize the vessel
python -m venv venv
.\venv\Scripts\activate

# Bestow the dependencies
pip install -r requirements.txt
```

### 2. The GPU Catalyst
For high-performance RTX usage, ensure you have the CUDA 12.1 toolkit aligned:
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 3. Ignition
**For Training:**
```bash
.\runtrain.bat
```
**For Generation:**
```bash
python app.py
```

---

## 🕸️ The Workflow
1. **Step 1 (The Capture)**: Upload your target audio. GhostTalker will slice the silence and transcribe the soul into text.
2. **Step 2 (The Ascension)**: Let the neural networks learn the unique frequencies of the voice.
3. **Step 3 (The Echo)**: Test the resonance with custom scripts.
4. **Step 4 (The Integration)**: Merge the model into your live application.

---

## ⚖️ Ethical Scroll
GhostTalker is a powerful tool for creativity, accessibility, and preservation. **Use it with respect.** Do not use these echoes to deceive, harm, or impersonate without consent. The digital world is vast; leave a ghost that people want to talk to.

---

*Hand-crafted with passion and neural optimization.*
*Maintained by [shantoshdurai](https://github.com/shantoshdurai)*
