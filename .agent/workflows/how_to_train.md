---
description: How to train a custom voice model using the clone-voice tool.
---

# How to Train a Custom Voice Model

This workflow guides you through training a new voice model using your own audio files.

## Prerequisites
- A clear audio recording of the target voice (recommended: 10-30 minutes total, split into smaller files if possible).
- WAV format is preferred, but MP3/FLAC are supported.
- Ensure your Python virtual environment is set up.

## Steps

1.  **Prepare your Audio**
    - Although the tool can handle large files, it is **highly recommended** to split your 1.5-hour audio into smaller chunks (e.g., 10-15 minutes each).
    - This prevents memory crashes and makes the "Step 1" preprocessing faster and safer.
    - Name them simply, e.g., `part1.wav`, `part2.wav`.

2.  **Launch the Training Interface**
    - Open a terminal in the `clone-voice` directory.
    - Run the training launcher:
      ```powershell
      .\runtrain.bat
      ```
    - Wait for the message: `Running on local URL:  http://0.0.0.0:5003`.

3.  **Open the Web Interface**
    - Open your browser and go to: [http://127.0.0.1:5003](http://127.0.0.1:5003).

4.  **Step 1: Preprocessing (The Data Crunch)**
    - **Model Name**: Give your model a unique name (e.g., `spiderman_v1`).
    - **Language**: Select the language of the audio (e.g., `en` for English).
    - **Upload**: Drag and drop your audio files (`part1.wav`, etc.) into the upload box.
    - **Click "Step 1"**: This starts the transcription process.
      - *Note:* This uses a Whisper model to transcribe and slice your audio. It might take a while! Watch the terminal for progress bars.

5.  **Step 2: Training (The Learning)**
    - Once Step 1 is done, you will see text appear in the "Training dataset" box.
    - **Click "Step 2"**: This starts the actual model training.
    - Since we set `num_epochs=4` and `batch_size=1`, this is optimized for laptops.
    - Expected time: 30-60 minutes depending on your GPU/CPU.

6.  **Step 3 & 4: Testing and Saving**
    - **Click "Step 3"**: Listen to a sample generation.
    - **Click "Step 4"**: This saves the model so you can use it in the main `app.py` interface.

## Using Your New Model
1.  Close the training terminal (Ctrl+C).
2.  Run the main app again: `python app.py`.
3.  Your new model (e.g., `spiderman_v1`) will appear in the "Model" dropdown list!
