# Using the Trained XTTS Model From Scratch

This guide explains how to run and integrate your trained XTTS model on a new machine from zero.

## 1) What this model is

Your trained model is a fine-tuned Coqui XTTS model for English single-speaker text-to-speech.
It was trained in an LJSpeech-style setup, which means the voice is consistent and suited to narration-style output.

### What is LJSpeech?
LJSpeech is a standard public dataset for TTS:
- Single female English speaker
- About 24 hours of speech
- Split into many short clips with text transcripts

A model trained in this style usually produces stable voice identity and clear pronunciation for English text.

## 2) What you need on a new machine

- Python 3.8 to 3.11
- pip
- Optional but recommended: NVIDIA GPU with CUDA-compatible PyTorch
- At least 2 GB free disk space

You also need the full model folder containing at least:
- model.pth
- config.json
- vocab.json
- dvae.pth
- mel_stats.pth
- speakers_xtts.pth

Do not run with partial files. Missing files cause load failures.

## 3) Fresh environment setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install TTS torch torchaudio soundfile numpy
```

If you need GPU acceleration, install the CUDA-matching PyTorch build from the official PyTorch site first, then install the remaining packages.

## 4) Folder layout to keep

Recommended minimal layout:

```text
project/
  inference.py
  ljspeech_model/
    model.pth
    config.json
    vocab.json
    dvae.pth
    mel_stats.pth
    speakers_xtts.pth
```

## 5) Basic run command

Generate one wav file:

```powershell
python inference.py --model_dir ./ljspeech_model --text "Hello world, this is a test." --output test.wav
```

If your script supports it, you can also run:

```powershell
python inference.py --model_dir ./ljspeech_model --interactive
```

Use CPU only (slower):

```powershell
python inference.py --model_dir ./ljspeech_model --text "CPU test" --output cpu_test.wav --cpu
```

## 6) Quick health check

After the first run, verify:
- A wav file is created
- Audio plays correctly
- No missing model file errors
- No CUDA or DLL mismatch errors

If quality sounds clipped, normalize output to around 0.95 peak before saving.

## 7) Programmatic integration (Python)

Use a wrapper function in your backend so application code only sends text and receives output path or bytes.

```python
import subprocess
from pathlib import Path

MODEL_DIR = Path("./ljspeech_model")


def tts_generate(text: str, output_path: str = "output.wav") -> str:
    cmd = [
        "python",
        "inference.py",
        "--model_dir",
        str(MODEL_DIR),
        "--text",
        text,
        "--output",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path
```

This keeps integration simple and avoids coupling app logic to model internals.

## 8) API integration pattern

Typical production flow:
1. Client sends text
2. Server validates and queues request
3. Worker runs inference with the trained model
4. Server returns wav URL or bytes

Use a queue for high traffic and cache repeated text outputs when possible.

## 9) Common errors and fixes

- `Missing file in model_dir`: ensure all required model files are present
- `CUDA not available`: install correct GPU-enabled PyTorch or run with CPU
- `DLL load failed` on Windows: reinstall torch/torchaudio in a clean venv
- `Out of memory`: reduce concurrency, shorter text chunks, or run CPU fallback

## 10) Performance tips

- Prefer GPU for lower latency
- Chunk long text by sentences and concatenate outputs
- Reuse a loaded model process instead of reloading per request
- Add request timeout and retry logic in your service layer

## 11) When to retrain

Use the current model as-is if you want the same voice and language behavior.
Retrain or fine-tune only when you need:
- New speaker identity
- Different accent/style
- Domain-specific pronunciation improvements

## 12) Minimum go-live checklist

- Environment reproducible with pinned package versions
- Model folder versioned and checksummed
- Startup test generates known test.wav
- Monitoring for latency, failures, and queue depth
- Fallback path if inference fails

---

If you want, the next step is I can also add:
- a one-command Windows launcher (`run_tts.bat`)
- a small FastAPI server (`POST /synthesize`)
- a requirements lock file for stable deployment
