# Voice Bot API — Project Architecture & History

## 🚀 Overview
This is a fully Dockerized AI Voice Bot Pipeline. It orchestrates three models directly from a single HTTP Post request:
1. **faster-whisper (STT):** Transcribes user uploaded `.wav` audio strictly on the CPU.
2. **Ollama / LLaMA 3.2 3B (LLM):** Ingests the transcription and streams a smart, context-aware reply.
3. **Coqui XTTS v2 / Fine-Tuned (TTS):** Uses custom LJSpeech PyTorch keys to instantly synthesize the LLaMA text into an audible response voice waveform via float32 PCM data.

## 🗺️ API Routes (Defined in `agent/routes/voice.py`)
- **`POST /voice/process`**: The core pipeline. Upload an audio file, receive a `.wav` file back. (Contains `Content-Disposition: attachment` for browser download).
- **`POST /voice/chat`**: Bypass STT/TTS; send text directly to the LLM.
- **`POST /voice/speak`**: Bypass STT/LLM; send text directly to the XTTS engine.
- **`GET /voice/health`**: Diagnostic check for STT, LLM, and TTS loading states.

## 🛠️ Critical System Fixes We Already Completed (DO NOT REVERT)

### 1) The TTS Missing Keys Bug (`tts_service.py`)
- **Issue:** The fine-tuned `model.pth` is a 'bare' PyTorch Lightning checkpoint containing exactly 426 GPT keys. It lacks the 130+ `hifigan_decoder` variables present in base XTTS models. Loading it directly resulted in 1.34 MB WAV files containing pure silence (zeroes) because the 'vocal cords' were randomly initialized.
- **Fix:** We permanently rigged the `_load_standard` fallback block to route bare checkpoints straight to `_load_finetuned()`. This function forces Docker to temporarily download the generic Coqui XTTS base model, surgically extracts its HiFi-GAN variables, merges them perfectly with our custom 426 GPT keys, and executes the payload with `strict=False` to bypass missing PyTorch keys.

### 2) The LLM / OOM Constraints (`llm_service.py`)
- **Issue:** Ollama local CPU inference takes time. The previous 120-second timeout violently closed the HTTP Pipeline, crashing FastAPI. Secondarily, activating LLaMA alongside STT/TTS demanded nearly 6 GB of RAM, triggering Docker Engine Exit Code 137 (OOM Killer) on constrained Windows Hosts.
- **Fix:** We permanently raised the `httpx.post` timeout inside `llm_service.py` to `True` 600.0 (10 Minutes). We simultaneously created an `expand_mem.txt` / `.wslconfig` setup script to force 12 GB of SSD Swap allocation to the underlying WSL2 network, mathematically destroying any chance of Exit Code 137.

## 💡 How to Deploy on a New Machine
1. Ensure the new host has Docker.
2. (Optional but Recommended) If the host is Windows with 8GB RAM, run the `expand_mem.txt` script to build a `.wslconfig` guaranteeing swap space, and run `wsl --shutdown`.
3. CD into the project root and spin the system permanently up using: \`docker compose up -d --build\`
4. Check startup health via: \`docker compose logs -f agent\`
