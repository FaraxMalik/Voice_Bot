# Voice Agent API (Local, Docker-Based)

Local voice agent backend that does one full conversational turn:

`Audio input -> STT (faster-whisper) -> LLM (Ollama / Llama 3.2) -> TTS (XTTS v2) -> Audio output`

Everything runs offline after the first downloads and model caching.

---

## What you need (high level)

1. Docker Desktop installed and running
2. The XTTS v2 fine-tuned model files placed in `ljspeech_model/` (required)
3. Internet access the first time (to download):
   - faster-whisper model
   - Ollama model
   - XTTS v2 *base* model (so your fine-tuned GPT weights can be merged correctly)

---

## Folder layout (important)

Your GitHub repo should contain this:

- `docker-compose.yml`
- `agent/` (FastAPI backend)
- `ljspeech_model/` (your fine-tuned XTTS model files + config)

The repo `.gitignore` ignores large weights (`*.pth`, `*.wav`). That is normal. Copy the required model files into `ljspeech_model/` before running.

### Required files in `ljspeech_model/`

Put these files exactly here:

- `ljspeech_model/config.json`
- `ljspeech_model/model.pth` (your fine-tuned checkpoint)
- `ljspeech_model/dvae.pth`
- `ljspeech_model/mel_stats.pth`
- `ljspeech_model/vocab.json`
- `ljspeech_model/speakers_xtts.pth`
- `ljspeech_model/reference.wav` (speaker reference audio)

---

## Configure (environment variables)

Edit `.env` (it is already provided in the repo).

Common variables:

**FastAPI**
- `API_HOST` (default `0.0.0.0`)
- `API_PORT` (default `8000`)

**Ollama (LLM)**
- `OLLAMA_BASE_URL` (default `http://ollama:11434`)
- `OLLAMA_MODEL` (default `llama3.2:3b`)

**STT (faster-whisper)**
- `STT_MODEL_SIZE` (default `small`)
- `STT_DEVICE` (default `cpu`)
- `STT_COMPUTE_TYPE` (default `int8`)

**TTS (XTTS v2)**
- `TTS_MODEL_DIR` (default `/app/models/tts`)
- `TTS_REFERENCE_WAV` (default `/app/models/tts/reference.wav`)
- `TTS_DEVICE` (default `cpu`)
- `COQUI_TOS_AGREED=1` (required for non-interactive Coqui downloads)

**Agent behavior**
- `AGENT_SYSTEM_PROMPT`
- `AGENT_MAX_HISTORY`

---

## Resource notes (OOM prevention)

This project can download and load large models. On machines with ~8GB RAM, the container may be OOM-killed (exit code `137`) during XTTS base model loading.

If you're on Windows with WSL2 + Docker Desktop:

1. Create/edit: `C:\Users\<YOUR_USER>\.wslconfig`
2. Add:

   ```ini
   [wsl2]
   memory=6GB
   swap=4GB
   ```
3. Restart WSL:
   ```powershell
   wsl --shutdown
   ```
4. Restart Docker Desktop.

For best results: use 16GB+ RAM if available.

---

## Run with Docker Compose

From the repo root:

```powershell
docker compose build
docker compose up -d
```

First run can take several minutes because models download and cache.

To watch startup progress:

```powershell
docker compose logs -f agent
```

When everything is ready you should see:

`ALL MODELS LOADED - API IS LIVE`

---

## Verify (quick checks)

Health:

```powershell
curl http://localhost:8000/voice/health
```

Expected:

`{"healthy":true,"services":{"stt":true,"llm":true,"tts":true}}` (when fully loaded)

TTS only:

```powershell
curl -X POST http://localhost:8000/voice/speak -F "text=Hello world" --output test.wav
```

Play `test.wav`.

---

## API endpoints

Base path: `/voice`

### 1) Full pipeline (recommended)

`POST /voice/process`

Form fields:
- `audio` (UploadFile, required)
- `session_id` (Form, optional, default: `default`)

Returns:
- `audio/wav` body
- headers:
  - `X-Transcript`
  - `X-Reply-Text`

Example:

```powershell
curl -X POST http://localhost:8000/voice/process `
  -F "audio=@some_audio.wav" `
  -F "session_id=test1" `
  --output reply.wav
```

### 2) STT only

`POST /voice/transcribe`

Form fields:
- `audio` (UploadFile, required)

Response JSON:
- `transcript`

### 3) LLM only

`POST /voice/chat`

Form fields:
- `text` (required)
- `session_id` (optional, default `default`)

Response JSON:
- `reply`

### 4) TTS only

`POST /voice/speak`

Form fields:
- `text` (required)

Returns:
- `audio/wav`

### 5) Clear chat history

`DELETE /voice/history/{session_id}`

Response JSON:
- `status`
- `session_id`

---

## Troubleshooting

### Container keeps restarting (exit code 137)

That's usually OOM. Increase RAM/WSL swap or stop/restart after caches exist.

Check logs:

```powershell
docker compose logs agent
```

### "Ollama not reachable"

Wait until Ollama is ready, or confirm Docker Desktop networking.

### "Model not found / downloading fails"

You need internet for the first run unless the Docker volumes already have cached models.

---

## Stop the stack

```powershell
docker compose down
```

