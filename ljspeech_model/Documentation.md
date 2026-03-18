# XTTS V1.2 Model Documentation

## Overview
This document describes the XTTS V1.2 model, including data sources, training process, results, and setup instructions. It is intended for users and researchers interested in understanding, reproducing, or deploying the model.

---

## 1. Data Source
- **Dataset:** LJSpeech-1.1 (13,100 audio clips, ~24 hours, single speaker)
- **Preparation:**
  - Original dataset was processed using a custom pipeline (see `audio_processor.py` and project logs).
  - 10,104 MP3 files were analyzed for quality; the best 5,000 were selected and converted to 22,050 Hz, mono, 16-bit PCM WAV files.
  - Quality metrics included energy variance, spectral centroid, and zero crossing rate.
  - Metadata and mapping files were generated for traceability.

---

## 2. Model Architecture
- **Base Model:** XTTS v2 (Coqui)
  - The base model files (DVAE, mel_stats, vocab, config) are included in the final model package for inference. No additional download is needed to run the model on a new device.
- **Fine-tuned Components:**
  - Text Tokenizer / Conditioner
  - GPT-2 Backbone (441M parameters)
  - Perceiver Resampler
  - HiFi-GAN Decoder
- **Frozen Components:**
  - DVAE Encoder (copied from base model)
- **What Was Trained:**
  - The training process fine-tuned the Text Tokenizer/Conditioner, GPT-2 backbone, Perceiver Resampler, and HiFi-GAN Decoder on the LJSpeech dataset. The DVAE Encoder was kept frozen and copied from the base model.
- **Training Script:** `train_ljspeech_xtts.py`
- **Framework:** PyTorch, Coqui TTS

---

## 3. Training Process
- **Hardware:** RTX 3090 (24GB VRAM), 32GB RAM, 1TB SSD
- **Batch Size:** 2 (effective 4 with gradient accumulation)
- **Epochs:** 6 (8-10 hours)
- **Optimizer:** AdamW
- **Learning Rate:** 5e-6
- **Validation:** 2% of data reserved for evaluation
- **Checkpoints:** Saved every 2,000 steps, rolling window of 3
- **Output:** Complete inference package (model weights, config, vocab, etc.)

---

## 4. Results
- **Audio Quality:**
  - Natural, human-like speech
  - Consistent speaker identity (LJSpeech)
  - No major artifacts or distortions
- **Evaluation:**
  - Subjective listening tests on random samples
  - All files passed quality checks (see `quality_check_report.txt`)
- **Output Files:**
  - `model.pth`, `dvae.pth`, `mel_stats.pth`, `vocab.json`, `config.json`, `speakers_xtts.pth`

---

## 5. Setup & Inference

### Setting Up This Model on a New Device

#### 1. Prepare Your Environment
- Recommended: Use a fresh Python 3.8+ virtual environment (e.g., with `venv` or `conda`).
- Ensure you have sufficient disk space (at least 2 GB free).

#### 2. Transfer Model Files
- Copy the entire `ljspeech_model/` folder (containing all required files: `model.pth`, `dvae.pth`, `mel_stats.pth`, `vocab.json`, `config.json`, `speakers_xtts.pth`) to your new device.
- All files needed for inference—including those from the XTTS v2 base model—are already included in this folder. You do not need to download the base model separately.
- Place it in your working directory or specify the path with `--model_dir`.

#### 3. Install Dependencies
- Install Python dependencies using the provided requirements file:
  ```bash
  pip install -r requirements.txt
  ```
- If you encounter issues with PyTorch or torchaudio, refer to the official installation guides for your OS and hardware (especially for GPU support).

#### 4. Test the Installation
- Run a quick test to verify everything is working:
  ```bash
  python inference.py --model_dir ./ljspeech_model --text "Test phrase" --output test.wav
  ```
- Check that `test.wav` is generated and plays correctly.

#### 5. Troubleshooting
- If you see errors about missing DLLs or CUDA, ensure your Python and PyTorch installations match your hardware (CPU/GPU).
- For Linux, you may need to install system packages like `libsndfile1`.
- See `TROUBLESHOOTING.md` for more help.

#### 6. Usage Examples
- **Single sentence:**
  ```bash
  python inference.py --model_dir ./ljspeech_model --text "Hello world" --output hello.wav
  ```
- **Interactive mode:**
  ```bash
  python inference.py --model_dir ./ljspeech_model --interactive
  ```
- **Batch from file:**
  ```bash
  python inference.py --model_dir ./ljspeech_model --file script.txt --output narration.wav
  ```

---

---

## 6. File Reference
- `train_ljspeech_xtts.py`: Full training pipeline
- `inference.py`: Speech generation from trained model
- `audio_processor.py`: Data cleaning and selection
- `logs/`: Reports, summaries, and quality checks
- `ljspeech_model/`: Final model package for inference

---

## 7. Notes & Recommendations
- Always verify output quality with listening tests.
- For best results, use a GPU for inference.
- The model is designed for English, single-speaker TTS.
- For multi-speaker or multilingual use, further fine-tuning is required.

---

## 8. Contact & Support
- See `README.md`, `QUICKSTART.md`, and `TROUBLESHOOTING.md` for more details.
- For technical issues, consult the logs and documentation in the `logs/` folder.

---

**Status:** Model fully trained and ready for deployment.
