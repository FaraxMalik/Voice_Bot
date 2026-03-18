import gc
import io
import json
import logging
import os
import shutil

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── PyTorch 2.6+ compatibility (default weights_only=False) ─────
_orig_torch_load = torch.load
def _safe_torch_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _safe_torch_load

# ── torchaudio backend fallback via soundfile ────────────────────
_orig_torchaudio_load = torchaudio.load
def _sf_load(path, *a, **kw):
    try:
        data, sr = sf.read(path, dtype="float32")
        w = torch.from_numpy(data)
        w = w.unsqueeze(0) if w.dim() == 1 else w.T
        return w, sr
    except Exception:
        return _orig_torchaudio_load(path, *a, **kw)
torchaudio.load = _sf_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager, LanguageManager
from TTS.utils.manage import ModelManager

from agent.config import get_settings

logger = logging.getLogger(__name__)

XTTS_BASE_CACHE = "/app/xtts_base"
XTTS_BASE_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_BASE_MODEL_DIRNAME = "tts_models--multilingual--multi-dataset--xtts_v2"
XTTS_BASE_REQUIRED_FILES = [
    "config.json", "model.pth", "dvae.pth", "mel_stats.pth",
    "vocab.json", "speakers_xtts.pth",
]

_model: Xtts | None = None
_gpt_cond_latent = None
_speaker_embedding = None


def _resolve_base_source_dir(model_path: str | None, config_path: str | None) -> str:
    candidates: list[str] = []
    if isinstance(config_path, str) and config_path:
        candidates.append(os.path.dirname(config_path))
    if isinstance(model_path, str) and model_path:
        candidates.append(model_path)
    candidates.append(os.path.join(XTTS_BASE_CACHE, XTTS_BASE_MODEL_DIRNAME))
    # Also check one level deeper under tts/ subfolder
    candidates.append(os.path.join(XTTS_BASE_CACHE, "tts", XTTS_BASE_MODEL_DIRNAME))

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    raise RuntimeError("Could not locate XTTS base config.json after download")


def _cache_base_files(source_dir: str) -> None:
    for fname in XTTS_BASE_REQUIRED_FILES:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(XTTS_BASE_CACHE, fname)
        if not os.path.exists(src):
            logger.warning("Base file not found (may be optional): %s", src)
            continue
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / 1e6
        logger.info("Cached: %s (%.1f MB)", fname, size_mb)


def _ensure_base_model() -> str:
    base_config = os.path.join(XTTS_BASE_CACHE, "config.json")
    base_model = os.path.join(XTTS_BASE_CACHE, "model.pth")

    if os.path.exists(base_config) and os.path.exists(base_model):
        logger.info("XTTS v2 base model already cached at %s", XTTS_BASE_CACHE)
        return XTTS_BASE_CACHE

    logger.info("XTTS v2 base model not found — downloading from Coqui (one-time)…")
    os.makedirs(XTTS_BASE_CACHE, exist_ok=True)

    manager = ModelManager(output_prefix=XTTS_BASE_CACHE)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            model_path, config_path, _ = manager.download_model(XTTS_BASE_MODEL_NAME)
            source_dir = _resolve_base_source_dir(model_path, config_path)
            _cache_base_files(source_dir)
            logger.info("XTTS v2 base model cached successfully")
            return XTTS_BASE_CACHE
        except Exception as exc:
            last_error = exc
            logger.warning("XTTS base download attempt %d failed: %s", attempt, exc)

    raise RuntimeError(f"Failed to download/cache XTTS base model: {last_error}")


def _is_full_inference_checkpoint(model_dir: str) -> bool:
    user_cfg = os.path.join(model_dir, "config.json")
    if not os.path.exists(user_cfg):
        return False

    # If ALL required inference files are present in the directory,
    # treat it as a complete inference package (even if config.json
    # is a training-style config without the "model" key).
    required_files = ["model.pth", "dvae.pth", "mel_stats.pth", "vocab.json"]
    all_present = all(
        os.path.exists(os.path.join(model_dir, f)) for f in required_files
    )
    if all_present:
        logger.info("All required inference files found — treating as full checkpoint")
        return True

    with open(user_cfg, "r") as f:
        cfg = json.load(f)
    return "model" in cfg or "gpt_cond_len" in cfg.get("model_args", {})


def _remap_trainer_keys(state: dict) -> dict:
    trainer_markers = {"conditioning_encoder.init.weight",
                       "text_embedding.weight",
                       "mel_embedding.weight"}
    if not (trainer_markers & set(state.keys())):
        return state

    logger.info("  Detected Trainer-format checkpoint — adding gpt. prefix…")
    KEEP = {"hifigan_decoder.", "mel_stats"}
    remapped: dict = {}
    for key, val in state.items():
        if any(key == p or key.startswith(p) for p in KEEP):
            remapped[key] = val
        else:
            remapped[f"gpt.{key}"] = val
    return remapped


def _extract_base_non_gpt_keys(base_model_path: str) -> dict:
    """
    Load the base XTTS model.pth and extract ONLY the non-GPT keys
    (hifigan_decoder, mel_stats). This avoids holding the full 1.87GB
    in memory — we load, filter, and immediately discard the GPT portion.
    """
    logger.info("  Loading base checkpoint to extract HiFi-GAN weights…")
    raw = torch.load(base_model_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "model" in raw:
        full_state = raw["model"]
        del raw
    else:
        full_state = raw
    gc.collect()

    # Strip xtts. prefix (same as Xtts.get_compatible_checkpoint_state_dict)
    for key in list(full_state.keys()):
        if key.startswith("xtts."):
            full_state[key.replace("xtts.", "", 1)] = full_state.pop(key)

    # Keep ONLY hifigan_decoder and mel_stats keys — discard GPT (saves ~1.5GB)
    kept: dict = {}
    for key in list(full_state.keys()):
        if key.startswith("hifigan_decoder.") or key == "mel_stats":
            kept[key] = full_state.pop(key)

    del full_state
    gc.collect()

    logger.info("  Extracted %d base keys (hifigan_decoder + mel_stats)", len(kept))
    return kept


def load_model() -> None:
    global _model, _gpt_cond_latent, _speaker_embedding
    settings = get_settings()
    model_dir = settings.tts_model_dir
    reference_wav = settings.tts_reference_wav

    logger.info("Loading TTS model from: %s", model_dir)

    if _is_full_inference_checkpoint(model_dir):
        logger.info("Model directory has full inference config — standard load path")
        _model = _load_standard(model_dir, settings)
    else:
        logger.info("Model directory has training-only config — memory-efficient load path")
        _model = _load_finetuned(model_dir, settings)

    logger.info("Computing speaker latents from: %s", reference_wav)
    if not os.path.exists(reference_wav):
        raise FileNotFoundError(f"Reference WAV not found: {reference_wav}")
    _gpt_cond_latent, _speaker_embedding = _model.get_conditioning_latents(
        audio_path=[reference_wav]
    )
    logger.info("TTS model loaded successfully")


def _load_standard(model_dir: str, settings) -> Xtts:
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    
    model_path = os.path.join(model_dir, "model.pth")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        # Standard format
        model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True, use_deepspeed=False)
        return model
    else:
        # Bare checkpoint format (common in some fine-tuning setups)
        # Because it lacks the ~130 hifigan_decoder keys, we MUST route it to the finetuned loader 
        # to merge the base weights in, otherwise it outputs total silence.
        logger.info("Bare checkpoint detected (426 keys) - routing to fine-tuned loader to grab base hifigan_decoder")
        del model
        del checkpoint
        import gc
        gc.collect()
        return _load_finetuned(model_dir, settings)

    model.to(settings.tts_device)
    return model


def _load_finetuned(model_dir: str, settings) -> Xtts:
    """
    Memory-efficient loading for 8GB machines.

    Instead of loading the full base model (1.87GB peak), we:
      1. Create the model architecture (empty weights)
      2. Load the fine-tuned GPT weights (~1.76GB)
      3. Load ONLY the hifigan_decoder + mel_stats from base (~0.2GB)
      4. Merge and apply with strict=False
    This keeps peak memory under ~2.5GB instead of ~4.5GB.
    """
    base_dir = _ensure_base_model()
    base_config_path = os.path.join(base_dir, "config.json")

    for needed in ["dvae.pth", "mel_stats.pth"]:
        dst = os.path.join(model_dir, needed)
        src = os.path.join(base_dir, needed)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info("  Copied base file to model dir: %s", needed)

    logger.info("  [1/5] Creating XTTS model architecture…")
    config = XttsConfig()
    config.load_json(base_config_path)
    config.model_dir = base_dir
    model = Xtts.init_from_config(config)

    vocab_path = os.path.join(model_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(base_dir, "vocab.json")

    speaker_path = os.path.join(model_dir, "speakers_xtts.pth")
    if not os.path.exists(speaker_path):
        speaker_path = os.path.join(base_dir, "speakers_xtts.pth")

    model.language_manager = LanguageManager(config)
    model.speaker_manager = None
    if os.path.exists(speaker_path):
        model.speaker_manager = SpeakerManager(speaker_path)
    if os.path.exists(vocab_path):
        model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    model.init_models()
    gc.collect()

    logger.info("  [2/5] Loading fine-tuned checkpoint (~1.76 GB)…")
    ft_path = os.path.join(model_dir, "model.pth")
    try:
        ft_state = torch.load(ft_path, map_location="cpu", weights_only=True)
    except Exception:
        logger.info("  weights_only=True failed, retrying with weights_only=False…")
        ft_state = torch.load(ft_path, map_location="cpu", weights_only=False)

    if isinstance(ft_state, dict) and "model" in ft_state:
        ft_state = ft_state["model"]

    ignore_prefixes = ("torch_mel_spectrogram_style_encoder",
                       "torch_mel_spectrogram_dvae", "dvae")
    for key in list(ft_state.keys()):
        if key.startswith("xtts."):
            ft_state[key.replace("xtts.", "", 1)] = ft_state.pop(key)
    for key in list(ft_state.keys()):
        if key.split(".")[0] in ignore_prefixes:
            del ft_state[key]

    ft_state = _remap_trainer_keys(ft_state)
    logger.info("  [2/5] Fine-tuned state: %d keys", len(ft_state))
    gc.collect()

    logger.info("  [3/5] Extracting HiFi-GAN weights from base model…")
    base_model_path = os.path.join(base_dir, "model.pth")
    base_keys = _extract_base_non_gpt_keys(base_model_path)

    logger.info("  [4/5] Merging fine-tuned GPT + base HiFi-GAN (%d + %d keys)…",
                len(ft_state), len(base_keys))
    merged = {**base_keys, **ft_state}
    del ft_state, base_keys
    gc.collect()

    mel_stats_path = os.path.join(model_dir, "mel_stats.pth")
    if "mel_stats" not in merged and os.path.exists(mel_stats_path):
        merged["mel_stats"] = torch.load(mel_stats_path, map_location="cpu", weights_only=True)

    result = model.load_state_dict(merged, strict=False)
    if result.missing_keys:
        logger.info("  Missing keys (expected): %d", len(result.missing_keys))
    if result.unexpected_keys:
        logger.warning("  Unexpected keys: %s", result.unexpected_keys[:5])
    del merged
    gc.collect()

    logger.info("  [5/5] Switching to eval / inference mode…")
    model.hifigan_decoder.eval()
    model.gpt.init_gpt_for_inference(kv_cache=model.args.kv_cache, use_deepspeed=False)
    model.gpt.eval()
    model.to(settings.tts_device)
    gc.collect()

    return model


def synthesize(text: str) -> bytes:
    if _model is None:
        raise RuntimeError("TTS model not loaded — call load_model() first")

    logger.info("Synthesizing: %s", text[:100])

    out = _model.inference(
        text=text,
        language="en",
        gpt_cond_latent=_gpt_cond_latent,
        speaker_embedding=_speaker_embedding,
    )

    audio_array = np.asarray(out["wav"], dtype=np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, samplerate=24000, format="WAV")
    buffer.seek(0)

    logger.info("Synthesized %d bytes of audio", buffer.getbuffer().nbytes)
    return buffer.read()


def is_ready() -> bool:
    return _model is not None
