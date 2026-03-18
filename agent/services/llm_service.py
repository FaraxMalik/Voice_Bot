import json
import logging
import time
from collections import defaultdict

import httpx

from agent.config import get_settings

logger = logging.getLogger(__name__)

_histories: dict[str, list[dict]] = defaultdict(list)


def _normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower()


def _list_available_models() -> set[str]:
    """Return the exact model names currently available in Ollama."""
    settings = get_settings()
    response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=10.0)
    response.raise_for_status()

    models = response.json().get("models", [])
    names: set[str] = set()
    for model in models:
        for key in ("name", "model"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                names.add(_normalize_model_name(value))

    return names


def _wait_for_ollama(max_wait: int = 120) -> None:
    """Block until the Ollama server is reachable."""
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/tags"
    waited = 0
    while waited < max_wait:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                logger.info("Ollama server is reachable")
                return
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(3)
        waited += 3
        logger.info("Waiting for Ollama server… (%ds)", waited)
    raise RuntimeError(f"Ollama not reachable at {settings.ollama_base_url} after {max_wait}s")


def _model_exists() -> bool:
    """Check if the configured model is already pulled in Ollama."""
    settings = get_settings()
    try:
        available_models = _list_available_models()
        configured_model = _normalize_model_name(settings.ollama_model)
        return configured_model in available_models
    except httpx.HTTPError:
        return False


def _pull_model() -> None:
    """Pull the LLaMA model into Ollama if not already present."""
    settings = get_settings()
    if _model_exists():
        logger.info("LLM model '%s' already available in Ollama", settings.ollama_model)
        return

    logger.info("LLM model '%s' not found — downloading (this may take several minutes)…", settings.ollama_model)
    url = f"{settings.ollama_base_url}/api/pull"
    payload = {"name": settings.ollama_model, "stream": True}

    with httpx.stream("POST", url, json=payload, timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)) as response:
        response.raise_for_status()
        last_log = 0
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                status = data.get("status", "")
                completed = data.get("completed", 0)
                total = data.get("total", 0)
                if total and completed:
                    pct = int(completed / total * 100)
                    if pct >= last_log + 10:
                        logger.info("  Downloading %s: %d%%", settings.ollama_model, pct)
                        last_log = pct
                elif status:
                    logger.info("  %s", status)
            except Exception:
                pass

    logger.info("LLM model '%s' downloaded successfully", settings.ollama_model)


def ensure_model() -> None:
    """Wait for Ollama, check if model exists, download if missing."""
    _wait_for_ollama()
    _pull_model()


def _build_messages(session_id: str, user_text: str) -> list[dict]:
    settings = get_settings()
    history = _histories[session_id]

    messages = [{"role": "system", "content": settings.agent_system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def _trim_history(session_id: str) -> None:
    settings = get_settings()
    history = _histories[session_id]
    max_turns = settings.agent_max_history * 2
    if len(history) > max_turns:
        _histories[session_id] = history[-max_turns:]


def chat(user_text: str, session_id: str = "default") -> str:
    """Send user text to Ollama/LLaMA and return the assistant reply."""
    settings = get_settings()
    messages = _build_messages(session_id, user_text)

    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
    }

    logger.info("Sending to LLM [session=%s]: %s", session_id, user_text[:100])

    try:
        response = httpx.post(url, json=payload, timeout=600.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("LLM request failed: %s", exc)
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    data = response.json()
    reply = data.get("message", {}).get("content", "").strip()

    _histories[session_id].append({"role": "user", "content": user_text})
    _histories[session_id].append({"role": "assistant", "content": reply})
    _trim_history(session_id)

    logger.info("LLM reply [session=%s]: %s", session_id, reply[:100])
    return reply


def clear_history(session_id: str = "default") -> None:
    _histories.pop(session_id, None)


def is_ready() -> bool:
    return _model_exists()
