from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"

    stt_model_size: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"

    tts_model_dir: str = "/app/models/tts"
    tts_reference_wav: str = "/app/models/tts/reference.wav"
    tts_device: str = "cpu"

    agent_system_prompt: str = (
        "You are a professional, helpful call agent. "
        "Respond concisely in 1-3 sentences. Be friendly and clear."
    )
    agent_max_history: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
