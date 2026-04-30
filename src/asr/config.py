from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", case_sensitive=False)

    host: str = "0.0.0.0"
    port: int = 8000
    allow_cpu: bool = False
    max_file_bytes: int = 100 * 1024 * 1024
    max_audio_seconds: float = 600.0
    queue_depth: int = 4
    default_model: str = "parakeet-tdt-0.6b-v2"
    enabled_models: str = "parakeet-tdt-0.6b-v2,seamless-m4t-v2"

    @property
    def enabled_model_ids(self) -> list[str]:
        return [m.strip() for m in self.enabled_models.split(",") if m.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_tests() -> None:
    global _settings
    _settings = None
