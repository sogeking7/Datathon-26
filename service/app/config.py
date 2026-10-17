from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    api_key: str = "dev-key-change-me"
    data_dir: Path = Path("data")
    max_upload_mb: int = 300
    workers: int = 2
    # ТЗ: результат нужен за 5 минут; задача, которая не уложилась, помечается failed.
    job_timeout_sec: int = 280
    allowed_extensions: tuple[str, ...] = (".mov", ".mp4", ".m4v", ".avi", ".mkv")


settings = Settings()
