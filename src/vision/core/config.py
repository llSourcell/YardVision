from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv


class VideoSettings(BaseModel):
    width: int = Field(default=1280, ge=64, description="Frame width in pixels")
    height: int = Field(default=720, ge=64, description="Frame height in pixels")
    fps: int = Field(default=30, ge=1, description="Target frames per second")


class ModelSettings(BaseModel):
    model_path: str = Field(default="models/dummy.onnx", description="Path to ONNX model")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    video: VideoSettings = Field(default_factory=VideoSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    analytics_enabled: bool = Field(default=True)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load application config.

    Priority: explicit file > .env > defaults.
    """
    load_dotenv()  # Allow env overrides
    # Future: parse from TOML/YAML if provided.
    # For now, simply return defaults; scaffolding point for extension.
    _ = path
    return AppConfig()

