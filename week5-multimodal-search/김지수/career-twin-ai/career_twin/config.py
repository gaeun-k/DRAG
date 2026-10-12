from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("CAREER_DATA_DIR", str(ROOT / "data")))
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen2.5:3b")
    embed_model: str = os.getenv("EMBED_MODEL", "embeddinggemma")
    vision_model: str = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
    dataset_id: str = "jminc/resume-matching-dataset-v2"
    max_file_bytes: int = 20 * 1024 * 1024
    max_pdf_pages: int = 150


SETTINGS = Settings()
