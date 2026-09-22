from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    device: str | None = 'cpu'
    batch_size: int = 16


def load_settings():
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    model_dir = Path(os.getenv('CLASSIFICATION_MODEL_DIR', 'models/classification')).expanduser()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    device = os.getenv('CLASSIFICATION_DEVICE', 'cpu').strip().lower()
    if device not in {'cpu', 'cuda', 'auto'}:
        raise ValueError('CLASSIFICATION_DEVICE: cpu, cuda или auto')
    batch_size = int(os.getenv('CLASSIFICATION_BATCH_SIZE', '16'))
    if batch_size < 1:
        raise ValueError('CLASSIFICATION_BATCH_SIZE должен быть положительным')
    return Settings(model_dir.resolve(), None if device == 'auto' else device, batch_size)
