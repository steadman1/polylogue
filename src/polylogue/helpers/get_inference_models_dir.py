from pathlib import Path

from polylogue.constants import INFERENCE_MODELS_DIR


def get_inference_models_dir() -> Path:
    if not INFERENCE_MODELS_DIR:
        raise ValueError("Set a INFERENCE_MODELS_DIR environment variable")

    return Path(INFERENCE_MODELS_DIR)
