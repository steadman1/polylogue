import importlib.util
import platform
import sys

from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.text_to_text_models.gguf_model import GGUFModel
from polylogue.inference.text_to_text_models.mlx_model import MLXModel


class ModelTypeUnsupportedError(Exception):
    pass


def get_supported_model_types() -> set[type[InferenceModel]]:
    supported: set[type[InferenceModel]] = set()

    supports_llama_cpp = can_run_llama_cpp()
    if supports_llama_cpp:
        supported.add(GGUFModel)

    supports_mlx_lm = can_run_mlx_lm()
    if supports_mlx_lm:
        supported.add(MLXModel)

    return supported


def can_run_llama_cpp() -> bool:
    """Checks if llama_cpp is available to be imported"""
    llama_lib = importlib.util.find_spec("llama_cpp")
    return llama_lib is not None


# written by Gemini
def can_run_mlx_lm() -> bool:
    """Verifies OS, Apple Silicon architecture, and MLX Metal device availability."""
    # 1. Check OS is macOS
    if sys.platform != "darwin":
        return False

    # 2. Check architecture is Apple Silicon (arm64 / aarch64)
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        return False

    # 3. Check MLX / MLX-LM imports and Metal GPU backend availability
    try:
        import mlx.core as mx

        # Verify the Metal backend is functional and default device is available
        return mx.metal.is_available()
    except (ImportError, AttributeError):
        return False
