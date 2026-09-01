from pathlib import Path

from polylogue.constants import GGUF_TARGET, MLX_TARGET
from polylogue.helpers.get_supported_model_types import (
    ModelTypeUnsupportedError,
    get_supported_model_types,
)
from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.text_to_text_models.gguf_model import GGUFModel
from polylogue.inference.text_to_text_models.mlx_model import MLXModel


class TextToTextFactory:
    @staticmethod
    def from_path(model_path: Path) -> InferenceModel:
        model_name: str = model_path.parts[-1]
        # need to decide whether to use mlx, llama, ... here based on file type/directory details
        if model_path.is_file() and model_name.endswith(GGUF_TARGET):
            # *.gguf files are handled by llama cpp
            if GGUFModel in get_supported_model_types():
                return GGUFModel(model_path)
            else:
                raise ModelTypeUnsupportedError(
                    "Dependecy(s) required to run GGUF models are not available"
                )

        if model_path.is_dir() and (model_path / MLX_TARGET).is_file():
            # mlx should check for a config.json and safetensors in target directory
            if MLXModel in get_supported_model_types():
                return MLXModel(model_path)
            else:
                raise ModelTypeUnsupportedError(
                    "Dependecy(s) required to run MLX models are not available"
                )

        # don't show this directly on front-end since it would expose path details
        raise ModelResolutionError(f"Model couldn't be loaded from {model_path}")


class ModelResolutionError(Exception):
    pass
