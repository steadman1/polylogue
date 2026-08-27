import mlx_lm
import llama_cpp
from pathlib import Path
from openai.types import Completion, CompletionChoice
from collections.abc import AsyncIterator
from datetime import datetime
import asyncio
import os
from typing import final
from polylogue.constants import MLX_TARGET, LLAMA_TARGET

@final
class TextToTextEngine:
    
    model: TextToTextModel
    
    model_path: Path
    model_name: str

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_name = model_path.parts[-1]

    def load(self) -> bool:
        if not self.model_path.exists():
            return False

        if self.model_path.is_file() and self.model_name.endswith(LLAMA_TARGET):
            # need to decide whether to use mlx or llama here based on file type/directory details
            llama_model = llama_cpp.Llama()
            
        
        if self.model_path.is_dir() and (self.model_path / MLX_TARGET).is_file().
            pass
        # *.gguf files are handled by llama cpp

        
        
        # mlx should check for a config.json and safetensors in target directory
        return False

    def destroy(self) -> None:
        ...


    # generation should format the raw dictionary into an openai Completion
    def generate(self) -> Completion:
        ...

    def stream_generate(self) -> AsyncIterator[Completion]:
        ...