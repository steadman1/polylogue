import mlx_lm
import llama_cpp
from pathlib import Path
from openai.types import Completion, CompletionChoice
from collections.abc import AsyncIterator
from datetime import datetime
import asyncio
from typing import final
from polylogue.objects.text_to_text_engine import TextToTextEngine

@final
class TextToTextEngine:
    
    model = None
    tokenizer = None
    model_path: Path
    model_name: str

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_name = model_path.parts[-1]

    def load(self) -> bool:
        # need to decide whether to use mlx or llama here based on file type/directory details

        # *.gguf files are handled by llama cpp

        
        
        # mlx should check for a config.json and safetensors in target directory

    def destroy(self) -> None:
        ...

    def generate(self) -> dict[str, str]:
        ...

    def stream_generate(self) -> AsyncIterator[dict[str, str]]:
        ...