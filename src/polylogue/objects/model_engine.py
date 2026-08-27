from pathlib import Path
from typing import Protocol, runtime_checkable
from collections.abc import AsyncIterator

# need to choose mlx_lm, llama_cpp, etc. to run each model
@runtime_checkable
class ModelEngine[SomeResponse](Protocol):

    model_path: Path

    def load(self, model_path) -> bool:
        ...

    def destroy(self) -> None:
        ...

    def generate(self) -> SomeResponse:
        ...

    def stream_generate(self) -> AsyncIterator[SomeResponse]:
        ...
