from polylogue.db.model_record import ModelRecord
from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.text_to_text_engine import TextToTextEngine
from polylogue.inference.text_to_text_factory import TextToTextFactory


class ModelCacheManager:
    def __init__(self, max_loaded: int = 1) -> None:
        self.max_loaded = max_loaded
        self._loaded_models: dict[str, tuple[InferenceModel, TextToTextEngine]] = {}

    def get_or_load(self, model_record: ModelRecord) -> TextToTextEngine:
        # 1. Return cached engine if model is already in memory
        if model_record.id in self._loaded_models:
            _, engine = self._loaded_models[model_record.id]
            return engine

        # 2. Evict least recently used / older models if capacity reached
        while len(self._loaded_models) >= self.max_loaded:
            _, (old_model, _) = self._loaded_models.popitem()
            if hasattr(old_model, "destroy"):
                old_model.destroy()

        # 3. Load model into memory
        model: InferenceModel = TextToTextFactory.from_path(model_record)
        if hasattr(model, "load"):
            model.load()

        engine = TextToTextEngine(model, model_record.id)
        self._loaded_models[model_record.id] = (model, engine)
        return engine

    def shutdown(self) -> None:
        for _, (model, _) in self._loaded_models.items():
            if hasattr(model, "destroy"):
                model.destroy()
        self._loaded_models.clear()
