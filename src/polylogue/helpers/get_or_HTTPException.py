from typing import Any

from pydantic import BaseModel, ConfigDict


class ValidatedCompletionCreateParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a field or extra attribute, falling back to default."""
        if hasattr(self, key):
            value = getattr(self, key)
            return value if value is not None else default

        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]

        return default

    def get_or_422(self, field: str) -> Any:
        """Retrieves a field value or raises a 422 error."""
        value = getattr(self, field, None)
        if value is None and self.model_extra and field in self.model_extra:
            value = self.model_extra[field]

        if value is None:
            raise HTTPException(f"Field {value} is undefined")
        return value


class HTTPException(Exception):
    pass
