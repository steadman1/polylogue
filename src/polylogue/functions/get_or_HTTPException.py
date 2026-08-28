from typing import Any
from fastapi import HTTPException, status
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
        """Retrieves a field value or raises a standard FastAPI 422 error."""
        value = getattr(self, field, None)
        if value is None:
            # Fallback check for dynamic extra dictionary fields
            if self.model_extra and field in self.model_extra:
                value = self.model_extra[field]

        if value is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=[
                    {
                        "type": "missing",
                        "loc": ["body", field],
                        "msg": f"Field required: '{field}'",
                        "input": self.model_dump(),
                    }
                ],
            )
        return value
