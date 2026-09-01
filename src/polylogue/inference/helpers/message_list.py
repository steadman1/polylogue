from collections import UserList
from collections.abc import Iterable
from typing import Any, Sequence

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from polylogue.inference.helpers.flatten_chat_messages import flatten_chat_messages


# written by Gemini
class MessageList(UserList[Any]):
    """A list container for chat message objects providing flattening and normalization."""

    def clean(self) -> Sequence[ChatCompletionMessageParam]:
        """Flattens nested content parts and normalizes messages to ChatCompletionMessageParam."""
        # 1. Convert any Pydantic models to dicts
        raw_dicts: list[dict[str, Any]] = [
            m.model_dump()
            if hasattr(m, "model_dump")
            else (
                m.dict() if hasattr(m, "dict") else dict(m)  # type: ignore[arg-type]
            )
            for m in self.data
        ]

        # 2. Flatten multimodal blocks in-place
        flattened = flatten_chat_messages(raw_dicts)

        # 3. Construct strictly typed ChatCompletionMessageParam dicts
        clean_messages: list[ChatCompletionMessageParam] = []
        for m in flattened:
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))

            if role == "system":
                clean_messages.append(
                    ChatCompletionSystemMessageParam(role="system", content=content)
                )
            elif role == "developer":
                clean_messages.append(
                    ChatCompletionDeveloperMessageParam(
                        role="developer", content=content
                    )
                )
            elif role == "assistant":
                clean_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=content
                    )
                )
            elif role == "user":
                clean_messages.append(
                    ChatCompletionUserMessageParam(role="user", content=content)
                )
            else:
                # Fallback for tool or custom roles
                clean_messages.append(
                    ChatCompletionUserMessageParam(role=role, content=content)
                )

        return clean_messages
