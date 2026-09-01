from typing import Any, cast


# written by Gemini
def flatten_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flattens multimodal/structured content blocks in chat messages into plain text strings.

    Args:
        messages: A list of message dictionaries with string roles and either
                  string content or a list of content part dictionaries.

    Returns:
        A list of dictionaries where every 'content' value is a flattened string.
    """
    flattened: list[dict[str, str]] = []

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content")

        if content is None:
            content_str = ""
        elif isinstance(content, str):
            content_str = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    # Extract text from OpenAI-style blocks {"type": "text", "text": "..."}
                    if part.get("type") == "text" and "text" in part:
                        parts.append(str(part["text"]))
                    elif "text" in part:
                        parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            content_str = "".join(parts)
        else:
            content_str = str(content)

        flattened.append({"role": role, "content": content_str})

    return flattened
