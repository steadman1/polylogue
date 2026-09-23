from __future__ import annotations

from collections.abc import Generator

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from polylogue.inference.chat_templates.chat_template import ChatTemplateConstants
from polylogue.inference.helpers.tool_call_parser import ToolCallParser


class StreamToolCallBuffer:
    """Manages token buffering and extracts tool invocations on partial stream chunks."""

    def __init__(self, constants: ChatTemplateConstants) -> None:
        self.constants = constants
        self.start_tag = constants.tool_call_start
        self.end_tag = constants.tool_call_end
        self.format = constants.tool_format

        self.buffer = ""
        self.tool_content = ""
        self.is_active = False

    def process_token(
        self, token: str
    ) -> Generator[str | list[ChatCompletionMessageToolCall], None, None]:
        self.buffer += token

        while self.buffer:
            if not self.is_active:
                # Check for complete start_tag
                start_idx = self.buffer.find(self.start_tag)
                if start_idx != -1:
                    # Yield any text leading up to start_tag
                    text_before = self.buffer[:start_idx]
                    if text_before:
                        yield text_before

                    self.is_active = True
                    self.tool_content = ""
                    # Advance buffer past start_tag
                    self.buffer = self.buffer[start_idx + len(self.start_tag) :]
                    continue

                # Check for possible partial start_tag at tail of buffer
                partial_len = self._get_partial_match_len(self.buffer, self.start_tag)
                if partial_len > 0:
                    # Yield characters confirmed not to belong to start_tag
                    safe_text = self.buffer[:-partial_len]
                    if safe_text:
                        yield safe_text
                    self.buffer = self.buffer[-partial_len:]
                    break
                else:
                    yield self.buffer
                    self.buffer = ""

            else:
                # Check for complete end_tag
                end_idx = self.buffer.find(self.end_tag)
                if end_idx != -1:
                    # Accumulate tool payload
                    self.tool_content += self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + len(self.end_tag) :]
                    self.is_active = False

                    # Parse extracted payload
                    calls = ToolCallParser.parse_payload(self.tool_content, self.format)

                    if calls:
                        yield calls
                    self.tool_content = ""
                    continue

                # Check for possible partial end_tag at tail of buffer
                partial_len = self._get_partial_match_len(self.buffer, self.end_tag)
                if partial_len > 0:
                    self.tool_content += self.buffer[:-partial_len]
                    self.buffer = self.buffer[-partial_len:]
                    break
                else:
                    self.tool_content += self.buffer
                    self.buffer = ""

    def flush(self) -> Generator[str | list[ChatCompletionMessageToolCall], None, None]:
        """Flushes trailing buffer data on stream completion."""
        if self.is_active:
            # Stream ended before closing tag: attempt fallback parse or emit raw
            self.tool_content += self.buffer

            calls = ToolCallParser.parse_payload(self.tool_content, self.format)
            if calls:
                yield calls
            else:
                yield self.start_tag + self.tool_content
            self.tool_content = ""
            self.buffer = ""
        else:
            if self.buffer:
                yield self.buffer
                self.buffer = ""

    @staticmethod
    def _get_partial_match_len(text: str, target: str) -> int:
        max_overlap = min(len(text), len(target) - 1)
        for length in range(max_overlap, 0, -1):
            if text.endswith(target[:length]):
                return length
        return 0
