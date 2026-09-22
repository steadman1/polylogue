from __future__ import annotations

from collections.abc import Generator

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from polylogue.inference.chat_templates.chat_template import ChatTemplateConstants
from polylogue.inference.helpers.tool_call_parser import ToolCallParser


class StreamToolCallBuffer:
    """Manages token buffering and extracts tool invocations on partial stream streams."""

    def __init__(self, constants: ChatTemplateConstants) -> None:
        self.constants = constants
        self.start_tag = constants.tool_call_start
        self.end_tag = constants.tool_call_end
        self.format = constants.tool_format

        self.tag_buffer = ""
        self.tool_content = ""
        self.is_active = False
        self.emitted_call = False

    def process_token(
        self, token: str
    ) -> Generator[str | list[ChatCompletionMessageToolCall], None, None]:
        """
        Consumes an incoming text chunk and yields either:
        - Plaintext string segments (outside tool call envelope)
        - list[ChatCompletionMessageToolCall] when a tool call has fully terminated
        """
        combined = self.tag_buffer + token

        if not self.is_active:
            # 1. Searching for opening delimiter
            match = self._find_target_or_prefix(self.start_tag, combined)
            if not match:
                yield combined
                self.tag_buffer = ""
                return

            start_idx, end_idx = match
            prefix_text = combined[:start_idx]
            if prefix_text:
                yield prefix_text

            matched_fragment = combined[start_idx:end_idx]
            if matched_fragment == self.start_tag:
                self.is_active = True
                self.tag_buffer = ""
                self.tool_content = matched_fragment
            else:
                self.tag_buffer = matched_fragment

        else:
            # 2. Searching for closing delimiter
            match = self._find_target_or_prefix(self.end_tag, combined)
            if not match:
                self.tool_content += combined
                self.tag_buffer = ""
                return

            start_idx, end_idx = match
            matched_fragment = combined[start_idx:end_idx]

            if matched_fragment == self.end_tag:
                self.tool_content += combined[:start_idx] + self.end_tag
                _, calls = ToolCallParser.parse(self.tool_content, self.format)
                if calls:
                    self.emitted_call = True
                    yield calls

                # Reset state for possible subsequent calls or plain output
                self.tool_content = ""
                self.tag_buffer = ""
                self.is_active = False
            else:
                self.tool_content += combined[:start_idx]
                self.tag_buffer = matched_fragment

    def flush(self) -> Generator[str, None, None]:
        """Flushes trailing buffer data on stream completion."""
        trailing = self.tag_buffer or (
            self.tool_content if not self.emitted_call else ""
        )
        if trailing and not self.emitted_call:
            yield trailing

    @staticmethod
    def _find_target_or_prefix(target: str, text: str) -> tuple[int, int] | None:
        if not text or not target:
            return None

        # Full occurrence match
        idx = text.find(target)
        if idx != -1:
            return (idx, idx + len(target))

        # Partial tail overlap match (e.g. text ends with '<tool')
        max_overlap = min(len(text), len(target) - 1)
        for length in range(max_overlap, 0, -1):
            if text.endswith(target[:length]):
                return (len(text) - length, len(text))

        return None
