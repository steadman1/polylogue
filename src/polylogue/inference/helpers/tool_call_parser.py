from __future__ import annotations

import json
import re
import uuid

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from polylogue.inference.chat_templates.chat_constants import (
    Delimiters,
    Patterns,
    ToolCallFormat,
)


class ToolCallParser:
    """Universal parser for extracting structured tool calls across model families."""

    @classmethod
    def parse(
        cls,
        text: str,
        fmt: ToolCallFormat = ToolCallFormat.XML_HERMES,
    ) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        if fmt == ToolCallFormat.GEMMA:
            return cls._parse_gemma(text)
        if fmt == ToolCallFormat.LLAMA:
            return cls._parse_llama(text)
        if fmt == ToolCallFormat.MISTRAL:
            return cls._parse_mistral(text)
        return cls._parse_xml(text)

    @classmethod
    def _parse_xml(cls, text: str) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        tool_calls: list[ChatCompletionMessageToolCall] = []

        def replacer(match: re.Match[str]) -> str:
            payload = match.group(1).strip()
            call = cls._extract_call_from_raw(payload)
            if call:
                tool_calls.append(call)
            return ""

        cleaned_text = Patterns.XML_ENVELOPE.sub(replacer, text).strip()
        return cleaned_text, tool_calls

    @classmethod
    def _parse_gemma(cls, text: str) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        tool_calls: list[ChatCompletionMessageToolCall] = []

        def replacer(match: re.Match[str]) -> str:
            payload = match.group(1).strip()
            call = cls._extract_gemma_call(payload)
            if call:
                tool_calls.append(call)
            return ""

        cleaned_text = Patterns.GEMMA_ENVELOPE.sub(replacer, text).strip()
        return cleaned_text, tool_calls

    @classmethod
    def _extract_gemma_call(cls, payload: str) -> ChatCompletionMessageToolCall | None:
        try:
            data = json.loads(payload)
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments", data.get("parameters", {}))
                return ChatCompletionMessageToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=Function(
                        name=data["name"],
                        arguments=json.dumps(args)
                        if isinstance(args, dict)
                        else str(args),
                    ),
                )
        except Exception:
            pass

        fn_match = Patterns.GEMMA_CALL.match(payload)
        if not fn_match:
            return None

        fn_name = fn_match.group(1).strip()
        raw_args = fn_match.group(2).strip()

        normalized = raw_args.replace(Delimiters.GEMMA_QUOTE_ESCAPE, '"')
        normalized = Patterns.JSON_UNQUOTED_KEY.sub(r'\1"\2":', normalized)

        try:
            parsed_json = json.loads(normalized)
            arguments = json.dumps(parsed_json)
        except json.JSONDecodeError:
            arguments = normalized

        return ChatCompletionMessageToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=Function(name=fn_name, arguments=arguments),
        )

    @classmethod
    def _parse_llama(cls, text: str) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        tool_calls: list[ChatCompletionMessageToolCall] = []

        def replacer(match: re.Match[str]) -> str:
            payload = match.group(1).strip()
            call = cls._extract_call_from_raw(payload)
            if call:
                tool_calls.append(call)
            return ""

        cleaned = Patterns.LLAMA_ENVELOPE.sub(replacer, text).strip()
        return cleaned, tool_calls

    @classmethod
    def _parse_mistral(
        cls, text: str
    ) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        tool_calls: list[ChatCompletionMessageToolCall] = []

        def replacer(match: re.Match[str]) -> str:
            payload = match.group(1).strip()
            try:
                data = json.loads(payload)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    name = item.get("name", "")
                    args = item.get("arguments", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            type="function",
                            function=Function(name=name, arguments=args_str),
                        )
                    )
            except Exception:
                pass
            return ""

        cleaned = Patterns.MISTRAL_ENVELOPE.sub(replacer, text).strip()
        return cleaned, tool_calls

    @classmethod
    def _extract_call_from_raw(
        cls, payload: str
    ) -> ChatCompletionMessageToolCall | None:
        try:
            data = json.loads(payload)
            name = data.get("name", "")
            args = data.get("arguments", data.get("parameters", {}))
            return ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=Function(
                    name=name,
                    arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                ),
            )
        except Exception:
            pass

        name_match = Patterns.XML_INNER_NAME.search(payload)
        args_match = Patterns.XML_INNER_ARGS.search(payload)
        if name_match:
            name = name_match.group(1).strip()
            arguments = args_match.group(1).strip() if args_match else "{}"
            return ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=Function(name=name, arguments=arguments),
            )
        return None
