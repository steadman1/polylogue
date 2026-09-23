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
    def parse_payload(
        cls,
        payload: str,
        fmt: ToolCallFormat = ToolCallFormat.XML_HERMES,
    ) -> list[ChatCompletionMessageToolCall]:
        """Parses raw content extracted from between model tool call delimiters."""
        payload = payload.strip()
        if not payload:
            return []

        if fmt == ToolCallFormat.GEMMA:
            call = cls._extract_gemma_call(payload)
            return [call] if call else []

        if fmt == ToolCallFormat.MISTRAL:
            return cls._extract_mistral_payload(payload)

        # Hermes, Llama, and general XML/JSON tool calls
        call = cls._extract_call_from_raw(payload)
        return [call] if call else []

    @classmethod
    def parse(
        cls,
        text: str,
        fmt: ToolCallFormat = ToolCallFormat.XML_HERMES,
    ) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        """Parses full message text containing enveloped tool call tags."""
        if fmt == ToolCallFormat.GEMMA:
            return cls._parse_envelopes(
                text, Patterns.GEMMA_ENVELOPE, cls._extract_gemma_call
            )
        if fmt == ToolCallFormat.LLAMA:
            return cls._parse_envelopes(
                text, Patterns.LLAMA_ENVELOPE, cls._extract_call_from_raw
            )
        if fmt == ToolCallFormat.MISTRAL:
            return cls._parse_mistral(text)
        return cls._parse_envelopes(
            text, Patterns.XML_ENVELOPE, cls._extract_call_from_raw
        )

    @classmethod
    def _parse_envelopes(
        cls,
        text: str,
        pattern: re.Pattern[str],
        extractor_fn,
    ) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        tool_calls: list[ChatCompletionMessageToolCall] = []

        def replacer(match: re.Match[str]) -> str:
            payload = match.group(1).strip()
            call = extractor_fn(payload)
            if call:
                if isinstance(call, list):
                    tool_calls.extend(call)
                else:
                    tool_calls.append(call)
            return ""

        cleaned_text = pattern.sub(replacer, text).strip()
        return cleaned_text, tool_calls

    @classmethod
    def _parse_mistral(
        cls, text: str
    ) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        tool_calls: list[ChatCompletionMessageToolCall] = []

        def replacer(match: re.Match[str]) -> str:
            payload = match.group(1).strip()
            tool_calls.extend(cls._extract_mistral_payload(payload))
            return ""

        cleaned = Patterns.MISTRAL_ENVELOPE.sub(replacer, text).strip()
        return cleaned, tool_calls

    @classmethod
    def _extract_mistral_payload(
        cls, payload: str
    ) -> list[ChatCompletionMessageToolCall]:
        calls: list[ChatCompletionMessageToolCall] = []
        try:
            data = json.loads(payload)
            items = data if isinstance(data, list) else [data]
            for item in items:
                name = item.get("name", "")
                args = item.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        type="function",
                        function=Function(name=name, arguments=args_str),
                    )
                )
        except Exception:
            pass
        return calls

    @classmethod
    def _extract_gemma_call(cls, payload: str) -> ChatCompletionMessageToolCall | None:
        # First normalize any Gemma quote escape sequences in the raw payload
        normalized_payload = re.sub(r'<\|\\?"\|>', '"', payload)

        try:
            data = json.loads(normalized_payload)
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

        # Regex search for call:function_name{...} format
        fn_match = Patterns.GEMMA_CALL.search(normalized_payload)
        if not fn_match:
            return None

        fn_name = fn_match.group(1).strip()
        raw_args = fn_match.group(2).strip()

        # Normalize escape tokens in args (handles both <|"|> and <|\"|>)
        normalized = re.sub(r'<\|\\?"\|>', '"', raw_args)

        # Quote unquoted JSON keys: {key: "value"} -> {"key": "value"}
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
    def _convert_xml_arguments_to_json(cls, raw_args: str) -> str:
        """Converts XML parameters, parameter tags, or raw text to a valid JSON string."""
        raw_args = raw_args.strip()
        if not raw_args or raw_args == "{}":
            return "{}"

        # If already valid JSON
        try:
            parsed = json.loads(raw_args)
            return json.dumps(parsed) if isinstance(parsed, dict) else raw_args
        except json.JSONDecodeError:
            pass

        arg_dict: dict[str, str | int | float | bool | list | dict] = {}

        # 1. Matches attribute style: <parameter=key>value</parameter>
        param_attr_matches = re.findall(
            r"<parameter=([a-zA-Z0-9_-]+)>(.*?)</parameter>", raw_args, flags=re.DOTALL
        )
        for key, val in param_attr_matches:
            val_clean = val.strip()
            try:
                arg_dict[key] = json.loads(val_clean)
            except Exception:
                arg_dict[key] = val_clean

        # 2. Matches standard XML tags: <key>value</key>
        tag_matches = re.findall(
            r"<([a-zA-Z0-9_-]+)>(.*?)</\1>", raw_args, flags=re.DOTALL
        )
        for key, val in tag_matches:
            if key in ("parameter", "arguments"):
                continue
            val_clean = val.strip()
            try:
                arg_dict[key] = json.loads(val_clean)
            except Exception:
                arg_dict[key] = val_clean

        if arg_dict:
            return json.dumps(arg_dict)

        return "{}"

    @classmethod
    def _extract_call_from_raw(
        cls, payload: str
    ) -> ChatCompletionMessageToolCall | None:
        # 1. Standard raw JSON dictionary
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                return ChatCompletionMessageToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=Function(
                        name=name,
                        arguments=json.dumps(args)
                        if isinstance(args, dict)
                        else str(args),
                    ),
                )
        except Exception:
            pass

        # 2. Attribute-style syntax: <function=name>...</function>
        fn_attr_match = re.search(
            r"<function=([a-zA-Z0-9_.:-]+)>(.*?)</function>", payload, flags=re.DOTALL
        )
        if fn_attr_match:
            name = fn_attr_match.group(1).strip()
            inner_content = fn_attr_match.group(2).strip()
            arguments = cls._convert_xml_arguments_to_json(inner_content)
            return ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=Function(name=name, arguments=arguments),
            )

        # 3. XML tag syntax: <name>...</name> and <arguments>...</arguments>
        name_match = Patterns.XML_INNER_NAME.search(payload)
        args_match = Patterns.XML_INNER_ARGS.search(payload)
        if name_match:
            name = name_match.group(1).strip()
            raw_arguments = args_match.group(1).strip() if args_match else "{}"
            arguments = cls._convert_xml_arguments_to_json(raw_arguments)
            return ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=Function(name=name, arguments=arguments),
            )

        return None
