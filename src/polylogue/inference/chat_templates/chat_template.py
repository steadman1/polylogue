from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from polylogue.inference.chat_templates.chat_constants import (
    Delimiters,
    Patterns,
    ProbeMarkers,
    ToolCallFormat,
)
from polylogue.inference.helpers.tool_call_parser import ToolCallParser


@dataclass(frozen=True)
class ChatTemplateConstants:
    system_start: str = ""
    system_end: str = ""
    user_start: str = ""
    user_end: str = ""
    assistant_start: str = ""
    assistant_end: str = ""
    think_start: str = Delimiters.THINK_START
    think_end: str = Delimiters.THINK_END
    tool_call_start: str = Delimiters.XML_TOOL_CALL_START
    tool_call_end: str = Delimiters.XML_TOOL_CALL_END
    tool_response_start: str = Delimiters.XML_TOOL_RESPONSE_START
    tool_response_end: str = Delimiters.XML_TOOL_RESPONSE_END
    tool_format: ToolCallFormat = ToolCallFormat.XML_HERMES
    generation_prompt: str = ""


class ChatTemplateDetector:
    """Extracts chat template delimiters, reasoning tags, and tool conventions."""

    @staticmethod
    def detect_tool_delimiters(
        template_str: str,
    ) -> tuple[str, str, str, str, ToolCallFormat]:
        """Returns:
        (tool_call_start, tool_call_end, tool_response_start, tool_response_end, tool_format)
        """
        if not template_str:
            return (
                Delimiters.XML_TOOL_CALL_START,
                Delimiters.XML_TOOL_CALL_END,
                Delimiters.XML_TOOL_RESPONSE_START,
                Delimiters.XML_TOOL_RESPONSE_END,
                ToolCallFormat.XML_HERMES,
            )

        # 1. Gemma / Gemma 4
        if (
            Delimiters.GEMMA_TOOL_CALL_START in template_str
            or Delimiters.GEMMA_TOOL_CALL_END in template_str
            or Delimiters.GEMMA_TOOL_RESPONSE_START in template_str
            or "<|tool_call>" in template_str
            or "<|tool_response>" in template_str
        ):
            return (
                Delimiters.GEMMA_TOOL_CALL_START,
                Delimiters.GEMMA_TOOL_CALL_END,
                Delimiters.GEMMA_TOOL_RESPONSE_START,
                Delimiters.GEMMA_TOOL_RESPONSE_END,
                ToolCallFormat.GEMMA,
            )

        if Delimiters.GEMMA_START_OF_TURN in template_str:
            if Delimiters.GEMMA_CALL_PREFIX in template_str:
                return (
                    Delimiters.GEMMA_CALL_PREFIX,
                    "\n",
                    Delimiters.GEMMA_TOOL_RESPONSE_START,
                    Delimiters.GEMMA_TOOL_RESPONSE_END,
                    ToolCallFormat.GEMMA,
                )
            return (
                Delimiters.GEMMA_TOOL_CALL_START,
                Delimiters.GEMMA_TOOL_CALL_END,
                Delimiters.GEMMA_TOOL_RESPONSE_START,
                Delimiters.GEMMA_TOOL_RESPONSE_END,
                ToolCallFormat.GEMMA,
            )

        # 2. Llama 3.1 / 3.2 function calling tags
        if Delimiters.LLAMA_PYTHON_TAG in template_str:
            return (
                Delimiters.LLAMA_PYTHON_TAG,
                Delimiters.LLAMA_EOM_ID,
                Delimiters.LLAMA_IPYTHON_START,
                Delimiters.LLAMA_EOT_ID,
                ToolCallFormat.LLAMA,
            )

        # 3. Mistral / Mixtral [TOOL_CALLS]
        if Delimiters.MISTRAL_TOOL_START in template_str:
            end_tag = (
                Delimiters.MISTRAL_TOOL_END
                if Delimiters.MISTRAL_TOOL_END in template_str
                else ""
            )
            return (
                Delimiters.MISTRAL_TOOL_START,
                end_tag,
                Delimiters.MISTRAL_TOOL_RESPONSE_START,
                Delimiters.MISTRAL_TOOL_RESPONSE_END,
                ToolCallFormat.MISTRAL,
            )

        # 4. DeepSeek tool calling
        if (
            Delimiters.DEEPSEEK_TOOL_START in template_str
            or Delimiters.DEEPSEEK_TOOLS_WRAP_START in template_str
        ):
            return (
                Delimiters.DEEPSEEK_TOOL_START,
                Delimiters.DEEPSEEK_TOOL_END,
                Delimiters.DEEPSEEK_TOOL_RESPONSE_START,
                Delimiters.DEEPSEEK_TOOL_RESPONSE_END,
                ToolCallFormat.DEEPSEEK,
            )

        # 5. Explicit XML / Hermes syntax
        if Delimiters.XML_TOOL_CALL_START in template_str:
            return (
                Delimiters.XML_TOOL_CALL_START,
                Delimiters.XML_TOOL_CALL_END,
                Delimiters.XML_TOOL_RESPONSE_START,
                Delimiters.XML_TOOL_RESPONSE_END,
                ToolCallFormat.XML_HERMES,
            )

        # 6. Fallback
        return (
            Delimiters.XML_TOOL_CALL_START,
            Delimiters.XML_TOOL_CALL_END,
            Delimiters.XML_TOOL_RESPONSE_START,
            Delimiters.XML_TOOL_RESPONSE_END,
            ToolCallFormat.UNKNOWN,
        )

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> ChatTemplateConstants:
        raw_template = getattr(tokenizer, "chat_template", "") or ""
        (
            tool_start,
            tool_end,
            tool_resp_start,
            tool_resp_end,
            tool_format,
        ) = cls.detect_tool_delimiters(raw_template)

        # Reasoning detection
        think_start = Delimiters.THINK_START
        think_end = Delimiters.THINK_END
        think_match = Patterns.THINK_OPENING.search(raw_template)
        if think_match:
            think_start = think_match.group(1)

        think_end_match = Patterns.THINK_CLOSING.search(raw_template)
        if think_end_match:
            think_end = think_end_match.group(1)

        # 1. System Prompt Probe
        try:
            sys_render = tokenizer.apply_chat_template(
                [{"role": "system", "content": ProbeMarkers.SYSTEM}],
                tokenize=False,
                add_generation_prompt=False,
            )
            sys_start, sys_end = cls._split_around_marker(
                sys_render, ProbeMarkers.SYSTEM
            )
        except Exception:
            sys_start, sys_end = "", ""

        # 2. User Prompt Probe
        user_render = tokenizer.apply_chat_template(
            [{"role": "user", "content": ProbeMarkers.USER}],
            tokenize=False,
            add_generation_prompt=False,
        )
        user_start, user_end = cls._split_around_marker(user_render, ProbeMarkers.USER)

        # 3. Assistant Prompt + Generation Prompt
        asst_render_with_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ProbeMarkers.USER_DUMMY_PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        asst_render_without_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ProbeMarkers.USER_DUMMY_PROMPT}],
            tokenize=False,
            add_generation_prompt=False,
        )
        gen_prompt = asst_render_with_prompt[len(asst_render_without_prompt) :]

        asst_render = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": ProbeMarkers.USER_DUMMY_TURN},
                {"role": "assistant", "content": ProbeMarkers.ASSISTANT},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        asst_start, asst_end = cls._split_around_marker(
            asst_render, ProbeMarkers.ASSISTANT
        )

        return ChatTemplateConstants(
            system_start=sys_start,
            system_end=sys_end,
            user_start=user_start,
            user_end=user_end,
            assistant_start=asst_start,
            assistant_end=asst_end,
            think_start=think_start,
            think_end=think_end,
            tool_call_start=tool_start,
            tool_call_end=tool_end,
            tool_response_start=tool_resp_start,
            tool_response_end=tool_resp_end,
            tool_format=tool_format,
            generation_prompt=gen_prompt,
        )

    @classmethod
    def from_model_dir(cls, model_path: Path | str) -> ChatTemplateConstants:
        model_dir = Path(model_path)
        config_file = model_dir / "tokenizer_config.json"
        tokenizer_file = model_dir / "tokenizer.json"

        data: dict = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                data.update(json.load(f))

        tokenizer_data: dict = {}
        if tokenizer_file.exists():
            with open(tokenizer_file, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)

        raw_template = data.get("chat_template", "")

        # 1. First attempt detection via Jinja template string
        (
            tool_start,
            tool_end,
            tool_resp_start,
            tool_resp_end,
            tool_format,
        ) = cls.detect_tool_delimiters(raw_template)

        # 2. Schema / Tokenizer Fallback for Gemma / Gemma 4
        is_gemma_tokenizer = (
            "Gemma" in data.get("tokenizer_class", "")
            or "Gemma" in tokenizer_data.get("tokenizer_class", "")
            or "Gemma" in data.get("processor_class", "")
            or "Gemma" in tokenizer_data.get("processor_class", "")
        )

        response_schema = data.get("response_schema") or tokenizer_data.get(
            "response_schema", {}
        )
        special_tokens = {
            **data.get("model_specific_special_tokens", {}),
            **tokenizer_data.get("model_specific_special_tokens", {}),
        }

        if is_gemma_tokenizer or "gemma" in str(response_schema).lower():
            tool_start = (
                data.get("stc_token")
                or tokenizer_data.get("stc_token")
                or special_tokens.get("stc_token")
                or Delimiters.GEMMA_TOOL_CALL_START
            )
            tool_end = (
                data.get("etc_token")
                or tokenizer_data.get("etc_token")
                or special_tokens.get("etc_token")
                or Delimiters.GEMMA_TOOL_CALL_END
            )
            tool_resp_start = (
                data.get("str_token")
                or tokenizer_data.get("str_token")
                or special_tokens.get("str_token")
                or Delimiters.GEMMA_TOOL_RESPONSE_START
            )
            tool_resp_end = (
                data.get("etr_token")
                or tokenizer_data.get("etr_token")
                or special_tokens.get("etr_token")
                or Delimiters.GEMMA_TOOL_RESPONSE_END
            )
            tool_format = ToolCallFormat.GEMMA

        # 3. Resolve Template Roles
        if Delimiters.CHATML_IM_START in raw_template:
            return ChatTemplateConstants(
                system_start=f"{Delimiters.CHATML_IM_START}system\n",
                system_end=f"{Delimiters.CHATML_IM_END}\n",
                user_start=f"{Delimiters.CHATML_IM_START}user\n",
                user_end=f"{Delimiters.CHATML_IM_END}\n",
                assistant_start=f"{Delimiters.CHATML_IM_START}assistant\n",
                assistant_end=f"{Delimiters.CHATML_IM_END}\n",
                think_start=Delimiters.THINK_START,
                think_end=Delimiters.THINK_END,
                tool_call_start=tool_start,
                tool_call_end=tool_end,
                tool_response_start=tool_resp_start,
                tool_response_end=tool_resp_end,
                tool_format=tool_format,
                generation_prompt=f"{Delimiters.CHATML_IM_START}assistant\n",
            )

        if Delimiters.LLAMA_HEADER_START in raw_template:
            return ChatTemplateConstants(
                system_start=f"{Delimiters.LLAMA_HEADER_START}system{Delimiters.LLAMA_HEADER_END}\n\n",
                system_end=Delimiters.LLAMA_EOT_ID,
                user_start=f"{Delimiters.LLAMA_HEADER_START}user{Delimiters.LLAMA_HEADER_END}\n\n",
                user_end=Delimiters.LLAMA_EOT_ID,
                assistant_start=f"{Delimiters.LLAMA_HEADER_START}assistant{Delimiters.LLAMA_HEADER_END}\n\n",
                assistant_end=Delimiters.LLAMA_EOT_ID,
                think_start=Delimiters.THINK_START,
                think_end=Delimiters.THINK_END,
                tool_call_start=tool_start,
                tool_call_end=tool_end,
                tool_response_start=tool_resp_start,
                tool_response_end=tool_resp_end,
                tool_format=tool_format,
                generation_prompt=f"{Delimiters.LLAMA_HEADER_START}assistant{Delimiters.LLAMA_HEADER_END}\n\n",
            )

        if Delimiters.GEMMA_START_OF_TURN in raw_template or is_gemma_tokenizer:
            soc = (
                data.get("soc_token")
                or tokenizer_data.get("soc_token")
                or Delimiters.GEMMA_CHANNEL_START
            )
            eoc = (
                data.get("eoc_token")
                or tokenizer_data.get("eoc_token")
                or Delimiters.GEMMA_CHANNEL_END
            )
            think_start = f"{soc}thought\n" if soc else Delimiters.THOUGHT_START
            think_end = eoc if eoc else Delimiters.THOUGHT_END

            sot = (
                data.get("sot_token")
                or tokenizer_data.get("sot_token")
                or Delimiters.GEMMA_START_OF_TURN
            )
            eot = (
                data.get("eot_token")
                or tokenizer_data.get("eot_token")
                or Delimiters.GEMMA_END_OF_TURN
            )

            return ChatTemplateConstants(
                system_start=f"{sot}system\n",
                system_end=f"{eot}\n",
                user_start=f"{sot}user\n",
                user_end=f"{eot}\n",
                assistant_start=f"{sot}model\n",
                assistant_end=f"{eot}\n",
                think_start=think_start,
                think_end=think_end,
                tool_call_start=tool_start,
                tool_call_end=tool_end,
                tool_response_start=tool_resp_start,
                tool_response_end=tool_resp_end,
                tool_format=tool_format,
                generation_prompt=f"{sot}model\n",
            )

        if (
            Delimiters.DEEPSEEK_ASSISTANT in raw_template
            or Delimiters.DEEPSEEK_USER in raw_template
        ):
            return ChatTemplateConstants(
                system_start="",
                system_end="",
                user_start=Delimiters.DEEPSEEK_USER,
                user_end="",
                assistant_start=Delimiters.DEEPSEEK_ASSISTANT,
                assistant_end=Delimiters.DEEPSEEK_EOS,
                think_start=Delimiters.THINK_START,
                think_end=Delimiters.THINK_END,
                tool_call_start=tool_start,
                tool_call_end=tool_end,
                tool_response_start=tool_resp_start,
                tool_response_end=tool_resp_end,
                tool_format=tool_format,
                generation_prompt=Delimiters.DEEPSEEK_ASSISTANT,
            )

        return ChatTemplateConstants(
            tool_call_start=tool_start,
            tool_call_end=tool_end,
            tool_response_start=tool_resp_start,
            tool_response_end=tool_resp_end,
            tool_format=tool_format,
        )

    @staticmethod
    def _extract_any_tool_calls(
        text: str,
        explicit_format: ToolCallFormat | None = None,
    ) -> tuple[str, list[ChatCompletionMessageToolCall]]:
        if explicit_format:
            return ToolCallParser.parse(text, fmt=explicit_format)

        candidates = [
            ToolCallFormat.XML_HERMES,
            ToolCallFormat.GEMMA,
            ToolCallFormat.LLAMA,
            ToolCallFormat.MISTRAL,
        ]
        for fmt in candidates:
            cleaned, calls = ToolCallParser.parse(text, fmt=fmt)
            if calls:
                return cleaned, calls

        return text, []

    @staticmethod
    def _serialize_tool_calls_to_native(
        tool_calls: list[ChatCompletionMessageToolCall],
        target_format: ToolCallFormat,
        tool_start: str,
        tool_end: str,
    ) -> str:
        rendered_calls: list[str] = []

        for call in tool_calls:
            fn_name = call.function.name
            try:
                args_obj = json.loads(call.function.arguments)
            except Exception:
                args_obj = call.function.arguments

            args_str = (
                json.dumps(args_obj) if isinstance(args_obj, dict) else str(args_obj)
            )

            if target_format == ToolCallFormat.GEMMA:
                rendered_calls.append(f"{tool_start}call:{fn_name}{args_str}{tool_end}")
            elif target_format == ToolCallFormat.LLAMA:
                payload = json.dumps({"name": fn_name, "parameters": args_obj})
                rendered_calls.append(f"{tool_start}{payload}{tool_end}")
            elif target_format == ToolCallFormat.MISTRAL:
                payload = json.dumps({"name": fn_name, "arguments": args_obj})
                rendered_calls.append(f"{tool_start}[{payload}]{tool_end}")
            elif target_format == ToolCallFormat.XML_HERMES:
                rendered_calls.append(
                    f"{tool_start}\n"
                    f"<name>{fn_name}</name>\n"
                    f"<arguments>{args_str}</arguments>\n"
                    f"{tool_end}"
                )
            else:
                payload = json.dumps({"name": fn_name, "arguments": args_obj})
                rendered_calls.append(f"{tool_start}{payload}{tool_end}")

        return "\n".join(rendered_calls)

    @staticmethod
    def _split_around_marker(text: str, marker: str) -> tuple[str, str]:
        if marker not in text:
            return ("", "")
        prefix, suffix = text.split(marker, 1)
        return (prefix, suffix)
