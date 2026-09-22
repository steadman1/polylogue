from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.inference.chat_templates.chat_constants import (
    Delimiters,
    Patterns,
    ProbeMarkers,
    ToolCallFormat,
)


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
    tool_format: ToolCallFormat = ToolCallFormat.XML_HERMES
    generation_prompt: str = ""


class ChatTemplateDetector:
    """Extracts chat template delimiters, reasoning tags, and tool conventions."""

    @staticmethod
    def detect_tool_delimiters(template_str: str) -> tuple[str, str, ToolCallFormat]:
        if not template_str:
            return (
                Delimiters.XML_TOOL_CALL_START,
                Delimiters.XML_TOOL_CALL_END,
                ToolCallFormat.XML_HERMES,
            )

        if Delimiters.XML_TOOL_CALL_START in template_str:
            return (
                Delimiters.XML_TOOL_CALL_START,
                Delimiters.XML_TOOL_CALL_END,
                ToolCallFormat.XML_HERMES,
            )

        if Delimiters.GEMMA_START_OF_TURN in template_str:
            if (
                Delimiters.GEMMA_TOOL_CALL_START in template_str
                or Delimiters.GEMMA_TOOL_CALL_END in template_str
            ):
                return (
                    Delimiters.GEMMA_TOOL_CALL_START,
                    Delimiters.GEMMA_TOOL_CALL_END,
                    ToolCallFormat.GEMMA,
                )
            if Delimiters.GEMMA_CALL_PREFIX in template_str:
                return (Delimiters.GEMMA_CALL_PREFIX, "\n", ToolCallFormat.GEMMA)
            return (
                Delimiters.GEMMA_TOOL_CALL_START,
                Delimiters.GEMMA_TOOL_CALL_END,
                ToolCallFormat.GEMMA,
            )

        if Delimiters.LLAMA_PYTHON_TAG in template_str:
            return (
                Delimiters.LLAMA_PYTHON_TAG,
                Delimiters.LLAMA_EOM_ID,
                ToolCallFormat.LLAMA,
            )

        if Delimiters.MISTRAL_TOOL_START in template_str:
            end_tag = (
                Delimiters.MISTRAL_TOOL_END
                if Delimiters.MISTRAL_TOOL_END in template_str
                else ""
            )
            return (Delimiters.MISTRAL_TOOL_START, end_tag, ToolCallFormat.MISTRAL)

        if (
            Delimiters.DEEPSEEK_TOOL_START in template_str
            or Delimiters.DEEPSEEK_TOOLS_WRAP_START in template_str
        ):
            return (
                Delimiters.DEEPSEEK_TOOL_START,
                Delimiters.DEEPSEEK_TOOL_END,
                ToolCallFormat.DEEPSEEK,
            )

        return (
            Delimiters.XML_TOOL_CALL_START,
            Delimiters.XML_TOOL_CALL_END,
            ToolCallFormat.UNKNOWN,
        )

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> ChatTemplateConstants:
        raw_template = getattr(tokenizer, "chat_template", "") or ""
        tool_start, tool_end, tool_format = cls.detect_tool_delimiters(raw_template)

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
            tool_format=tool_format,
            generation_prompt=gen_prompt,
        )

    @classmethod
    def from_model_dir(cls, model_path: Path | str) -> ChatTemplateConstants:
        config_file = Path(model_path) / "tokenizer_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Missing {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_template = data.get("chat_template", "")
        tool_start, tool_end, tool_format = cls.detect_tool_delimiters(raw_template)

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
                tool_format=tool_format,
                generation_prompt=f"{Delimiters.LLAMA_HEADER_START}assistant{Delimiters.LLAMA_HEADER_END}\n\n",
            )

        if Delimiters.GEMMA_START_OF_TURN in raw_template:
            return ChatTemplateConstants(
                system_start=f"{Delimiters.GEMMA_START_OF_TURN}system\n",
                system_end=f"{Delimiters.GEMMA_END_OF_TURN}\n",
                user_start=f"{Delimiters.GEMMA_START_OF_TURN}user\n",
                user_end=f"{Delimiters.GEMMA_END_OF_TURN}\n",
                assistant_start=f"{Delimiters.GEMMA_START_OF_TURN}model\n",
                assistant_end=f"{Delimiters.GEMMA_END_OF_TURN}\n",
                think_start=Delimiters.THOUGHT_START,
                think_end=Delimiters.THOUGHT_END,
                tool_call_start=tool_start,
                tool_call_end=tool_end,
                tool_format=tool_format,
                generation_prompt=f"{Delimiters.GEMMA_START_OF_TURN}model\n",
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
                tool_format=tool_format,
                generation_prompt=Delimiters.DEEPSEEK_ASSISTANT,
            )

        return ChatTemplateConstants(
            tool_call_start=tool_start,
            tool_call_end=tool_end,
            tool_format=tool_format,
        )

    @staticmethod
    def _split_around_marker(text: str, marker: str) -> tuple[str, str]:
        if marker not in text:
            return ("", "")
        prefix, suffix = text.split(marker, 1)
        return (prefix, suffix)
