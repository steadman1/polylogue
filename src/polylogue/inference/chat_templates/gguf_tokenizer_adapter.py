from jinja2.sandbox import ImmutableSandboxedEnvironment

from polylogue.inference.chat_templates.chat_template import (
    ChatTemplateConstants,
    ChatTemplateDetector,
)


class GGUFTokenizerAdapter:
    """Wraps the raw GGUF Jinja string to provide an apply_chat_template interface."""

    def __init__(self, raw_chat_template: str):
        self.chat_template = raw_chat_template
        self.env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self.template = self.env.from_string(raw_chat_template)

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list[dict] | None = None,
    ) -> str:
        # Standard Jinja rendering expected by HuggingFace / GGUF templates
        return self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            bos_token="<s>",
            eos_token="</s>",
        )
