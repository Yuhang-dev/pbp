from __future__ import annotations

import re
from typing import Any

TURN_RE = re.compile(r"\n\n(Human|Assistant):")


def hh_prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    """Convert an Anthropic HH-style prompt prefix into chat-template messages."""
    matches = list(TURN_RE.finditer(prompt))
    if not matches:
        return [{"role": "user", "content": prompt.strip()}]

    messages: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        role_name = match.group(1)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        content = prompt[content_start:content_end].strip()
        if not content:
            continue
        role = "user" if role_name == "Human" else "assistant"
        messages.append({"role": role, "content": content})

    if not messages:
        return [{"role": "user", "content": prompt.strip()}]
    return messages


def format_prompt(
    prompt: str,
    tokenizer: Any | None = None,
    *,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
) -> str:
    """Format a prompt once, then use the exact string for base and instruct log-probs."""
    if tokenizer is not None and use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = hh_prompt_to_messages(prompt)
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    return prompt
