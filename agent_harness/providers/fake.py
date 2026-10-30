"""A scripted provider: deterministic runs, no network, no spend.

This is what the suite and the demo use. It is labelled "fake" everywhere it can
be seen, so nobody confuses a scripted run with a model run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

from ..errors import ProviderError
from .base import Completion, Message, estimate_tokens


@dataclass
class ScriptedReply:
    """Match on any substring of the last user message, or match everything."""
    contains: str = ""
    text: str = "ok"
    model: Optional[str] = None
    finish_reason: str = "stop"
    once: bool = False

    def matches(self, prompt: str) -> bool:
        return not self.contains or self.contains.lower() in prompt.lower()


Matcher = Callable[[Sequence[Message], str], Optional[str]]


class FakeProvider:
    name = "fake"
    attempt_ceiling = 1

    def __init__(self, script: Sequence[Union[ScriptedReply, Dict[str, str]]] = (), fallback: str = "ok"):
        self.script: List[ScriptedReply] = []
        for item in script:
            if isinstance(item, ScriptedReply):
                self.script.append(item)
            else:
                self.script.append(ScriptedReply(contains=item.get("contains", ""),
                                                 text=item.get("text", "ok"),
                                                 model=item.get("model"),
                                                 finish_reason=item.get("finish_reason", "stop"),
                                                 once=bool(item.get("once"))))
        self.fallback = fallback
        self.calls: List[Dict[str, object]] = []

    def complete(self, messages: Sequence[Message], model: str, temperature: float = 0.0,
                 max_tokens: int = 1024) -> Completion:
        prompt = next((m.content for m in reversed(list(messages)) if m.role == "user"), "")
        if not prompt:
            raise ProviderError("fake provider got no user message")
        self.calls.append({"model": model, "prompt": prompt[:200]})
        finish = "stop"
        for reply in self.script:
            if reply.matches(prompt):
                text, finish = reply.text, reply.finish_reason
                if reply.once:
                    self.script.remove(reply)
                break
        else:
            text = self.fallback
        input_tokens = sum(estimate_tokens(m.content) for m in messages)
        return Completion(text=text, model=model, input_tokens=input_tokens,
                          output_tokens=estimate_tokens(text), finish_reason=finish,
                          raw={"provider": "fake"})

    @classmethod
    def from_json_file(cls, path: str) -> "FakeProvider":
        try:
            payload = json.loads(open(path).read())
        except (OSError, ValueError) as exc:
            raise ProviderError("cannot read the script {}: {}".format(path, exc)) from None
        return cls(payload.get("script", payload), fallback=payload.get("fallback", "ok"))
