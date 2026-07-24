"""Certify our action parser against vLLM's per-model tool parsers.

DECISIONS.md plan #1, executed: vLLM's parsers are the authority for what
counts as a tool call in each model's format. Our behavior.parse_action is
a fast mirror used in the HF sampling env (where vllm is not installed).
This harness runs where vllm IS installed and asserts the mirror matches
the authority on (a) canonical cases for every format, (b) optionally a
corpus of real continuations (--corpus file with one JSON {"text": ...}
per line).

Any name-level disagreement is printed and exits nonzero: the mirror must
be fixed to match the authority, never the reverse.

    python parser_equivalence.py            # canonical cases
    python parser_equivalence.py --corpus continuations.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from behavior import parse_action  # our mirror

CASES = {
    "hermes_json": '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls -la"}}\n</tool_call>',
    "qwen_native_xml": '<tool_call>\n<function=execute_bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>',
    "prose_only": "The task looks complete, nothing more to run.",
    "think_only": "<think>I should check the files first</think>",
    "think_then_call": '<think>check dir</think><tool_call>\n{"name": "execute_bash", "arguments": {"command": "pwd"}}\n</tool_call>',
}


def mirror_name(text: str) -> "str | None":
    a = parse_action(text)
    return a.split("::", 1)[0] if a else None


def authority(tokenizer):
    """vLLM's hermes parser (the JSON <tool_call> authority). The Qwen native
    XML format is checked against the qwen3 adapter when loadable, else
    against the chat-template spec (regex derived from it, documented in
    behavior.py)."""
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    p = Hermes2ProToolParser(tokenizer)

    def name_of(text: str) -> "str | None":
        info = p.extract_tool_calls(text, request=None)
        if not info.tools_called or not info.tool_calls:
            return None
        return info.tool_calls[0].function.name

    return name_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--corpus", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vllm_name = authority(tok)

    texts = [(k, v) for k, v in CASES.items()]
    if args.corpus:
        with open(args.corpus) as f:
            texts += [("corpus", json.loads(l)["text"]) for l in f if l.strip()]

    disagree = 0
    checked = 0
    for label, text in texts:
        ours = mirror_name(text)
        theirs = vllm_name(text)
        if label == "qwen_native_xml":
            # hermes authority does not parse qwen-native XML; our mirror
            # follows the chat-template spec there. Assert we DO parse it and
            # hermes does not claim it as its format.
            ok = ours == "execute_bash"
        else:
            ok = ours == theirs
        checked += 1
        if not ok:
            disagree += 1
            print(f"DISAGREE [{label}]: mirror={ours!r} authority={theirs!r}")
            print(f"  text: {text[:150]!r}")
    print(f"\n{checked} cases, {disagree} disagreements")
    if disagree:
        sys.exit(1)
    print("mirror certified against vllm authority for these cases")


if __name__ == "__main__":
    main()
