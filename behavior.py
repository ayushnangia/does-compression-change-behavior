"""Ask the model what it does next, and understand the answer.

Two things live here:
1. sampling the model's next action a few times (so we get a distribution,
   not one lucky/unlucky draw), and
2. parsing each continuation into a clean label: a tool call, a "lookup"
   (re-read/search), a "commit" (edit/run/answer), or "no action" (prose).
"""

from __future__ import annotations

import json
import re

# a tool call in the trace format
TOOLCALL_RE = re.compile(r"<tool_calls>\s*(.*?)\s*(?:</tool_calls>|$)", re.DOTALL)
# tools offered by the --scaffold menu, written like read_file(...)
MENU_RE = re.compile(r"\b(read_file|grep|run_tests|edit|submit)\b\s*\(", re.I)

# the scaffold menu tools, classified explicitly (exact names)
MENU_KIND = {"read_file": "lookup", "grep": "lookup",
             "run_tests": "commit", "edit": "commit", "submit": "commit"}
# fallback for free-form trace tool names: gathering info vs committing a change
LOOKUP_WORDS = re.compile(r"\b(cat|ls|grep|find|head|tail|less|view|read|open|search|glob)\b", re.I)
COMMIT_WORDS = re.compile(r"\b(str_replace|create|write|insert|edit|patch|pytest|test|run|submit|finish|answer)\b", re.I)


def parse_action(text: str) -> "str | None":
    """Return a normalized action like 'edit::path=foo.py', or the menu form
    'read_file::foo.py', or None if the continuation contains no tool call."""
    m = MENU_RE.search(text)
    if m:
        name = m.group(1).lower()
        argm = re.search(re.escape(m.group(1)) + r"\s*\(([^)]*)\)", text)
        return f"{name}::{(argm.group(1)[:60] if argm else '')}"
    m = TOOLCALL_RE.search(text)
    if not m:
        return None
    body = m.group(1)
    try:
        obj = json.loads(body)
        call = obj[0] if isinstance(obj, list) and obj else obj
        fn = call.get("function", call) if isinstance(call, dict) else {}
        name = fn.get("name") or call.get("name")
    except Exception:
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', body)
        name = nm.group(1) if nm else None
    return name if name else None


def action_kind(action: "str | None") -> str:
    """'lookup', 'commit', or 'none'."""
    if not action:
        return "none"
    name = action.split("::", 1)[0].lower()
    if name in MENU_KIND:            # scaffold menu tools: exact match
        return MENU_KIND[name]
    blob = action.lower()
    if COMMIT_WORDS.search(blob):
        return "commit"
    if LOOKUP_WORDS.search(blob):
        return "lookup"
    return "commit"  # a tool call that isn't clearly a lookup counts as commit


def sample_actions(model, tokenizer, context_ids, device, *,
                   samples=8, max_new=384, temperature=0.7, noise_std=0.0, seed=0):
    """Sample `samples` next-actions. Returns the list of parsed action labels
    (None where the model produced no tool call).

    `noise_std` > 0 adds Gaussian noise to the logits at each step — the
    control that asks "is any perturbation this size damaging, or summaries
    specifically?"
    """
    import torch

    processors = None
    if noise_std > 0:
        from transformers import LogitsProcessorList

        gen = torch.Generator(device="cpu").manual_seed(seed)

        def add_noise(_ids, scores):
            return scores + torch.randn(scores.shape, generator=gen).to(scores.device, scores.dtype) * noise_std

        processors = LogitsProcessorList([add_noise])

    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            torch.tensor([context_ids], device=device),
            attention_mask=torch.ones(1, len(context_ids), device=device),
            max_new_tokens=max_new, do_sample=True, temperature=temperature,
            top_p=1.0, top_k=0, num_return_sequences=samples,
            logits_processor=processors,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    actions = []
    for i in range(out.shape[0]):
        text = tokenizer.decode(out[i, len(context_ids):], skip_special_tokens=True)
        actions.append(parse_action(text))
    return actions
