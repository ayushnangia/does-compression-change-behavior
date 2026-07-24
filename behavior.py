"""Ask the model what it does next, and understand the answer.

Two things live here:
1. sampling the model's next action a few times (so we get a distribution,
   not one lucky/unlucky draw), and
2. parsing each continuation into a clean label: a tool call, a "lookup"
   (re-read/search), a "commit" (edit/run/answer), or "no action" (prose).
"""

from __future__ import annotations

import ast
import json
import re

# a tool call — both the trace format (<tool_calls>, plural) and the native
# Qwen3/3.5 chat format (<tool_call>, singular) that newer models emit
TOOLCALL_RE = re.compile(r"<tool_calls?>\s*(.*?)\s*(?:</tool_calls?>|$)", re.DOTALL)
# reasoning blocks of thinking models: text inside <think> is deliberation,
# not an executed action, so it is stripped before parsing. An unclosed
# <think> that consumes the whole continuation honestly parses as no action.
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
# tools offered by the --scaffold menu, written like read_file(...)
MENU_RE = re.compile(r"\b(read_file|grep|run_tests|edit|submit)\b\s*\(", re.I)
# hedged mention of a menu tool on the same line = prose, not an action
HEDGE_RE = re.compile(r"\b(could|would|might|may|maybe|perhaps|instead of|"
                      r"such as|for example|e\.g\.|like|or)\b[^\n]{0,50}$", re.I)

# the scaffold menu tools, classified explicitly (exact names)
MENU_KIND = {"read_file": "lookup", "grep": "lookup",
             "run_tests": "commit", "edit": "commit", "submit": "commit"}
# fallback for free-form trace tool names: gathering info vs committing a change
LOOKUP_WORDS = re.compile(r"\b(cat|ls|grep|find|head|tail|less|view|read|open|search|glob)\b", re.I)
COMMIT_WORDS = re.compile(r"\b(str_replace|create|write|insert|edit|patch|pytest|test|run|submit|finish|answer)\b", re.I)


def parse_action(text: str) -> "str | None":
    """Return a normalized action like 'edit::path=foo.py', or the menu form
    'read_file::foo.py', or None if the continuation contains no tool call."""
    text = THINK_RE.sub("", text)
    m = TOOLCALL_RE.search(text)
    if not m:
        # menu-call syntax (read_file(...)): block formats take precedence,
        # and a hedged mention ("I could grep(x)", "e.g. edit(...)") is
        # prose, not an action. "I will read_file(x)" commits and counts.
        mm = MENU_RE.search(text)
        if mm:
            line_prefix = text[:mm.start()].rsplit("\n", 1)[-1]
            if not HEDGE_RE.search(line_prefix):
                name = mm.group(1).lower()
                argm = re.search(re.escape(mm.group(1)) + r"\s*\(([^)]*)\)", text)
                return f"{name}::{(argm.group(1)[:60] if argm else '')}"
        return None
    body = m.group(1)
    # trace rows store tool calls as PYTHON-repr dicts (single quotes), the
    # models emit either that or real JSON — try both, then a quote-agnostic
    # regex. ast.literal_eval only parses literals, so it is safe on
    # model-generated text.
    obj = None
    try:
        obj = json.loads(body)
    except Exception:
        try:
            obj = ast.literal_eval(body)
        except Exception:
            obj = None
    name, args = None, ""
    if obj is not None:
        call = obj[0] if isinstance(obj, list) and obj else obj
        if isinstance(call, dict):
            fn = call.get("function", call)
            name = ((fn.get("name") if isinstance(fn, dict) else None)
                    or call.get("name")
                    or call.get("function_name"))   # Terminus trajectory format
            raw = (fn.get("arguments", "") if isinstance(fn, dict) else "") \
                or call.get("arguments", "")
            if isinstance(raw, str):
                args = raw
            else:
                # model-generated args can be ANY literal (sets crash plain
                # json.dumps — this killed two jobs). default=str makes
                # serialization total; sort for label stability.
                try:
                    args = json.dumps(raw, default=str, sort_keys=True)
                except Exception:
                    args = str(raw)
    if name is None:
        # Qwen3.5 NATIVE format (chat-template spec; vLLM parses this with
        # its qwen3_engine parser): <function=name><parameter=k>v</parameter>...
        fx = re.search(r"<function=([\w.-]+)>", body)
        if fx:
            params = re.findall(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>",
                                body, re.DOTALL)
            args = json.dumps({k: v for k, v in params}, sort_keys=True,
                              default=str) if params else ""
            name = fx.group(1)
            return f"{name}::{args[:60]}" if args else name
        # GLM-4.7 NATIVE format (chat-template spec; vLLM's glm47_moe parser):
        # <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        gx = re.match(r"\s*([\w.-]+)\s*<arg_key>", body)
        if gx:
            pairs = re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
                               body, re.DOTALL)
            args = json.dumps({k: v for k, v in pairs}, sort_keys=True,
                              default=str) if pairs else ""
            return f"{gx.group(1)}::{args[:60]}" if args else gx.group(1)
        nm = re.search(r'["\'](?:function_)?name["\']\s*:\s*["\']([^"\']+)["\']', body)
        if not nm:
            return None
        name = nm.group(1)
        am = re.search(r'["\']arguments["\']\s*:\s*["\']?(.{0,80})', body)
        args = am.group(1) if am else ""
    # include (truncated) arguments: with agentic traces most calls share one
    # tool name (execute_bash), so name-only labels would make every action
    # look identical and flatten action_change to zero.
    return f"{name}::{args[:60]}" if args else name


def parse_diagnosis(text: str) -> str:
    """When parse_action returns None, say WHY. Distinguishes behavioral
    silence from parser blindness (issue: NO_ACTION conflates four different
    failure modes, and parser misses silently inflate divergence).

    Categories:
      acted            parse_action found a call
      toolish_unparsed contains tool-call-like syntax we failed to parse
                       (> 0 in any cell = investigate the parser, not the model)
      think_runaway    an unclosed <think> consumed the continuation
      empty            no meaningful text
      prose            deliberate text with no tool call (true halt)
    """
    if parse_action(text) is not None:
        return "acted"
    stripped = THINK_RE.sub("", text)
    if re.search(r"<tool_calls?|function_name|\"name\"\s*:", text) and stripped.strip():
        return "toolish_unparsed"
    if "<think>" in text and not stripped.strip():
        return "think_runaway"
    if not text.strip():
        return "empty"
    return "prose"


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


# generation ends as soon as the first tool call closes: deployment-sized
# budget without paying for it when the model acts early. Covers all
# formats the parser accepts (trace, Qwen-native, GLM-native close with
# </tool_call(s)>; menu calls are short).
STOP_STRINGS = ["</tool_call>", "</tool_calls>"]


def sample_texts(model, tokenizer, context_ids, device, *,
                 samples=8, max_new=10240, temperature=0.7, top_p=1.0, seed=0):
    """Sample `samples` raw continuations (decoded text) from `context_ids`.
    top_p=1.0 (full distribution) measures true behavior; pass e.g. 0.9 to
    emulate production sampling. max_new=10240 is deployment-parity (Terminus
    allows 10240; thinking models may deliberate at length before acting) -
    it was 768 until 2026-07-24, which could truncate long deliberation into
    a false halt; see docs/DECISIONS.md."""
    import torch

    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            torch.tensor([context_ids], device=device),
            attention_mask=torch.ones(1, len(context_ids), device=device),
            max_new_tokens=max_new, do_sample=True, temperature=temperature,
            top_p=top_p, top_k=0, num_return_sequences=samples,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            stop_strings=STOP_STRINGS, tokenizer=tokenizer,
        )
    return [tokenizer.decode(out[i, len(context_ids):], skip_special_tokens=True)
            for i in range(out.shape[0])]


def sample_actions(model, tokenizer, context_ids, device, *,
                   samples=8, max_new=10240, temperature=0.7, seed=0):
    """Sample `samples` next-actions. Returns the list of parsed action labels
    (None where the model produced no tool call)."""
    texts = sample_texts(model, tokenizer, context_ids, device, samples=samples,
                         max_new=max_new, temperature=temperature, seed=seed)
    return [parse_action(t) for t in texts]
