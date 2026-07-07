"""The ways to shrink an agent's history.

Each compressor takes the OLD part of the context (everything except the most
recent slice) and returns replacement token ids. The caller always keeps the
recent slice raw and appends it after — that mirrors how real systems compact
old history but keep the latest turns verbatim.

Signature for every compressor:
    compress(old_ids, tokenizer, model, device) -> list[int]
`model` is only used by compressors that need to generate text (summary,
paraphrase); the others ignore it.
"""

from __future__ import annotations

import re

# tokens that look like file paths, used by the `pointer` compressor
PATH_RE = re.compile(r"[\w./-]*\.\w+|[\w./-]+/[\w./-]+")

SUMMARIZE = (
    "\n\n[Summarize the agent trace above so the work can be continued. "
    "Keep file paths, commands, errors, and decisions. Be concise.]\nSummary:"
)
PARAPHRASE = (
    "\n\n[Reword the agent trace above using different words but keep the "
    "exact same XML tag structure and every file path, command, number, and "
    "decision.]\nReworded:"
)


def _generate(model, tokenizer, prompt_ids, max_new, device) -> "list[int]":
    import torch

    with torch.no_grad():
        out = model.generate(
            torch.tensor([prompt_ids], device=device),
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return out[0, len(prompt_ids):].tolist()


def _ids(tokenizer, text) -> "list[int]":
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def keep_recent(old_ids, tokenizer, model, device, *, keep=0.5):
    """Keep only the last `keep` fraction of the old history."""
    n = max(1, int(len(old_ids) * keep))
    return old_ids[-n:]


def summary(old_ids, tokenizer, model, device, *, budget=256):
    """Replace the old history with a model-written summary, wrapped as a turn
    (the way a real system would insert a compaction note)."""
    prompt = old_ids + _ids(tokenizer, SUMMARIZE)
    gen = _generate(model, tokenizer, prompt, budget, device)
    text = tokenizer.decode(gen, skip_special_tokens=True)
    wrapped = f"<turn index=0 role=user>\n<content>\n[Summary of work so far: {text}]\n</content>\n</turn>\n"
    return _ids(tokenizer, wrapped)


def paraphrase(old_ids, tokenizer, model, device, *, budget=None):
    """Reword the old history, keeping all information. Control: separates
    'lost information' from 'the model is just sensitive to wording'."""
    prompt = old_ids + _ids(tokenizer, PARAPHRASE)
    gen = _generate(model, tokenizer, prompt, budget or len(old_ids), device)
    return gen or old_ids


def pointer(old_ids, tokenizer, model, device):
    """Drop the old history but leave pointers to where the info lives.
    Control: damage the agent CAN recover from (it can re-read the files)."""
    text = tokenizer.decode(old_ids, skip_special_tokens=True)
    files = list(dict.fromkeys(PATH_RE.findall(text)))[:8]
    note = (f"[Earlier work compacted. Full details remain in: {', '.join(files)}. "
            "Re-read these as needed.]\n") if files else \
           "[Earlier work compacted; re-read project files as needed.]\n"
    return _ids(tokenizer, note)


def hallucinator(old_ids, tokenizer, model, device):
    """Drop the old history and assert completion with no pointers.
    Control: damage the agent CANNOT recover from (the dangerous case)."""
    note = ("[Earlier work compacted. All necessary analysis is complete and "
            "the approach is settled; proceed confidently.]\n")
    return _ids(tokenizer, note)


# `full` and `noise` are handled specially in run_experiment.py:
#   full  = keep the old history unchanged
#   noise = keep full context but perturb the model's logits at generation
#           time to match summary's likelihood-damage (a control done in the
#           behavior step, not here)

TEXT_COMPRESSORS = {
    "keep_recent": keep_recent,
    "summary": summary,
    "paraphrase": paraphrase,
    "pointer": pointer,
    "hallucinator": hallucinator,
}
