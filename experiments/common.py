"""Shared plumbing for the behavioral experiments (exp3-exp5)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {name} ...")
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    ).eval()
    return tokenizer, model, model.device


def save_result(name: str, payload: dict, out_dir: str = None):
    out = Path(out_dir or REPO / "experiments" / "results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {path}")


def kind_profile(actions, action_kind) -> dict:
    """Fractions of none / lookup / commit among sampled actions."""
    kinds = [action_kind(a) for a in actions]
    return {k: kinds.count(k) / len(kinds) for k in ("none", "lookup", "commit")}
