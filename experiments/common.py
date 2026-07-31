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


def _provenance() -> dict:
    """Reproducibility stamp attached to every result file: exactly which
    code, environment, and inputs produced this number (ICLR-grade: a result
    without provenance is an anecdote)."""
    import hashlib
    import os
    import subprocess

    prov = {"argv": sys.argv, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        prov["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
    except Exception:
        prov["git_commit"] = None
    for mod in ("torch", "transformers"):
        try:
            prov[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            pass
    # hash any --examples-file style input so the exact dataset is pinned
    for i, a in enumerate(sys.argv):
        if a.endswith(".json") and Path(a).exists():
            prov[f"input_sha256:{Path(a).name}"] = hashlib.sha256(
                Path(a).read_bytes()).hexdigest()[:16]
    return prov


def save_result(name: str, payload: dict, out_dir: str = None):
    out = Path(out_dir or REPO / "experiments" / "results")
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["_provenance"] = _provenance()
    path = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {path}")


def kind_profile(actions, action_kind) -> dict:
    """Fractions of none / lookup / commit among sampled actions."""
    kinds = [action_kind(a) for a in actions]
    return {k: kinds.count(k) / len(kinds) for k in ("none", "lookup", "commit")}
