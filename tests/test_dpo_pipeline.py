"""End-to-end DPO pipeline test on CPU with a small real model (Qwen3.5-0.8B,
same architecture family as the 4B target). Exercises the exact chain that
has failed in queue three times: pairs -> DPOConfig -> DPOTrainer(PEFT +
precomputed ref) -> train -> save adapter -> load adapter -> toggle -> generate.

    python tests/test_dpo_pipeline.py     (login node, ~3-5 min, no GPU)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

tmp = Path(tempfile.mkdtemp(prefix="dpo_test_"))

# Tiny random-weight model: the risks under test (trl flag combos, PEFT,
# precomputed ref log-probs, adapter save/load) are architecture-agnostic,
# and a 6M-param model makes the test run in seconds on CPU. The real
# tokenizer is reused so the text path is authentic.
print("[0/3] building tiny test model ...")
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
_cfg = Qwen2Config(vocab_size=len(_tok), hidden_size=64, intermediate_size=128,
                   num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, max_position_embeddings=4096)
_m = Qwen2ForCausalLM(_cfg)
MODEL = str(tmp / "tiny")
_m.save_pretrained(MODEL)
_tok.save_pretrained(MODEL)
pairs = tmp / "pairs.jsonl"
out = tmp / "adapter"

# 10 short synthetic pairs (content irrelevant; API/save-path is under test)
with pairs.open("w") as f:
    for i in range(10):
        f.write(json.dumps({
            "prompt": f"<turn index={i} role=assistant>\n<content>\nwork on "
                      f"file src/mod{i}.py\n</content>\n</turn>\n\n[Summarize "
                      "the agent trace above.]",
            "chosen": f"Edited src/mod{i}.py; tests pass; next: refactor.",
            "rejected": "Stuff happened.",
            "d_chosen": 0.3, "d_rejected": 0.8,
        }) + "\n")

print(f"[1/3] training via the REAL script on {MODEL} (CPU) ...")
r = subprocess.run(
    [sys.executable, str(REPO / "experiments" / "dpo_train.py"),
     "--model", MODEL, "--pairs", str(pairs), "--out", str(out),
     "--epochs", "1", "--min-gap", "0.2"],
    capture_output=True, text=True, timeout=1800)
if r.returncode != 0:
    print("TRAIN FAILED:\n", r.stdout[-1500:], "\n", r.stderr[-2500:])
    sys.exit(1)
print("   train ok:", [l for l in r.stdout.splitlines() if "adapter saved" in l])

print("[2/3] verifying adapter artifacts ...")
files = sorted(p.name for p in out.glob("*"))
assert "adapter_config.json" in files, f"no adapter_config.json! got {files}"
assert any("adapter_model" in f for f in files), f"no adapter weights! got {files}"
print("   artifacts ok:", files[:4])

print("[3/3] loading + toggling adapter, generating (the never-executed eval path) ...")
from peft import PeftModel
from transformers import AutoModelForCausalLM

tok = _tok
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                            trust_remote_code=True).eval()
summ = PeftModel.from_pretrained(base, str(out)).eval()
ids = tok("Summarize: fixed bug in", return_tensors="pt")
for state, fn in (("dpo", summ.enable_adapter_layers),
                  ("base", summ.disable_adapter_layers)):
    fn()
    with torch.no_grad():
        gen = summ.generate(**ids, max_new_tokens=5,
                            pad_token_id=tok.eos_token_id)
    print(f"   {state}: generated {gen.shape[1] - ids['input_ids'].shape[1]} tokens ok")

print("\nDPO PIPELINE TEST: ALL CLEAR")
