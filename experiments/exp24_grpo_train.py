"""EXP 24C (2 GPUs) — direct-reward GRPO for an extractive compactor.

GPU 0 trains a selector model. GPU 1 serves the frozen executor through vLLM.
The selector emits {"keep":[turn indices]}; selected trace turns are copied
byte-for-byte. There is no GAE and no learned critic: each history is one
contextual-bandit group, scored immediately by downstream action preservation.

Launch through exp24_job.sh, not directly, so the executor is ready first.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from exp24_credit import selector_reward
from exp24_data import completion_text, parse_keep, render_selection


def _tool(action):
    return action.split("::", 1)[0] if action else None


class ExecutorReward:
    """Callable TRL reward backed by a frozen OpenAI-compatible vLLM server."""

    def __init__(self, url, model, samples, max_tokens, log_path):
        # TRL 0.29 records reward function names and assumes function-like
        # callables expose __name__. Keep the stateful class but satisfy that
        # installed interface explicitly (selector gate 813716).
        self.__name__ = "executor_behavior_reward"
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.samples = samples
        self.max_tokens = max_tokens
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _one(self, completion, header, units_json, recent_text,
             logged_action, budget_chars, task, row_id):
        import requests
        completion = completion_text(completion)
        from behavior import parse_action
        from metrics import _verb

        units = json.loads(units_json)
        keep, valid = parse_keep(completion, len(units))
        selected_chars = sum(len(units[i]) for i in keep)
        ratio = selected_chars / max(1, int(budget_chars))
        if not valid or ratio > 1:
            reward = selector_reward(tool_agreement=0, verb_agreement=0,
                acting_rate=0, valid=valid, budget_ratio=ratio)
            record = {"id": row_id, "task": task, "completion": completion,
                      "keep": keep, "valid": valid, "budget_ratio": ratio,
                      "reward": reward, "actions": []}
        else:
            context = render_selection(header, units, keep, recent_text)
            # Match the live post-compaction interface: a fresh native chat
            # receives one handoff message containing exact selected history.
            # Raw /v1/completions over an old trace prefix was invalidated by
            # Qwen3.8 preflight 803548.
            handoff = (
                "You are resuming a command-line task after context compaction. "
                "Selected prior trace blocks are copied below verbatim, followed "
                "by the recent history. Continue from the current terminal state "
                "and respond in the required Terminus JSON command format.\n\n" +
                context)
            payload = {"model": self.model,
                       "messages": [{"role": "user", "content": handoff}],
                       "n": self.samples, "max_tokens": self.max_tokens,
                       "temperature": 1.0, "top_p": 1.0,
                       "reasoning_effort": "low"}
            response = requests.post(self.url, json=payload, timeout=1800)
            response.raise_for_status()
            texts = [c["message"].get("content") or ""
                     for c in response.json()["choices"]]
            actions = [parse_action(t) for t in texts]
            acting = sum(a is not None for a in actions) / len(actions)
            tool = sum(_tool(a) == _tool(logged_action) for a in actions) / len(actions)
            verb = sum(_verb(a) == _verb(logged_action) for a in actions) / len(actions)
            reward = selector_reward(tool_agreement=tool, verb_agreement=verb,
                acting_rate=acting, valid=True, budget_ratio=ratio)
            record = {"id": row_id, "task": task, "completion": completion,
                      "keep": keep, "valid": True, "budget_ratio": ratio,
                      "reward": reward, "tool_agreement": tool,
                      "verb_agreement": verb, "acting": acting,
                      "actions": actions}
        with self.lock, self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return reward

    def __call__(self, prompts, completions, **kw):
        n = len(completions)
        fields = [kw[k] for k in ("header", "units_json", "recent_text",
                  "logged_action", "budget_chars", "task", "id")]
        args = [(completions[i],) + tuple(field[i] for field in fields)
                for i in range(n)]
        with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
            return list(pool.map(lambda x: self._one(*x), args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--eval-file")
    ap.add_argument("--executor-url", default="http://127.0.0.1:8001")
    ap.add_argument("--executor-name", default="qwen38-27b-exp24")
    ap.add_argument("--source-model", default="Qwen/Qwen3.8-27B",
                    help="hard lineage guard for on-policy training rows")
    ap.add_argument("--out", default="results/exp24_grpo")
    ap.add_argument("--reward-log", default="results/exp24_grpo_rewards.jsonl")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--executor-samples", type=int, default=4)
    ap.add_argument("--executor-max-tokens", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="pilot plumbing: set e.g. 10; main run leaves -1")
    ap.add_argument("--min-train-examples", type=int, default=1000,
                    help="hard power gate; lower only for labeled plumbing pilots")
    ap.add_argument("--preflight-only", action="store_true",
                    help="construct model/LoRA/TRL trainer but do not train")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    files = {"train": args.train_file}
    if args.eval_file: files["validation"] = args.eval_file
    ds = load_dataset("json", data_files=files)
    for split in ds:
        lineages = set(ds[split]["source_model"])
        if lineages != {args.source_model}:
            raise SystemExit(f"OFF-POLICY DATA REFUSED: {split} has {lineages}, "
                             f"required {args.source_model}")
    if len(ds["train"]) < args.min_train_examples:
        raise SystemExit(f"UNDERPOWERED DATA REFUSED: {len(ds['train'])} train "
                         f"rows < {args.min_train_examples}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # GRPO batches variable-length decoder-only prompts. Right padding changes
    # the generation boundary and triggered a Transformers correctness warning
    # on every step of otherwise-green pilot 828066.
    tokenizer.padding_side = "left"
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"])
    reward = ExecutorReward(args.executor_url, args.executor_name,
        args.executor_samples, args.executor_max_tokens, args.reward_log)
    cfg = GRPOConfig(
        output_dir=args.out, learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.generations,
        gradient_accumulation_steps=2, num_generations=args.generations,
        max_completion_length=64, temperature=1.0, top_p=1.0,
        beta=0.01, scale_rewards="group", loss_type="dr_grpo",
        importance_sampling_level="sequence", gradient_checkpointing=True,
        # Qwen3.5 otherwise starts every completion with unconstrained
        # <think> prose and exhausts the 64-token structured-action budget.
        # This is the model's native chat-template switch, not prompt hacking.
        chat_template_kwargs={"enable_thinking": False},
        bf16=torch.cuda.is_available(), logging_steps=1, save_steps=50,
        save_total_limit=2, report_to=[], eval_strategy="no")
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward, args=cfg,
        train_dataset=ds["train"], processing_class=tokenizer, peft_config=peft)
    if args.preflight_only:
        # Competence gate, not just an import gate: native-chat Qwen3.5 must
        # emit at least one parseable selector action before GRPO can have a
        # non-constant group reward. Pilot 813724 established the all-invalid
        # failure mode for raw string prompts.
        sample = ds["train"][0]
        prompt = sample["prompt"]
        if not isinstance(prompt, list):
            raise SystemExit("SELECTOR PREFLIGHT REFUSED: prompt is not native chat")
        encoded = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
            return_dict=True).to(trainer.model.device)
        with torch.inference_mode():
            generated = trainer.model.generate(
                **encoded, max_new_tokens=64, do_sample=True, temperature=1.0,
                top_p=1.0, num_return_sequences=4,
                pad_token_id=tokenizer.pad_token_id)
        prompt_len = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True)
        n_units = len(json.loads(sample["units_json"]))
        parsed = [parse_keep(text, n_units)[1] for text in texts]
        for text, valid in zip(texts, parsed):
            print(f"selector_candidate valid={valid}: {text!r}")
        if not any(parsed):
            raise SystemExit("SELECTOR PREFLIGHT REFUSED: 0/4 valid JSON candidates")
        print(f"EXP24 SELECTOR PREFLIGHT GREEN: model + LoRA + GRPOTrainer; "
              f"valid_json={sum(parsed)}/4")
        return
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved exp24 GRPO adapter -> {args.out}")


if __name__ == "__main__":
    main()
