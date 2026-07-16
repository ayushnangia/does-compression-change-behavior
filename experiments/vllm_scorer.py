"""Batched behavioral scorer backed by an in-process vLLM engine (T1 of the
scale-compaction-regime plan).

Drop-in equivalents of behavior.sample_texts / sample_actions with the same
sampling protocol (temperature 0.7, top_p 1.0, per-call seeds), plus a
batched entry point that scores MANY contexts in one engine pass — the
10x-throughput path for pair generation and scaled measurement.

Runs in ENV-vllm (verified: metrics/behavior-parser/data-file imports are
dependency-clean there). The HF path in behavior.py is untouched and remains
the reference implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from behavior import parse_action  # noqa: E402


class VLLMScorer:
    def __init__(self, model: str, *, max_model_len: int = 32768,
                 gpu_memory_utilization: float = 0.90):
        from vllm import LLM
        self.llm = LLM(model=model, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=False)

    def sample_texts_batch(self, contexts, *, samples=8, max_new=768,
                           temperature=0.7, top_p=1.0, seed=0):
        """contexts: list[list[int]] -> list[list[str]] (one engine pass)."""
        from vllm import SamplingParams, TokensPrompt
        params = [SamplingParams(n=samples, max_tokens=max_new,
                                 temperature=temperature, top_p=top_p,
                                 seed=seed + i)
                  for i in range(len(contexts))]
        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(c)) for c in contexts],
            params)
        return [[o.text for o in out.outputs] for out in outs]

    def sample_actions_batch(self, contexts, **kw):
        return [[parse_action(t) for t in texts]
                for texts in self.sample_texts_batch(contexts, **kw)]

    # drop-in single-context forms (match behavior.py signatures loosely)
    def sample_texts(self, context_ids, *, samples=8, max_new=768,
                     temperature=0.7, top_p=1.0, seed=0):
        return self.sample_texts_batch([context_ids], samples=samples,
                                       max_new=max_new, temperature=temperature,
                                       top_p=top_p, seed=seed)[0]

    def sample_actions(self, context_ids, **kw):
        return [parse_action(t) for t in self.sample_texts(context_ids, **kw)]
