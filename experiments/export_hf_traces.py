#!/usr/bin/env python3
"""Export valid Qwen3.8 Harbor trajectories as a citable HF dataset folder.

Includes clean and agent-timeout trajectories (both are genuine on-policy
model behavior), but excludes trials that never started an agent environment
and every retired/port-collision run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

DEFAULT_RUNS = [
    ("baseline", 809200, "tb2-qwen38-27b-valid-easy25"),
    ("r2", 834653, "tb2-qwen38-27b-valid-r2-easy25"),
    ("r3", 834654, "tb2-qwen38-27b-valid-r3-easy25"),
    ("r4", 834686, "tb2-qwen38-27b-valid-r4-easy25"),
    ("r5", 834739, "tb2-qwen38-27b-valid-r5-easy25"),
    ("r6", 834740, "tb2-qwen38-27b-valid-r6-easy25"),
    ("r7", 834741, "tb2-qwen38-27b-valid-r7-easy25"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exception_kind(text: str) -> str | None:
    for name in ("AgentTimeoutError", "EnvironmentStartTimeoutError",
                 "OutputLengthExceededError"):
        if name in text:
            return name
    return "OtherException" if text.strip() else None


def dataset_card(repo_id: str, rows: list[dict], runs: list[tuple]) -> str:
    clean = sum(r["termination"] == "clean" for r in rows)
    timed = sum(r["termination"] == "agent_timeout" for r in rows)
    tasks = len({r["task"] for r in rows})
    return f"""---
license: mit
task_categories:
- text-generation
language:
- en
pretty_name: Qwen3.8 Terminal-Bench On-Policy Compaction Traces
tags:
- coding-agents
- context-compaction
- terminal-bench
- on-policy
size_categories:
- n<1K
---

# Qwen3.8 Terminal-Bench On-Policy Compaction Traces

Citable trajectory release for **Does Compression Change Behavior?**

- **Acting model:** `Qwen/Qwen3.8-27B` bf16
- **Scaffold:** Harbor `terminus-2`
- **Context budget:** 65,536 tokens; serving window 77,824
- **Generation:** native `reasoning_effort=low`, temperature 1, top-p 1
- **Tasks:** Terminal-Bench 2 easy25
- **Trajectories:** {len(rows)} across {tasks} tasks ({clean} clean, {timed} agent-timeout)
- **Included runs:** {', '.join(str(job) for _, job, _ in runs)}

`AgentTimeoutError` trajectories are included because they contain genuine
on-policy model actions before Harbor's episode wall-clock limit. Trials with
`EnvironmentStartTimeoutError` have no model trajectory and are excluded.
Retired raw-completion proxies, xhigh-timeout collections, and port-collision
job 834655 are excluded.

## Files

- `data/traces.jsonl`: metadata plus byte-exact trajectory JSON stored as a string.
- `data/index.jsonl`: metadata without trajectory bodies.
- `raw/<run>/<trial>/trajectory.json`: original Harbor files.
- `checksums.sha256`: SHA-256 for every released file.

## Intended use

Behavioral analysis of context compaction and coding-agent action selection.
This release does **not** establish a Terminal-Bench Pass@1 score; infrastructure
failures and agent timeouts must not be silently scored as task failures.
The Apptainer tasks had the shared `/scratch` filesystem mounted. Some agents
inspected paths or outputs from prior experiments, so these traces are suitable
for compaction/behavior analysis but not an uncontaminated benchmark score.
Raw traces also retain cluster paths and the collection username for provenance;
a secret scan found no GitHub/Hugging Face/AWS tokens or private keys.

## Citation

```bibtex
@misc{{nangia2026compressiontraces,
  author       = {{Nangia, Ayush}},
  title        = {{Qwen3.8 Terminal-Bench On-Policy Compaction Traces}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

Repository: https://github.com/ayushnangia/does-compression-change-behavior
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="ayushnangia/does-compression-change-behavior-traces")
    ap.add_argument("--push", action="store_true",
                    help="create/update a public Hugging Face dataset repo")
    args = ap.parse_args()
    jobs = Path(args.jobs_dir)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "raw").mkdir()

    rows: list[dict] = []
    seen = set()
    for run, job_id, dirname in DEFAULT_RUNS:
        root = jobs / dirname
        if not root.is_dir():
            raise SystemExit(f"missing declared run: {root}")
        for trial in sorted(p for p in root.iterdir() if p.is_dir()):
            trajectory = trial / "agent" / "trajectory.json"
            if not trajectory.is_file():
                continue
            raw = trajectory.read_bytes()
            # Parse once to prove each released body is valid JSON.
            json.loads(raw)
            task, _, trial_id = trial.name.partition("__")
            trace_id = f"{run}__{task}__{trial_id}"
            if trace_id in seen:
                raise SystemExit(f"duplicate trace id: {trace_id}")
            seen.add(trace_id)
            exc_text = ((trial / "exception.txt").read_text(errors="replace")
                        if (trial / "exception.txt").exists() else "")
            exc = exception_kind(exc_text)
            if exc == "EnvironmentStartTimeoutError":
                raise SystemExit(f"environment failure unexpectedly has trace: {trial}")
            reward_path = trial / "verifier" / "reward.txt"
            reward = None
            if reward_path.is_file():
                try:
                    reward = float(reward_path.read_text().strip())
                except ValueError:
                    reward = None
            termination = "agent_timeout" if exc == "AgentTimeoutError" else (
                "clean" if exc is None else "other_exception")
            meta = {
                "trace_id": trace_id,
                "run": run,
                "source_job_id": job_id,
                "task": task,
                "trial_id": trial_id,
                "source_model": "Qwen/Qwen3.8-27B",
                "scaffold": "terminus-2",
                "context_budget_tokens": 65536,
                "serving_window_tokens": 77824,
                "reasoning_effort": "low",
                "termination": termination,
                "exception_type": exc,
                "verifier_reward": reward,
                "trajectory_bytes": len(raw),
                "trajectory_sha256": sha256_bytes(raw),
            }
            raw_dest = out / "raw" / run / trial.name / "trajectory.json"
            raw_dest.parent.mkdir(parents=True, exist_ok=True)
            raw_dest.write_bytes(raw)
            rows.append({**meta, "trajectory_json": raw.decode("utf-8")})

    with (out / "data" / "traces.jsonl").open("w") as full, \
         (out / "data" / "index.jsonl").open("w") as index:
        for row in rows:
            full.write(json.dumps(row, ensure_ascii=False) + "\n")
            index.write(json.dumps({k: v for k, v in row.items()
                                    if k != "trajectory_json"}) + "\n")
    (out / "README.md").write_text(dataset_card(args.repo_id, rows, DEFAULT_RUNS))
    (out / "LICENSE").write_text((Path(__file__).resolve().parent.parent / "LICENSE").read_text())
    release = {
        "repo_id": args.repo_id,
        "source_model": "Qwen/Qwen3.8-27B",
        "runs": [{"name": n, "job_id": j, "jobs_dir": d}
                 for n, j, d in DEFAULT_RUNS],
        "trajectories": len(rows),
        "tasks": sorted({r["task"] for r in rows}),
        "termination_counts": {
            k: sum(r["termination"] == k for r in rows)
            for k in ("clean", "agent_timeout", "other_exception")
        },
        "version": "v1-r2-r7",
        "limitations": [
            "shared /scratch was mounted; some agents inspected prior-run paths",
            "not valid for aggregate Terminal-Bench Pass@1",
        ],
        "explicit_exclusions": [
            "job 834655 (co-scheduled localhost port collision)",
            "jobs 803407/803548 (invalid raw-completion proxies)",
            "job 805029 (xhigh reasoning request-timeout contamination)",
        ],
    }
    (out / "release.json").write_text(json.dumps(release, indent=2) + "\n")

    checksum_lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and
                       p.name != "checksums.sha256"):
        checksum_lines.append(f"{sha256_bytes(path.read_bytes())}  {path.relative_to(out)}")
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(release, indent=2))
    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF push refused: set HF_TOKEN in gitignored .env")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="dataset", private=False,
                        exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset",
                          folder_path=out,
                          commit_message="Release 132 valid Qwen3.8 on-policy traces (v1-r2-r7)")
        print(f"PUSHED https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
