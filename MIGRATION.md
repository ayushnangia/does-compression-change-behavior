# Migration runbook: Narval -> H100 cluster (Trillium / Nibi / Killarney / Fir)

Why: Narval A100-40GB cannot serve Qwen3.5-35B-A3B bf16 (vllm 0.25 MoE
multi-GPU engine init hangs at TP=2 and TP=4, jobs 66203598 et al.), and the
64k arms OOM. H100-80GB fits the 35B on a single GPU and doubles headroom.
Bonus: Nibi/Killarney/Fir have internet on compute nodes, which removes the
entire offline-baking tax (no --examples-file prefetch dance, no sif
pre-baking, HF downloads just work).

## What travels how

| asset | size | method |
|---|---|---|
| repo + ALL examples_*.json + results | ~60MB | `git clone` (everything tracked) |
| TB2 trajectories (153 files, 132 compaction events) | 2.1MB | `migration_payload.tar.gz` IN this repo |
| DPO checkpoints (rounds 1+2, LoRA) | 120MB | rsync from narval (below); backstop at narval `$HOME/dpo_checkpoints.tar.gz` |
| HF model cache | ~500GB | re-download on target (internet available) |
| TB2 task sifs | ~25GB | re-bake on target (`tb2-eval/bake_all_sifs.sh`), faster with internet |
| venvs | - | rebuild (same CVMFS stack on all Alliance clusters) |

## Steps on the target cluster

1. Clone:
   `cd $SCRATCH && git clone https://github.com/ayushnangia/does-compression-behavior dccb`
   (use the PAT from .env if private; .env itself must be copied by hand, it is gitignored)
2. Unpack trajectories: `tar xzf migration_payload.tar.gz` -> `tb2_trajectories/`
3. Checkpoints (only if resuming E-B work):
   `rsync narval.alliancecan.ca:/home/anangia/dpo_checkpoints.tar.gz .`
4. Venvs (adjust python module names to target's `module avail`):
   - compress: `module load cuda python/3.11 gcc arrow; python -m venv $SCRATCH/ENV-compress2`
     then `pip install torch transformers trl peft accelerate` (+ `fla-core causal-conv1d` for Qwen3.5)
   - vllm: `module load cuda python/3.12; python -m venv $SCRATCH/ENV-vllm2; pip install vllm`
   - harbor: python 3.12 venv, `pip install harbor==0.20`
5. Models: `HF_HOME=$SCRATCH/hf python prefetch.py` (or plain `huggingface-cli download` - internet works)
6. Gate: `python tests/run_tests.py` before ANY submission (same rule as Narval)

## Cluster-specific notes

- Accounts: def-zhijing / def-rgrosse are Alliance-wide; Killarney may need the
  aip- allocation (check `sshare -U` after login)
- H100 needs torch >= 2.5.1 (our stack is 2.9, fine)
- Slurm GPU spec changes: `--gpus-per-node=a100:1` -> check target's gres name
  (`h100`, `h100_80gb`, or bare `--gpus=1`; see target docs)
- torch 2.9 reads PYTORCH_CUDA_ALLOC_CONF not PYTORCH_ALLOC_CONF; job scripts
  set both (keep it)
- vllm scorer: chunked prefill (max_num_batched_tokens=1024) + util 0.80 is
  REQUIRED for exact_tool_distribution on long contexts (7.6GB fp32 logprob
  spike otherwise; see AUDIT.md exp19 entry). On 80GB cards you may relax to
  util 0.90 but keep the chunking.
- Apptainer: same workflow if needed, but with compute-node internet harbor
  can pull images directly on some clusters - test one task first

## First H100 jobs, in order

1. tb2_bf16: Qwen3.5-35B-A3B single-GPU vLLM serve + Terminus-2, easy-25
   subset first. This is the run Narval could not do; expect nonzero Pass@1
   (paper reports 27% class for this model on TB2).
2. If (1) produces solved tasks: regenerate on-policy 16k/32k examples from
   COMPETENT trajectories, rerun exp8 grounding with task success available.
3. 64k arms of exp4/exp8 (previously OOM).
4. GLM-4.5-Air (106B) on 3xH100 if the group wants oracle-scale.

## Trillium-specific (if target = Trillium, most H100 nodes: 63 nodes x 4x H100-SXM-80GB, NVLink)

Trillium is like Narval in the one way that matters: **no internet on compute
nodes**. Our offline discipline ports unchanged (prefetch on login node,
HF_HUB_OFFLINE=1, --examples-file, sifs baked on login). Differences:

1. Login: `trillium.alliancecan.ca` (CPU) / `trillium-gpu.alliancecan.ca` (GPU side)
2. **Whole-node scheduling**: no `--mem` needed (you get the node's 749GB);
   GPU request is `--gpus-per-node=h100:N` (N=1-4). Drop `--mem` and
   `--cpus-per-task` lines from job headers.
3. **venv goes in $HOME on the login node** (SciNet recommendation), NOT
   scratch: `python -m venv ~/ENV-compress2` etc. Update job scripts'
   `source` lines accordingly.
4. **`debugjob -g 1`** gives an interactive GPU for up to 2h with fast start.
   USE THIS FIRST for the vLLM 35B serve test - it is exactly the interactive
   debug loop Narval never gave us (home/project are read-only inside
   debugjob; work from scratch).
5. NVLink within node: the 106B GLM-4.5-Air fits TP=4 on ONE node with fast
   interconnect - Trillium is the only listed cluster where that is routine.
6. Storage is VAST NVMe (fast small-file reads) - the HF cache and sif
   copies will be quicker than Narval Lustre.

Trillium first-command sequence (VERIFIED on trig-login01, 2026-07-23):
```bash
ssh trillium.alliancecan.ca
cd $SCRATCH && git clone <repo> dccb && cd dccb && tar xzf migration_payload.tar.gz
# NOTE: gcc must be loaded BEFORE cuda; use virtualenv --no-download, NOT
# python -m venv (plain venv cannot see module-provided packages, so pip
# pulls the pyarrow/opencv "noinstall" dummies and dies)
module load gcc cuda python/3.11 arrow/19.0.1
virtualenv --no-download ~/ENV-compress2 && source ~/ENV-compress2/bin/activate
pip install --no-index torch transformers trl peft accelerate fla-core \
    causal-conv1d 'datasets>=3,<4' sentencepiece protobuf
# datasets must stay <4: datasets 5.x wants pyarrow>=21, arrow/19.0.1 has 19.0.1
deactivate
# vllm venv: opencv/4.13.0 required (vllm dep opencv-python-headless>=4.13)
module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0
virtualenv --no-download ~/ENV-vllm2 && source ~/ENV-vllm2/bin/activate
pip install --no-index vllm    # -> vllm 0.25.0, torch 2.11
deactivate
# harbor venv: no Alliance wheel, comes from PyPI (login node has internet)
virtualenv --no-download ~/ENV-harbor2 && source ~/ENV-harbor2/bin/activate
pip install harbor==0.20
deactivate
HF_HOME=$SCRATCH/hf hf download Qwen/Qwen3.5-35B-A3B   # 67GB, ~15 min
python tests/run_tests.py                        # gate, as always
# TB2 tasks WITHOUT touching Narval: harbor registry has them
mkdir -p $SCRATCH/tb2/{jobs,sif_cache}
harbor download terminal-bench@2.0 -o $SCRATCH/tb2      # 89 tasks -> tb2/terminal-bench/
bash experiments/bake_sifs_trillium.sh           # pre-bake easy-25 sifs (~44GB)
debugjob -g 1 srun bash experiments/smoke_35b_trillium.sh   # vLLM 35B single-GPU smoke test
# or, queue-through-maintenance chain:
#   sbatch experiments/smoke_35b_sbatch.sh, then tb2_bf16_trillium_job.sh
#   with --dependency=afterok:<smoke_id>
```

Trillium smoke test VERIFIED on a login-node H100 (2026-07-23, maintenance
still on; login nodes have 4x H100 and SciNet blesses short tests there):
single GPU serves bf16 35B at 32k, 74.1GB used, 126k KV tokens, completion OK.
Two failure modes found and fixed en route, baked into the scripts:
1. NO `stdbuf` around `vllm serve`: Gentoo's libstdbuf.so is LD_PRELOADed
   into nvcc's /bin/sh (system glibc) and every flashinfer JIT build dies
   with GLIBC_ABI_DT_RELR. PYTHONUNBUFFERED=1 suffices.
2. `--max-num-seqs 128`: Qwen3.5 linear-attention needs one Mamba cache
   block per decode seq; only ~135 blocks fit next to 66GB of weights, and
   vllm's default max_num_seqs=1024 aborts engine init.
Flashinfer JIT kernels now cached in ~/.cache/flashinfer for compute jobs.

Trillium files added during migration:
- `experiments/bake_sifs_trillium.sh` - pre-bake TB2 sifs into harbor's exact
  cache naming (`<image with / and : -> _>.sif`), from the config's task list
- `experiments/smoke_35b_trillium.sh` - single-H100 bf16 35B serve + one
  completion; the 5-minute gate before the 16h job
- `experiments/smoke_35b_sbatch.sh` - same, as a batch job for dependency chains
- `experiments/tb2_bf16_trillium_job.sh` - the TB2 easy-25 run, Trillium
  headers (h100:1, TP=1, no --mem/--cpus), venvs from $HOME

## What does NOT need to move

- Old job logs (*.out): archived in git history where they mattered (AUDIT.md
  records every verdict); raw logs stay on Narval until scratch purge
- Narval-specific job headers: every experiments/*_job.sh needs the gres line
  edited anyway - do it per-job as you resubmit, not in bulk
