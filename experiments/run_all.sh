#!/bin/bash
# Reproduce every finding in the README, in dependency order.
#
#   bash run_all.sh cpu    # analyses that need no GPU (minutes)
#   bash run_all.sh gpu    # the core behavioral suite (one A100/H100, ~2 days)
#   bash run_all.sh queue  # submit each GPU experiment as its own Slurm job
#
# Every experiment writes results/<name>_<timestamp>.json and prints its
# summary table. Verdicts and caveats for each: ../AUDIT.md.
set -eu
cd "$(dirname "$0")"
MODE=${1:?usage: run_all.sh cpu|gpu|queue}

# gate: nothing runs if the test suite fails
( cd .. && python tests/run_tests.py )

CPU_EXPS=(
  "exp1_credit_decay.py"                # C1: credit decay arithmetic
  "exp2_grpo_weighting.py"              # C2: GRPO segment bias (Monte Carlo)
)

# name : script : data file (all data files are tracked in the repo root)
GPU_EXPS=(
  "exp3:exp3_target_stability.py:../data/examples_64.json"
  "exp4:exp4_block_ablation.py:../data/examples_64.json"          # THE BLOCK LAW
  "exp5:exp5_format_vs_content.py:../data/examples_64.json"
  "exp6:exp6_rate_distortion.py:../data/examples_64.json"
  "exp7:exp7_compaction_chain.py:../data/examples_64.json"
  "exp8:exp8_grounded_agreement.py:../data/examples_onpolicy.json" # grounding
  "exp9:exp9_summary_policies.py:../data/examples_16k.json"
  "exp10:exp10_propagation.py:../data/examples_16k.json"
  "exp11:exp11_best_of_n.py:../data/examples_16k.json"             # SELECTION WORKS
  "exp12:exp12_portability.py:../data/examples_64.json"
  "exp14:exp14_interface_fragility.py:../data/examples_16k.json"   # FORMAT CLIFFS
  "exp17:exp17_minimal_core.py:../data/examples_16k_large.json"    # MINIMAL CORE
  "exp20:exp20_ood_bridge.py:../data/examples_64.json"             # CONTAINMENT LAW
  "exp21:exp21_canonical_skeleton.py:../data/examples_16k_large.json" # ONE-LINERS
)

case $MODE in
cpu)
  for s in "${CPU_EXPS[@]}"; do echo "== $s =="; python "$s"; done
  ;;
gpu)
  for e in "${GPU_EXPS[@]}"; do
    IFS=: read -r name script data <<< "$e"
    echo "== $name: $script =="
    python "$script" --examples-file "$data"
  done
  ;;
queue)
  for e in "${GPU_EXPS[@]}"; do
    IFS=: read -r name script data <<< "$e"
    sbatch --job-name "$name" --gpus-per-node=1 --cpus-per-task=8 --mem=64G \
        --time=0-12:00 --output="${name}_%j.out" \
        --wrap "module load cuda python gcc arrow 2>/dev/null;
                source \$SCRATCH/ENV-compress2/bin/activate;
                export HF_HOME=\$SCRATCH/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1;
                export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
                python $script --examples-file $data"
  done
  ;;
esac
