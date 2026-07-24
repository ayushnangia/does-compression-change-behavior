# Does compression change behavior?

When an LLM agent's context is compacted (summarized, trimmed, rewritten),
does the agent still do the same thing? We measure it directly: give the
same model the full history and a compressed history, sample its next
actions from both, and count the disagreement - always against the noise
floor of sampling twice from the identical context.

21 experiments on Qwen3.5 (9B/27B/35B-A3B) and GLM-4.7-Flash, on-policy
agent traces from 4k to 64k tokens. Every result JSON is in
`experiments/results/`; every claim's confounds are in [AUDIT.md](docs/AUDIT.md).

## Findings

| # | finding | numbers | experiment |
|---|---|---|---|
| 1 | **Agents freeze when you delete their own past actions.** Tool calls are 25% of tokens but carry the behavior; tool outputs are 36% and nearly free to delete. | halt rate 47% vs 19% control; observations = random control (D 0.50 vs 0.51). Replicated on GLM. | exp4 |
| 2 | **Copying beats rewriting, predictably.** Verbatim n-gram overlap between compressed and original predicts behavior preservation; fluency (NLL) predicts nothing. | Spearman rho = -0.55 (containment) vs +0.11 (NLL) | exp20 |
| 3 | **Models react to the format of memory, not just content.** Same content, wrong wrapper: one model drops from 75% acting to 0%. One-liner shorthand loses 19 points bare vs inside native `<tool_calls>` tags. | +19 pts, p=0.013 (content held constant) | exp21, exp14 |
| 4 | **The useful memory is tiny.** The action skeleton at approx 2% of a 16k context keeps most behavior; canonical one-liners fit the entire history in approx 870 tokens. | 0.59 agreement at 330 tokens; extractive-2% beats abstractive summaries by approx 30 pts | exp17, exp21 |
| 5 | **Select, don't train.** Generating N compressions and picking by measured behavior change works; DPO-training a compressor on those pairs gave a clean null. | selection p=0.0004 (+16k replication); DPO D 0.73 vs base 0.75, p=0.35 | exp11, expB |
| 6 | **Summaries are the worst compressor tested**, at every budget, in every head-to-head. Plain keep-recent is the boring champion at normal budgets. | e.g. exp21: keep-recent 0.68 vs summary approx 0.44-class results throughout | exp6, exp9, exp15 |
| 7 | **Sampled tool-level distortion is mostly noise; floor-referencing is load-bearing.** Exact logprob TV is 10x smaller than the 8-sample estimate. | exact 0.029 vs sampled 0.275 | exp19 |

Plus two analyses of the CompactionRL training scheme (arXiv:2607.05378):
credit decays to 13.5% by chain depth 4 (exp1) and segments-as-samples
inflates GRPO gradient mass 3.2x (exp2).

**Honest limits**: grounding is agreement-with-logged-actions (0.67-0.73),
not task success - quantized models score 0% on Terminal-Bench (see
`tb2/README.md`); most N are 13-24; one trace domain. Full ledger: AUDIT.md.

## Repo layout

```
behavior.py compressors.py data.py metrics.py scaffold.py    core library
experiments/exp*.py       one experiment each, self-documenting headers
experiments/run_all.sh    reproduce everything: cpu | gpu | queue
experiments/results/      every result JSON ever produced
data/                     all measurement datasets (4k-64k) + raw trajectories
tb2/                      offline Terminal-Bench 2.0 harness (vLLM+harbor+Apptainer)
docs/                     AUDIT (claims ledger), COAUTHOR (full briefing),
                          MIGRATION (cluster runbook), slides
tests/run_tests.py        68 checks; gate for every job submission
```

Docs: [AUDIT.md](docs/AUDIT.md) claims and confounds ledger -
[COAUTHOR.md](docs/COAUTHOR.md) complete technical briefing -
[MIGRATION.md](docs/MIGRATION.md) H100 cluster runbook -
[slides.html](docs/slides.html) talk.

Everything else (Slurm job scripts per cluster, figures, planning docs,
paper-review notes) lives on the `research-archive` branch - main carries
only what you need to read, run, and check the findings.

## Quickstart

```bash
pip install -r requirements.txt
python tests/run_tests.py                 # no GPU needed
cd experiments && bash run_all.sh cpu     # analyses, minutes

# with a GPU (A100-40GB+; H100 recommended):
HF_HOME=... python ../prefetch.py         # cache models (internet needed once)
bash run_all.sh gpu                       # or: run_all.sh queue (Slurm)
```

The method in one paragraph: for each example we hold out the recent turns,
compress the older history with the compressor under test, then sample 8
next actions from (compressed + recent) and 8 from the full context.
Distortion D is total variation between the two action distributions at
three granularities (tool / tool+verb / exact args), floor-referenced
against a full-vs-full resample, with paired permutation tests across
examples. On-policy traces come from our own TB2 runs (`tb2/`); grounded
agreement compares sampled actions to the action the agent actually took.
