# EXP 22: Outcome-grounded compaction (the harbor experiment)

The experiment every reviewer will ask for: plug compaction POLICIES into
live Terminal-Bench episodes and measure Pass@1 per policy. Bridges our
behavioral metric D to task outcome.

## Why Terminus-2, not Terminus-KIRA
KIRA (CompactionRL's scaffold) is not in harbor 0.20 (latest). Terminus-2
has native compaction: a 3-step summarize->question->answer subagent
pipeline triggered proactively (default: <8000 free tokens) and at context
limit. That stock pipeline IS our arm C - the deployed summarizer, unmodified.
Our arms subclass Terminus2 and override ONLY `_summarize()` (triggering
identical by construction), loaded via harbor's import-path factory - no fork.

## Arms (compaction_agents.py, consumption-tested against harbor 0.20)
| arm | agent | policy |
|---|---|---|
| A | KeepRecentTerminus | newest messages verbatim, rest dropped |
| B | RawSkeletonTerminus | byte-exact command-bearing messages + verbatim tail |
| C | StockCappedTerminus | deployed 3-step summarizer |
| D | Terminus2 enable_summarize=False | no compaction (truncation) baseline |
| E | LearnedSelectorTerminus | frozen Qwen3.5 LoRA selects exact blocks from Qwen3.8's own live history |

A/B/C/E use the assigned policy for the first three compaction events, then an
identical keep-recent fallback. Arm E is delegated self-compaction: it emits
only `{"keep":[...]}`, uses the same 24k-character handoff envelope, and logs
every invalid/overflow/endpoint fallback for intention-to-treat reporting. It
may run only after held-out selector validation freezes an adapter hash. Arm B
was changed after exp23 rejected the
canonical one-liner proposal: raw skeleton+tail tied keep-recent, while
rewriting lost about 13 agreement points.

## CompactionRL parity (their experiments section, adopted verbatim)
- temperature 1.0, top_p 1.0 (serving side)
- up to 250 interaction turns
- <= 3 compaction operations per trajectory
- full model context window; Pass@1, mean of 2 runs

## Run matrix and cost
Phase 1 freezes one lineage: `Qwen/Qwen3.8-27B` bf16. Five arms x 89 tasks x
up to 2 runs = 890 episodes. The easy25 baseline/pilot gates that spend. The
learned arm first runs a one-task endpoint/compaction smoke, then easy25; no
full-89 allocation is released from an invalid or fallback-dominated pilot. A
second model is deferred until one complete, valid Qwen3.8 row exists.
Binomial reality: at ~25% base rate, 89x2 resolves arm gaps of roughly 7pts;
smaller true gaps are why D is reported alongside outcomes.

## Readouts
1. Pass@1 per arm (A vs B is a tie hypothesis; learned E is paired against A,
   while C and D remain summary/no-compaction baselines with no assumed ordering)
2. D measured offline at each logged compaction event vs the arm's Pass@1
   delta - the D->outcome bridge (the paper's practical payoff)
3. Post-compaction halt/derail rates from trajectories (ties to finding 1)

## Validity gates

- Agent/arm smoke (July 30, job 66693784): green for custom-agent loading and
  live compaction, but its reward is not valid evidence because it predated the
  verifier repair.
- Offline verifier smoke (Aug 6): green; real pytest executes and writes reward.
- Oracle smoke (Aug 18): **green, reward=1** on
  `break-filter-js-from-html`; reproducible via `tb2/oracle_smoke.sh`.

## Original agent smoke details

One-task episode on Narval A100 (9B): vLLM up (ENV-vllm3), custom agent
loaded via import_path, 18 live turns, OUR COMPACTION POLICY FIRED 15 TIMES
(3 policy + capped fallback), verifier ran, trajectory recorded. Reward 0.0
as expected for the 9B - capability was never the smoke's question.
Eight attempts; every failure was infrastructure, each fixed and documented
in the job script itself. Two knobs for the real run:
- raise the per-episode agent timeout (AgentTimeoutError at ~20 min here)
- compaction fires frequently at threshold 8000 on a 32k window - real runs
  on bigger windows will be closer to CompactionRL's <=3 ops regime

All previously-open config questions verified in the smoke: import_path
syntax, max_turns kwarg, TaskConfig dicts, unique ports and job names.
