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
| B | OneLinerTerminus | full canonical action history (exp21, wrapped) + verbatim tail |
| C | Terminus2 stock | deployed 3-step summarizer |
| D | Terminus2 enable_summarize=False | no compaction (truncation) baseline |

## CompactionRL parity (their experiments section, adopted verbatim)
- temperature 1.0, top_p 1.0 (serving side)
- up to 250 interaction turns
- <= 3 compaction operations per trajectory
- full model context window; Pass@1, mean of 2 runs

## Run matrix and cost
4 arms x 2 models (Qwen3.5-35B-A3B, Qwen-AgentWorld-35B-A3B) x 89 tasks x 2
runs = 1,424 episodes ~ 700 H200-hours. Binomial reality: at ~25% base rate,
89x2 resolves arm gaps of ~7pts; smaller true gaps are why D exists (report
both).

## Readouts
1. Pass@1 per arm (primary, pre-registered ordering hypothesis: B >= A > C >= D)
2. D measured offline at each logged compaction event vs the arm's Pass@1
   delta - the D->outcome bridge (the paper's practical payoff)
3. Post-compaction halt/derail rates from trajectories (ties to finding 1)

## Still to verify before H200 window (each free, login node)
- [ ] harbor job-config syntax for import-path agents (factory supports it;
      confirm the YAML field name on a one-task smoke)
- [ ] cap-at-3-compactions kwarg (or add counter guard in subclasses)
- [ ] n_max_turns=250 config field
- [ ] one-task smoke on Narval A100 (9B model) end-to-end before any H200 time
