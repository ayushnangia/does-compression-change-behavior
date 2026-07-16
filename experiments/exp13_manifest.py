"""EXP 13 (GPU) — Recoverable compression: does a DELETION MANIFEST help?

Idea (P. Chopra / group thread): after compaction, inject a note saying WHAT
was deleted and HOW to retrieve it — not just a content summary. Rationale:
our measured failure mode is halting (the agent doesn't know what it doesn't
know); a manifest converts unknown-unknowns into known-unknowns with a
retrieval path.

Conditions at one matched budget (all in the same wrapper, controlling the
format seam):
  summary           — model-written content summary (status quo)
  manifest_only     — deterministic deletion manifest: files touched,
                      commands run, counts, retrieval instructions. No
                      model generation -> no hallucination risk.
  summary_manifest  — manifest first, summary in the remaining budget.

Two evaluation modes:
  plain     — next-action: acting / change / lookup-share / grounded agree
  scaffold  — the recovery test (tool menu + lookups served from the real
              deleted history): where "how to retrieve" should actually pay.

    python exp13_manifest.py --examples-file ../examples_64.json
"""

from __future__ import annotations

import argparse
import re

from common import load_model, save_result, REPO  # noqa: F401
from stats import fmt_ci, paired_permutation_p
from behavior import action_kind, sample_actions
from compressors import PATH_RE, SUMMARIZE, _generate, _ids
from data import load_examples_file
from metrics import acting_rate, action_change
from scaffold import recover_actions

WRAP = ("<turn index=0 role=user>\n<content>\n{body}\n</content>\n</turn>\n")
CMD_RE = re.compile(r'\\?"command\\?":\s*\\?"([^"\\]{5,90})')


def build_manifest(old_text: str) -> str:
    """Deterministic deletion manifest: what was cut + how to get it back."""
    files = list(dict.fromkeys(PATH_RE.findall(old_text)))[:10]
    cmds = list(dict.fromkeys(CMD_RE.findall(old_text)))[:6]
    turns = old_text.count("<turn ")
    calls = old_text.count("<tool_calls>")
    lines = [f"[Compaction notice: {turns} turns including {calls} tool calls "
             "were deleted from the context.]"]
    if files:
        lines.append("Deleted work involved these files (re-read them to "
                     "recover details): " + ", ".join(files))
    if cmds:
        lines.append("Commands previously run (re-run to recover their "
                     "output): " + "; ".join(cmds))
    lines.append("If information seems missing, retrieve it via the files/"
                 "commands above before acting.")
    return "\n".join(lines)


def build_condition(name, old_ids, old_text, budget, tokenizer, model, device):
    if name == "summary":
        gen = _generate(model, tokenizer, old_ids + _ids(tokenizer, SUMMARIZE),
                        budget, device)
        body = "[Summary of work so far: " + \
            tokenizer.decode(gen, skip_special_tokens=True).strip() + "]"
    elif name == "manifest_only":
        body = build_manifest(old_text)
    elif name == "summary_manifest":
        manifest = build_manifest(old_text)
        used = len(_ids(tokenizer, manifest))
        room = max(32, budget - used)
        gen = _generate(model, tokenizer, old_ids + _ids(tokenizer, SUMMARIZE),
                        room, device)
        body = manifest + "\n[Summary of work so far: " + \
            tokenizer.decode(gen, skip_special_tokens=True).strip() + "]"
    else:
        raise ValueError(name)
    ids = _ids(tokenizer, WRAP.format(body=body))
    return ids[: budget + 64]      # small slack for the wrapper itself


def agreement(actions, logged):
    if logged is None:
        return None
    lt = logged.split("::", 1)[0]
    return sum(a is not None and a.split("::", 1)[0] == lt
               for a in actions) / len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=24)
    ap.add_argument("--scaffold-examples", type=int, default=12,
                    help="recovery-mode examples (serial, expensive)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)
    conds = ["summary", "manifest_only", "summary_manifest"]

    plain = {c: {"acting": [], "change": [], "lookup": [], "agree": []}
             for c in conds}
    scaf = {c: {"acting": [], "change": [], "used": []} for c in conds}
    floors = []
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        old = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old, skip_special_tokens=True)

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        for c in conds:
            ids = build_condition(c, old, old_text, args.budget,
                                  tokenizer, model, device)
            ctx = ids + ex.recent_ids
            acts = sample_actions(model, tokenizer, ctx, device,
                                  samples=args.samples, seed=base + 3)
            plain[c]["acting"].append(acting_rate(acts))
            plain[c]["change"].append(action_change(full_a, acts))
            plain[c]["lookup"].append(
                sum(action_kind(a) == "lookup" for a in acts) / len(acts))
            ag = agreement(acts, ex.logged_action)
            if ag is not None:
                plain[c]["agree"].append(ag)

            # recovery mode on the first --scaffold-examples usable examples:
            # lookups are served the REAL deleted history (scaffold.py)
            if usable <= args.scaffold_examples:
                racts, used = recover_actions(
                    model, tokenizer, ctx, old_text, device,
                    samples=args.samples, seed=base + 4)
                scaf[c]["acting"].append(acting_rate(racts))
                scaf[c]["change"].append(action_change(full_a, racts))
                scaf[c]["used"].append(sum(used) / len(used))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   floor: {floor:.3f}   budget: {args.budget}\n")
    print("--- plain next-action ---")
    print(f"{'condition':18s} {'acting (CI)':>16s} {'change (CI)':>16s} "
          f"{'lookup':>7s} {'agree':>6s}")
    rows = {}
    for c in conds:
        n = len(plain[c]["acting"])
        ag = plain[c]["agree"]
        rows[c] = {
            "acting": sum(plain[c]["acting"]) / n,
            "change": sum(plain[c]["change"]) / n,
            "lookup": sum(plain[c]["lookup"]) / n,
            "agree": sum(ag) / len(ag) if ag else float("nan"),
        }
        print(f"{c:18s} {fmt_ci(plain[c]['acting']):>16s} "
              f"{fmt_ci(plain[c]['change']):>16s} "
              f"{rows[c]['lookup']:7.2f} {rows[c]['agree']:6.2f}")
    # PRIMARY ENDPOINT: acting rate, summary_manifest vs summary (plain mode)
    p = paired_permutation_p(plain["summary_manifest"]["acting"],
                             plain["summary"]["acting"])
    print(f"paired p (acting, summary_manifest vs summary): {p:.3f}")

    srows = {}
    if scaf["summary"]["acting"]:
        print("\n--- recovery scaffold (lookups served from deleted history) ---")
        print(f"{'condition':18s} {'acting':>7s} {'change':>7s} {'used lookup':>12s}")
        for c in conds:
            n = len(scaf[c]["acting"])
            srows[c] = {k: sum(scaf[c][k]) / n for k in ("acting", "change", "used")}
            print(f"{c:18s} {srows[c]['acting']:7.2f} {srows[c]['change']:7.2f} "
                  f"{srows[c]['used']:12.2f}")

    save_result("exp13_manifest", {
        "model": args.model, "usable": usable, "floor": floor,
        "budget": args.budget, "plain": rows, "scaffold": srows,
        "raw": {c: plain[c] for c in conds},
    })


if __name__ == "__main__":
    main()
