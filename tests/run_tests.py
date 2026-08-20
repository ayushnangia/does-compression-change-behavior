"""BS-test for the whole repo: every function testable without a GPU.

    python tests/run_tests.py        (from the repo root)
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

PASS, FAIL = 0, []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(f"{name}: {detail}")
        print(f"  FAIL {name} {detail}")


# ---------------- behavior.py ----------------
from behavior import parse_action, action_kind, MENU_KIND

check("parse plural", parse_action(
    '<tool_calls>\n[{"function": {"name": "execute_bash", "arguments": "x"}}]\n</tool_calls>'
) == "execute_bash::x")
check("parse singular", (parse_action(
    '<tool_call>{"name": "edit", "arguments": {"p": 1}}</tool_call>') or "").startswith("edit::"))
check("parse python-repr", (parse_action(
    "<tool_calls>\n[{'function': {'arguments': '{\"command\": \"ls\"}', 'name': 'execute_bash'}}]\n</tool_calls>"
) or "").startswith("execute_bash::"))
check("parse qwen-native-xml", (parse_action(
    "<tool_call>\n<function=execute_bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>")
    or "").startswith("execute_bash::"))
check("parse think-strip", parse_action("<think>call <tool_call>{\"name\":\"x\"}</tool_call></think>done") is None)
check("parse think-then-act", (parse_action(
    '<think>hm</think><tool_call>{"name": "run_tests"}</tool_call>') or "") == "run_tests")
check("parse unclosed think", parse_action("<think>endless deliberation") is None)
check("parse set-args (job-killer regression)", (parse_action(
    "<tool_calls>[{'function_name': 'b', 'arguments': {'x', 'y'}}]</tool_calls>") or "").startswith("b::"))
check("parse prose", parse_action("The task is complete.") is None)
check("parse menu", parse_action('I will read_file("src/a.py") now') == 'read_file::"src/a.py"')
check("parse menu-hedged-prose", parse_action("Maybe I could grep(pattern) the file first.") is None)
check("parse menu-example-prose", parse_action("Use a tool, for example edit(path).") is None)
check("parse glm-native", (parse_action(
    "<tool_call>execute_bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>")
    or "").startswith("execute_bash::"))
check("parse raw Terminus JSON", (parse_action(
    'Analysis first\n{"analysis":"x","plan":"y","commands":'
    '[{"keystrokes":"ls -la\\n","duration":0.1}]}') or "").startswith(
        'bash_command::'))
check("parse raw Terminus empty commands", parse_action(
    '{"analysis":"done","plan":"none","commands":[]}') is None)
check("kind menu-lookup", action_kind("read_file::x") == "lookup")
check("kind menu-commit", action_kind("submit::") == "commit")
check("kind bash-ls", action_kind('execute_bash::{"command": "ls -la"}') == "lookup")
check("kind bash-edit", action_kind('str_replace_editor::{"command": "create"}') == "commit")
check("kind none", action_kind(None) == "none")
a1 = parse_action('<tool_calls>[{"function":{"name":"execute_bash","arguments":"{\\"command\\": \\"go test\\"}"}}]</tool_calls>')
a2 = parse_action('<tool_calls>[{"function":{"name":"execute_bash","arguments":"{\\"command\\": \\"ls src\\"}"}}]</tool_calls>')
check("labels distinguish args", a1 != a2, f"{a1} vs {a2}")

# ---------------- metrics.py ----------------
from metrics import (acting_rate, action_change, action_change_tools,
                     action_entropy, lookup_rate, normalized_change)

check("acting_rate", acting_rate([None, "a", "b", None]) == 0.5)
check("change identical", action_change(["a", "b"], ["a", "b"]) == 0.0)
check("change disjoint", action_change(["a"], ["b"]) == 1.0)
rng = random.Random(0)
for _ in range(200):
    xs = [rng.choice(["a", "b", None]) for _ in range(8)]
    ys = [rng.choice(["a", "c", None]) for _ in range(8)]
    d1, d2 = action_change(xs, ys), action_change(ys, xs)
    if not (abs(d1 - d2) < 1e-12 and 0.0 <= d1 <= 1.0):
        check("change symmetric+bounded", False, f"{d1} {d2}")
        break
else:
    check("change symmetric+bounded (fuzz x200)", True)
check("coarse ignores args", action_change_tools(["a::x", "b::y"], ["a::z", "b::w"]) == 0.0)
check("coarse sees tools", action_change_tools(["a::x"], ["b::x"]) == 1.0)
check("entropy collapsed", action_entropy(["a"] * 8) == 0.0)
check("entropy uniform8", abs(action_entropy(list("abcdefgh")) - 3.0) < 1e-9)
check("lookup_rate", lookup_rate(["read_file::x", "submit::", None], action_kind) == 1 / 3)
check("normalized floor", normalized_change(0.3, 0.3) in (0.0,))
import math
check("normalized degenerate", math.isnan(normalized_change(0.5, 1.0, 1.0)))

# ---------------- stats.py ----------------
from stats import bootstrap_ci, paired_permutation_p, mean

xs = [0.4, 0.5, 0.6, 0.55, 0.45, 0.5, 0.62, 0.38]
lo, hi = bootstrap_ci(xs)
check("ci brackets mean", lo <= mean(xs) <= hi, f"{lo} {mean(xs)} {hi}")
check("perm p null high", paired_permutation_p([1, 2, 3, 4, 5], [1.01, 2, 3, 4, 5]) > 0.3)
check("perm p effect low", paired_permutation_p([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]) < 0.1)
# calibration under the null: p should be roughly uniform
rng = random.Random(1)
ps = []
for t in range(120):
    a = [rng.gauss(0, 1) for _ in range(10)]
    b = [x + rng.gauss(0, 1) for x in a]
    ps.append(paired_permutation_p(a, b, n_perm=400, seed=t))
frac05 = sum(p <= 0.05 for p in ps) / len(ps)
check("perm p calibrated under null", frac05 <= 0.12, f"frac(p<=.05)={frac05:.2f}")

# ---------------- scaffold.py ----------------
import scaffold
from scaffold import serve_lookup, MENU_INSTRUCTIONS

old = "edited src/util/parse.py then saw TypeError at line 42"
check("lookup finds path", "parse.py" in serve_lookup("read_file::src/util/parse.py", old))
check("lookup finds text", "TypeError" in serve_lookup("grep::TypeError", old))
check("lookup notfound", "no earlier record" in serve_lookup("read_file::nope.zz", old))
check("lookup empty arg", "nothing specified" in serve_lookup("grep::", old))
for tool in ("read_file", "grep", "run_tests", "edit", "submit"):
    check(f"menu advertises {tool}", tool in MENU_INSTRUCTIONS and tool in MENU_KIND)


class Tok:
    def __call__(self, t, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in t]}
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


tok = Tok()
script = iter(['read_file("src/util/parse.py")', 'edit("parse.py")'])
scaffold.sample_texts = lambda *a, **k: [next(script)]
act, used = scaffold.recover_action(None, tok, tok("c")["input_ids"], old, None, max_lookups=2)
check("recover lookup->commit", used and action_kind(act) == "commit", f"{act} {used}")
scaffold.sample_texts = lambda *a, **k: ["nothing to do here"]
act, used = scaffold.recover_action(None, tok, tok("c")["input_ids"], old, None, max_lookups=2)
check("recover halt", act is None and not used)

# ---------------- data.py ----------------
from data import Example, save_examples, load_examples_file

ex = Example(context_ids=[1, 2, 3, 4], recent_ids=[3, 4], repo="r",
             logged_action="a::b", future_segments=[[5, 6]])
save_examples([ex], "/tmp/_t.json")
back = load_examples_file("/tmp/_t.json")[0]
check("example roundtrip", back == ex)
check("old-file compat", Example(**{"context_ids": [1], "recent_ids": [1], "repo": "x"}).logged_action is None)

# ---------------- compressors.py ----------------
from compressors import keep_recent, pointer, hallucinator

check("keep_recent half", keep_recent(list(range(10)), tok, None, None) == list(range(5, 10)))
ptr = tok.decode(pointer(tok("work in src/main.py done")["input_ids"], tok, None, None))
check("pointer cites file", "src/main.py" in ptr)
check("hallucinator asserts done", "proceed" in tok.decode(hallucinator([1], tok, None, None)))

# ---------------- exp4 / exp6 / exp13 / exp11 helpers ----------------
from exp4_block_ablation import BLOCK_RES, random_trim
trace = ("<turn index=1 role=assistant>\n<content>\nthinking...\n</content>\n"
         "<tool_calls>\nCALL\n</tool_calls>\n</turn>\n"
         "<turn index=2 role=tool>\n<content>\nOBSERVATION\n</content>\n</turn>\n")
obs = BLOCK_RES["drop_observations"].sub(lambda m: m.group(1), trace)
rea = BLOCK_RES["drop_reasoning"].sub(lambda m: m.group(1), trace)
check("drop_obs keeps reasoning", "thinking" in obs and "OBSERVATION" not in obs)
check("drop_rea keeps obs", "OBSERVATION" in rea and "thinking" not in rea)
check("both keep tool calls", "CALL" in obs and "CALL" in rea)
out = random_trim(list(range(1000)), 300, random.Random(0))
check("trim budget+order", len(out) <= 300 and all(a < b for a, b in zip(out, out[1:])))

from exp6_rate_distortion import compress as e6_compress
for name in ("keep_recent", "random", "block_aware"):
    ids = e6_compress(name, tok(trace * 30)["input_ids"], trace * 30, 200, tok, None, None, random.Random(0))
    check(f"exp6 {name} budget", len(ids) <= 200, str(len(ids)))

from exp13_manifest import build_manifest
m = build_manifest(trace + '"command": "grep -n foo bar.py"')
check("manifest counts turns", "2 turns" in m)
check("manifest retrieval hint", "re-run" in m or "re-read" in m)

from exp11_best_of_n import agreement
check("agree tool-level", agreement(["a::x", "a::y", "b::z", None], "a::q") == 0.5)
check("agree none-logged", agreement(["a::x"], None) is None)

# ---------------- cross-script consistency ----------------
import re as _re2
wrap_users = {}
for f in ["experiments/exp7_compaction_chain.py", "experiments/exp11_best_of_n.py",
          "experiments/exp12_portability.py", "experiments/exp13_manifest.py",
          "experiments/exp9_summary_policies.py", "experiments/exp6_rate_distortion.py",
          "compressors.py"]:
    t = (REPO / f).read_text()
    # join python adjacent-string-literal continuations: '..." \n "...' -> '...'
    joined = _re2.sub(r'["\']\s*\n\s*f?["\']', '', t)
    if "[Summary of work so far:" in joined:
        wrap_users[f] = True
check("summary wrapper used consistently", len(wrap_users) == 7, str(list(wrap_users)))

# every experiment result JSON on disk parses and has no NaN-breaking values
bad = []
for rf in (REPO / "experiments" / "results").glob("*.json"):
    try:
        json.loads(rf.read_text())
    except Exception as e:
        bad.append(f"{rf.name}: {e}")
check("all result JSONs parse", not bad, str(bad))


# ---- real generation path: sample_texts consumed end-to-end (tiny model) ----
# the monkeypatched checks above never execute model.generate; this does,
# so the stop_strings/tokenizer kwargs are verified against the installed
# transformers, not assumed. Random weights; we only test mechanics.
import behavior as _bh
from transformers import AutoTokenizer as _AT, Qwen2Config as _QC, Qwen2ForCausalLM as _QM
_tok = _AT.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
_cfg = _QC(vocab_size=_tok.vocab_size + 64, hidden_size=64, num_hidden_layers=2,
           num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
           max_position_embeddings=512)
_m = _QM(_cfg).eval()
_ids = _tok("hello world", add_special_tokens=False)["input_ids"]
_texts = _bh.sample_texts(_m, _tok, _ids, "cpu", samples=2, max_new=8, seed=0)
check("generate+stop_strings consumed", len(_texts) == 2 and all(isinstance(x, str) for x in _texts))

# no stubs / TODOs anywhere
import re as _re
hits = []
for f in list(REPO.glob("*.py")) + list((REPO / "experiments").glob("*.py")):
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if _re.search(r"\bTODO\b|\bFIXME\b|\bXXX\b|raise NotImplementedError\(\)|\bpass  # stub", line):
            hits.append(f"{f.name}:{i}")
check("no stubs/TODOs", not hits, str(hits))


# ---------------- D-metric fixes (issues 1,2,5,6,8,9,10) ----------------
from behavior import parse_diagnosis
from metrics import action_change_verbs, debiased_change, harm_score
check("diag acted", parse_diagnosis('<tool_call>{"name": "x"}</tool_call>') == "acted")
check("diag toolish_unparsed", parse_diagnosis('call {"name": "broken') == "toolish_unparsed")
check("diag prose", parse_diagnosis("Task complete.") == "prose")
check("verb same-verb=0", action_change_verbs(
    ['bash_command::{"keystrokes": "ls -a"}'] * 8,
    ['bash_command::{"keystrokes": "ls b"}'] * 8) == 0.0)
_ex, _p = debiased_change(["x"] * 8, ["y"] * 8)
check("debiased detects", _ex > 0.5 and _p < 0.05)
_ex0, _p0 = debiased_change(["x", "y"] * 4, ["x", "y"] * 4)
check("debiased null", _ex0 <= 0.02 and _p0 > 0.3)
_h = harm_score(["a::1"] * 4 + [None] * 4, ["b::2"] * 8, logged="b::9")
check("harm ignores beneficial divergence", _h["halt_increase"] == 0.0 and _h["agree_drop"] == 0.0)
from compressors import TEXT_COMPRESSORS as _TC
check("summary_native registered", "summary_native" in _TC)

# ---------------- adaptive sampling chunks (Aug-4 OOM fix) ----------------
from behavior import sample_chunk_size

check("chunk small ctx = full batch", sample_chunk_size(4096, 8) == 8)
check("chunk 32k = half", sample_chunk_size(32768, 8) == 4)
check("chunk 64k = quarter", sample_chunk_size(65536, 8) == 2)
check("chunk 209k = singles", sample_chunk_size(209715, 8) == 1)
check("chunk never zero", sample_chunk_size(300000, 1) == 1)

# ---------------- exp23 hybrid budget accounting ----------------
# The exp22 lesson encoded as a gate: policies decide WHAT fills the budget,
# never HOW MUCH. fit_hybrid must never exceed the budget, must cap the
# skeleton at half, must fill the rest with a verbatim suffix of old_ids.
from exp23_oneliner_tail import build_units, fit_hybrid


class _StubTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 251 for c in text]}


_old_text = ("turn one <tool_calls>[{\"function\": {\"name\": \"execute_bash\", "
             "\"arguments\": \"{\\\"command\\\": \\\"grep -n foo bar.py\\\"}\"}}]</tool_calls> "
             "obs <tool_calls>[{\"function\": {\"name\": \"read_file\", "
             "\"arguments\": \"{\\\"path\\\": \\\"a.py\\\"}\"}}]</tool_calls> more")
_old_ids = list(range(1000))
_units = build_units(_old_text, _StubTok(), canonical=True)
check("exp23 units extracted", len(_units) == 2,
      f"got {len(_units)}")
check("exp23 units wrapped", all(chr(u[0] % 251) == "<" for u in _units))
_comp, _sk, _tl = fit_hybrid(_old_ids, _units, budget=100)
check("exp23 budget respected", len(_comp) <= 100, f"len={len(_comp)}")
check("exp23 skeleton capped at half", _sk <= 50, f"skel={_sk}")
check("exp23 tail fills remainder exactly", _sk + _tl == len(_comp) and _tl == 100 - _sk)
check("exp23 tail is verbatim suffix", _comp[_sk:] == _old_ids[-_tl:])
_c2, _s2, _t2 = fit_hybrid(_old_ids, _units, budget=10_000)
check("exp23 big budget keeps full skeleton", _s2 == sum(len(u) for u in _units))
_c3, _s3, _t3 = fit_hybrid(_old_ids, _units, budget=16)
check("exp23 tiny budget degrades to pure tail", _s3 == 0 and _c3 == _old_ids[-16:])
_units_raw = build_units(_old_text, _StubTok(), canonical=False)
check("exp23 raw blocks longer than canonical",
      sum(len(u) for u in _units_raw) > sum(len(u) for u in _units))

# ---------------- exp22 live raw-skeleton policy ----------------
from exp22.policy_utils import fit_raw_skeleton

_live_texts = ["OBS-old", "CMD exact --flag x", "OBS-new", "CMD newest"]
_live_mask = [False, True, False, True]
_live_skel, _live_tail = fit_raw_skeleton(_live_texts, _live_mask, 40)
check("exp22 live budget respected", len(_live_skel) + len(_live_tail) <= 40)
check("exp22 live skeleton byte-exact", _live_skel == "CMD newest")
check("exp22 live tail byte-exact", "\n\n".join(_live_texts).endswith(_live_tail))
try:
    fit_raw_skeleton(["x"], [], 10)
    _alignment_raises = False
except ValueError:
    _alignment_raises = True
check("exp22 live policy alignment guard", _alignment_raises)

# ---------------- exp24 direct-reward extractive GRPO ----------------
from exp24_credit import group_advantages, event_weights, selector_reward
from exp24_data import extract_turns, parse_keep, render_selection, stable_split

_ga = group_advantages([0.0, 1.0, 2.0, 3.0])
check("exp24 group advantages centered", abs(sum(_ga)) < 1e-9)
check("exp24 constant group zero", group_advantages([1.0] * 4) == [0.0] * 4)
check("exp24 trajectory mass fixed", abs(sum(event_weights(4)) - 1.0) < 1e-9)
check("exp24 no early-credit decay", len(set(event_weights(4))) == 1)
check("exp24 invalid selection penalized", selector_reward(
      tool_agreement=1, verb_agreement=1, acting_rate=1, valid=False,
      budget_ratio=0.5) < 0)
_trace = "<agent_trace>\n<turn index=0 role=user>A</turn>\n<turn index=1 role=agent>B</turn>\n"
_hdr, _turns = extract_turns(_trace)
check("exp24 exact turns extracted", _hdr == "<agent_trace>\n" and len(_turns) == 2)
_partial_hdr, _ = extract_turns("TRUNCATED GARBAGE<turn index=1 role=agent>B</turn>")
check("exp24 truncated prefix is not free header", _partial_hdr == "")
_keep, _valid = parse_keep('noise {"keep":[1,0,1]}', 2)
check("exp24 selector JSON strict", _valid and _keep == [0, 1])
check("exp24 selector rejects range", not parse_keep('{"keep":[2]}', 2)[1])
check("exp24 parser finds first valid JSON", parse_keep(
      'bad {x} then {"keep":[1]} trailing {"junk":0}', 2) == ([1], True))
_rendered = render_selection(_hdr, _turns, [1], "RECENT")
check("exp24 renderer copies exact bytes", _rendered == _hdr + _turns[1] + "RECENT")
check("exp24 task split deterministic", stable_split("task-x") == stable_split("task-x"))
from exp24_grpo_train import ExecutorReward
_exp24_log = REPO / "tests" / ".exp24_reward_test.jsonl"
_exp24_log.unlink(missing_ok=True)
_executor_reward = ExecutorReward("http://127.0.0.1:9", "unused", 1, 8, _exp24_log)
check("exp24 reward callable exposes TRL name", _executor_reward.__name__ ==
      "executor_behavior_reward")
_bad_reward = _executor_reward(
    prompts=["p"], completions=["bad"], header=["h"],
    units_json=['["u"]'], recent_text=["r"], logged_action=["bash::x"],
    budget_chars=[1], task=["t"], id=[0])
check("exp24 invalid completion needs no executor", _bad_reward[0] < 0)
_exp24_log.unlink(missing_ok=True)
# vLLM otherwise silently imports model generation_config (Qwen top_k=20,
# top_p=.95), invalidating the declared temp=1/top_p=1 protocol.
check("TB2 disables model generation defaults", "--generation-config vllm" in
      (REPO / "tb2/eval_tb2.sh").read_text())
_exp24_job_text = (REPO / "experiments/exp24_job.sh").read_text()
check("exp24 disables model generation defaults", "--generation-config vllm" in
      _exp24_job_text)
check("exp24 final executor is Qwen3.8-27B",
      "Qwen/Qwen3.8-27B" in _exp24_job_text and "exp24_qwen38_data" in _exp24_job_text)
check("exp24 uses valid Trillium whole-node request",
      "#SBATCH --gpus-per-node=h100:4" in _exp24_job_text)
check("exp24 executor uses working Alliance Python 3.12/opencv stack",
      "module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0" in
      _exp24_job_text and "module purge" in _exp24_job_text)
check("exp24 selector uses complete clean-node Alliance stack",
      "python-build-bundle/2026a" in _exp24_job_text and
      "scipy-stack/2026a" in _exp24_job_text)
_exp24_trainer_text = (REPO / "experiments/exp24_grpo_train.py").read_text()
_exp24_selector_gate = (REPO / "experiments/exp24_selector_preflight.sh").read_text()
check("exp24 has one-GPU selector construction gate",
      "--preflight-only" in _exp24_trainer_text and bool(_exp24_selector_gate))
check("selector gate loads hierarchical StdEnv separately",
      "module load StdEnv/2023\nmodule load gcc" in _exp24_selector_gate and
      "module list" in _exp24_selector_gate)
check("exp24 preserves Alliance sitecustomize PYTHONPATH",
      'PYTHONPATH=$ROOT:${PYTHONPATH:-}' in _exp24_selector_gate and
      'PYTHONPATH=$ROOT:${PYTHONPATH:-}' in _exp24_job_text)
check("exp24 trainer has lineage guard", "OFF-POLICY DATA REFUSED" in
      _exp24_trainer_text)
check("exp24 reward uses native chat interface",
      "/v1/chat/completions" in _exp24_trainer_text and
      '"reasoning_effort": "low"' in _exp24_trainer_text)
check("exp24 reward rejects raw completion proxy",
      'url.rstrip("/") + "/v1/completions"' not in _exp24_trainer_text)
_exp24_entry = subprocess.run(
    [sys.executable, str(REPO / "experiments/exp24_prefetch_onpolicy.py"), "--help"],
    cwd="/tmp", capture_output=True, text=True)
check("exp24 prefetch standalone entrypoint", _exp24_entry.returncode == 0,
      _exp24_entry.stderr[-300:])
_q38_preflight = (REPO / "tb2/preflight_qwen38_27b.sh").read_text()
check("Qwen3.8 preflight runs real Harbor",
      "bash tb2/eval_tb2.sh" in _q38_preflight)
check("Qwen3.8 preflight requires Harbor-parsed commands",
      "tool_calls" in _q38_preflight and "actions > 0" in _q38_preflight)
check("Qwen3.8 preflight requires executed verifier",
      "test-stdout.txt" in _q38_preflight and "pytest did not execute" in _q38_preflight)
check("Qwen3.8 preflight rejects trial exceptions", "exception.txt" in _q38_preflight)
_eval_tb2 = (REPO / "tb2/eval_tb2.sh").read_text()
check("Qwen3.8 uses native low reasoning effort",
      "reasoning_effort: low" in _eval_tb2)
check("Qwen3.8 request timeout fits output budget", "timeout: 1800" in _eval_tb2)
check("Qwen3.8 preflight does not use raw completion proxy",
      "/v1/completions" not in _q38_preflight and
      "from behavior import parse_action" not in _q38_preflight)

print(f"\n{PASS} checks passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
print("ALL CLEAR")
