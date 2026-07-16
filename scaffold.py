"""The recovery test (the honest control behind `--scaffold`).

A person with a compressed memory doesn't fail \u2014 they look something up. An
agent can too. So a *different* next action after compression might be the
agent adaptively going to re-read, not damage. This module implements that
test:

1. Offer the compressed agent an explicit tool menu that includes lookups
   (`read_file`, `grep`) alongside task actions (`edit`, `run_tests`,
   `submit`).
2. If it chooses a lookup, hand back the *relevant slice of the original,
   pre-compression history* (the exact thing compression threw away) as an
   observation, then let it continue.
3. Repeat up to `max_lookups` times, then return the final action it commits.

If the compressed agent recovers to the same action the full-context agent
took, that compression was fine. If it can't \u2014 even with lookups \u2014 the
information loss was real.

One rollout at a time (each sampled continuation branches differently after a
lookup, so they can't be batched). Small models / small N make this fine.
"""

from __future__ import annotations

from behavior import action_kind, parse_action, sample_texts

# The menu shown to the agent after a compressed context. The tool names line
# up with behavior.MENU_KIND so parse_action / action_kind classify them.
MENU_INSTRUCTIONS = (
    "\n\n[Earlier history was compacted. Continue by calling exactly ONE tool:\n"
    "  read_file(\"path\")   \u2014 re-read a file from the earlier work\n"
    "  grep(\"pattern\")      \u2014 search the earlier work for something\n"
    "  run_tests()           \u2014 run the test suite\n"
    "  edit(\"path\")         \u2014 modify a file to make progress\n"
    "  submit()              \u2014 finish the task\n"
    "Use read_file/grep if you need information that was compacted away; "
    "otherwise take the task action. Call one tool now.]\n"
)


def _ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def serve_lookup(action: str, old_text: str, *, budget_chars: int = 800) -> str:
    """Hand back the relevant slice of the original (pre-compression) history
    for a lookup action like `read_file::src/foo.py` or `grep::TypeError`.

    We literally search the dropped history and return the matching windows \u2014
    this is the information the agent would have had without compression.
    """
    name, _, arg = action.partition("::")
    arg = arg.strip().strip("'\"() ")
    if not arg:
        return "<observation>\n[nothing specified to look up]\n</observation>"

    low = old_text.lower()
    needles = [arg.lower()]
    if "/" in arg:                       # for a path, also try the basename
        needles.append(arg.rsplit("/", 1)[-1].lower())

    positions: list[int] = []
    for nl in needles:
        if not nl:
            continue
        start = 0
        while len(positions) < 3:
            i = low.find(nl, start)
            if i == -1:
                break
            positions.append(i)
            start = i + max(1, len(nl))
        if positions:
            break

    if not positions:
        return f"<observation>\n[no earlier record of '{arg}' found]\n</observation>"

    chunks = []
    for i in sorted(set(positions)):
        s, e = max(0, i - 200), min(len(old_text), i + 400)
        chunks.append(old_text[s:e].strip())
    body = "\n...\n".join(chunks)[:budget_chars]
    return f"<observation>\n{body}\n</observation>"


def recover_action(model, tokenizer, compressed_ids, old_text, device, *,
                   max_lookups: int = 2, max_new: int = 768,
                   temperature: float = 0.7, seed: int = 0):
    """Run one recovery rollout. Returns (final_action, used_lookup).

    `final_action` is the first non-lookup action the agent commits to (or the
    last action if it exhausts its lookup budget, or None if it never acts).
    `used_lookup` is True if it consulted the original history at least once.
    """
    menu_ids = _ids(tokenizer, MENU_INSTRUCTIONS)
    ctx = list(compressed_ids) + menu_ids
    used_lookup = False
    action = None

    for step in range(max_lookups + 1):
        text = sample_texts(model, tokenizer, ctx, device, samples=1,
                            max_new=max_new, temperature=temperature,
                            seed=seed + step)[0]
        action = parse_action(text)
        # A lookup (and budget remaining) -> serve history and continue.
        if action_kind(action) == "lookup" and step < max_lookups:
            used_lookup = True
            obs = serve_lookup(action, old_text)
            ctx = ctx + _ids(tokenizer, text) + _ids(tokenizer, obs) + menu_ids
            continue
        break

    return action, used_lookup


def recover_actions(model, tokenizer, compressed_ids, old_text, device, *,
                    samples: int = 8, max_lookups: int = 2, max_new: int = 768,
                    temperature: float = 0.7, seed: int = 0):
    """Run `samples` independent recovery rollouts.

    Returns (actions, used_lookups): parallel lists of final action labels and
    whether each rollout consulted the original history.
    """
    actions, used = [], []
    for s in range(samples):
        a, u = recover_action(model, tokenizer, compressed_ids, old_text, device,
                              max_lookups=max_lookups, max_new=max_new,
                              temperature=temperature, seed=seed + s * 101)
        actions.append(a)
        used.append(u)
    return actions, used
