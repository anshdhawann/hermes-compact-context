# AGENTS.md

Ponytail rules for this repo: **the laziest solution that actually works.**
(Adapted from the `ponytail` skill, MIT.)

The engine is one file, zero runtime deps, one assert-based test script. That
is the design, not an accident — preserve it.

1. **YAGNI.** Speculative features get skipped in one line, not built.
2. **Reuse what's already in `__init__.py`** before adding anything. Stdlib
   before dependencies — zero runtime deps is a feature.
3. **Shortest working diff**, root cause not symptom: fix it once where all
   callers route through.
4. Mark deliberate corners with a `# ponytail:` comment naming the ceiling
   and the upgrade path.
5. `python3 tests/test_compact_engine.py` (offline, no frameworks, no Hermes
   install required) must pass before every push — the script IS the CI. One
   runnable check per fixed bug, added to the same script.
5a. Tests drive the REAL entry points (`_load_config`, `compress`), never set
   engine internals directly — a v2.4.0 bug survived because tests pinned
   `_explicit_percent` instead of loading config. Tests never read the
   developer's real `~/.hermes/config.yaml` — the suite installs a fake
   `load_config`; ambient config once masked a bug.
5b. Protocol invariants (role alternation, tool-call/tool pairing) are
   asserted on the FULL assembled output of adversarial inputs (tool-dense
   tails, boundaries cutting a tool transaction), not just handcrafted happy
   paths. Consecutive `tool` results are legal; user/user and
   assistant/assistant are not.
5c. Message membership is POSITIONAL (index), never dict-equality (`in`) —
   two turns can carry identical content ("continue") and both must survive;
   equality-based dedup silently dropped the later one (v2.5.1). Every
   fail-open `return messages` must be weighed against the ≥95%-of-window
   rescue: skipping is only safe BELOW the urgency line — the v2.5.0 guard
   bypassed the rescue precisely when it was needed. Rescue tests must make
   the BODY itself trigger the condition (oversized/unreadable), not just
   pin a huge `current_tokens` on a tiny conversation.
6. Behavior change → bump `version:` in `plugin.yaml` in the same commit.
7. **Never simplify away:** graceful degradation (LLM failure → messages
   unchanged), the secret scrub, strict role alternation, and the failure
   ladder (main-model fallback → mechanical rescue at ≥95% of the window →
   fail open with backoff). These exist because each one was a live incident
   or a reviewed near-miss; the rescue especially must survive refactors —
   it is the only path that prevents a dead summarizer from killing the
   session to context overflow.
