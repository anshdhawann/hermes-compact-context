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
5. `python3 tests/test_compact_engine.py` (offline, no frameworks) must pass
   before every push — the script IS the CI. One runnable check per fixed bug,
   added to the same script.
6. Behavior change → bump `version:` in `plugin.yaml` in the same commit.
7. **Never simplify away:** graceful degradation (LLM failure → messages
   unchanged), the secret scrub, strict role alternation. These exist because
   each one was a live incident.
