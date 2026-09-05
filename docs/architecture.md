# Architecture

How `hermes-compact-context` works, and where the design comes from.

## Design origin

The design is inspired by the compaction behavior of [ZCode](https://zcode.z.ai) (Z.AI), observed in its shipped application (v3.7.x, Aug 2026). ZCode's `/compact` "never forgets": the full conversation is archived to disk, a pointer to the transcript is injected into the compacted context, an exhaustive handoff summary is generated, and recent messages stay verbatim. This engine is an independent, original implementation of that design for Hermes Agent's context-engine plugin interface. No ZCode code is included; the summary prompt is original wording with the same section structure.

## Lifecycle

```
session start → on_session_start(session_id)
after each LLM call → update_from_response(usage)   (tracks prompt/completion tokens)
each turn → should_compress(prompt_tokens)
  fires when last_prompt_tokens >= threshold_tokens
  threshold_tokens = min(threshold_tokens_cfg, context_length × threshold_percent)  when both are explicitly configured
                  | threshold_tokens_cfg (fixed, compact-context.threshold_tokens)
                  | context_length × threshold_percent (default 0.20, compact-context.threshold_percent)
  → compress(messages, current_tokens, focus_topic)
session end → on_session_end(session_id, messages)
```

The engine swaps in via config: `context.engine: compact-context`. Hermes discovers it through the plugin system (`register()` → `ctx.register_context_engine`).

## Persistence requirement: compression.in_place

The host persists the compacted message list ONLY when `compression.in_place: true`. With the default (`false`, rotation mode), Hermes archives the session and spawns a child that reloads the FULL parent history via the parent chain — the compaction is never saved, the child re-triggers compression, and the session loops re-summarizing the same conversation forever (observed live: a 414-message / 429K-token session compacted to 5 messages, then the child came back at 414 and did it again). **`hermes config set compression.in_place true` is a required install step**, not a tuning option.

## Compress flow

1. **Boundaries** — `head` = system prompt + first `preserve_first_n` non-system messages; `tail` = last `preserve_last_n` messages; `body` = everything between.
2. **Transcript archive** — the FULL message list (head + body + tail, tool outputs included) is written as JSONL to `~/.hermes/sessions/<id>/compaction_transcript_<ts>.jsonl` (or `compact-context.transcript_dir` override). The path is injected into the summary message.
3. **Summarization prompt** — body messages are formatted as a dense text transcript (tool outputs >4K chars truncated with a marker pointing at the on-disk transcript). The prompt requests an `<analysis>` + `<summary>` reply with 10 sections (ZCode's nine minus its Problem Solving/Pending Tasks, plus User Preferences, Security and Constraints, Key Decisions — and, since v2.2, ZCode's "All user messages" so the user's voice survives verbatim). `focus_topic` (from `/compress [topic]`) is forwarded and prioritized.
4. **Summary call** — routed via `call_llm(task="compression")`; a dedicated summarizer model (config `compact-context.model`/`provider`) overrides the main runtime when set. `max_tokens = target_tokens × 1.5`. On any failure the engine returns messages unchanged (graceful degradation).
5. **Boundary repair (before assembly)** — head+tail run through the bidirectional tool-pair sanitizer FIRST: tools whose calling assistant was summarized into the body are dropped, and assistants whose tool results fell into the body lose those `tool_calls` (either shape is an instant 400). This happens before the marker decision so a deleted orphan can never invalidate it.
6. **Assembly** — `[system+note] [head] [summary message] [tail] [last user message?]`. The summary role is chosen to avoid consecutive same-role messages; if the message after the summary would repeat the summary's role, a synthetic boundary marker is inserted with the OPPOSITE role of the summary (not hardcoded — a hardcoded `user` marker produced three consecutive user messages whenever the summary itself was user-role). The last user message is appended only when it repairs the list (dangling trailing tool / empty tail) or the session genuinely ends on it — after an assistant's answer it would read as a fresh re-ask of a completed task.
7. **Metadata** — the summary message is marked `_compressed_summary: true` (underscore keys are stripped by wire sanitizers before reaching the API).

## Why role alternation matters

OpenAI-format backends reject adjacent messages with the same role. The summary message's role must differ from both its predecessor (last head message) and successor (first tail message or the appended last user message). The engine resolves this with the summary-role flip; when the successor would still repeat the summary's role, a synthetic boundary marker is inserted with the opposite role of the summary.

## Token accounting

- Threshold resolution lives in `_recompute_threshold()`, run at init, config load, and every `update_model()`: a fixed `threshold_tokens` (> 0, from `compact-context.threshold_tokens`) overrides the percent rule; when **both** are explicitly configured they compose as **min(fixed, percent×window)** — an explicit percent is one that was present and valid (0.05–0.95) in config, tracked via `_explicit_percent`, so a fixed-only setup is never capped by the default percent. min() is always under 95% of the window, so it can't trip the overflow guard; a fixed-only value ≥ 95% of the window can never fire before overflow, so it falls back to percent with a warning. Re-checked per model switch because the window can change under a fixed value.
- `should_compress()` compares `last_prompt_tokens` (from provider usage). If the provider omits `prompt_tokens`, compression never fires.
- `compression.threshold` in config.yaml governs ONLY the built-in compressor, not this engine.

## Testing

`tests/test_compact_engine.py` runs without network (the summarizer LLM is stubbed) and without a Hermes install (`agent.*` stubs installed when absent; `hermes_cli.load_config` overridden with a fake so tests never read the developer's real config — an ambient-config leak once masked a threshold bug). It verifies: transcript written with all messages, pointer injected, tail preserved, head preserved with no body leak, final user message present, 10-section prompt + focus topic + truncation marker, summarizer call config, strict role alternation, a regression case for the boundary marker (tool-ended head → user-role summary + user-started tail → marker must flip to `assistant`), the fixed/percent/min() threshold modes through the real config-load path, orphaned-tool tails, unanswered tool_calls at the head boundary, private-key block scrubbing, backoff probing, tool-argument formatting, and the no-stale-re-ask rule after an answered tail.

```bash
python3 tests/test_compact_engine.py
# REPO path: defaults to ~/.hermes/hermes-agent, override with HERMES_REPO
```

## Hardening (v2.1)
- `threshold_percent` is config-driven (was class-attr only).
- `transcript_retain` keeps the N most recent transcript files (default 2) to bound disk usage.
- Post-summary secret scrub is code-enforced (sk-*, sbp_*, gho_, Bearer, private keys, key=value) — not just a prompt instruction.
- Body-too-large guard (80% of context_length) skips a turn when the body itself would overflow the summarizer.

## Hardening (v2.4.1 — external review)
- "Explicit" threshold percent now requires the config key to be PRESENT (a valid-range default used to mark it explicit and silently cap fixed-only thresholds).
- Tool-pair repair is bidirectional and runs on head+tail BEFORE the marker decision: orphaned tools can no longer be deleted after influencing it (left user,user adjacency), and head assistants lose calls whose results were summarized away.
- Last-user-message append is conditional: repairs dangling-tool endings and normal user endings; skipped after an assistant answer (was a stale re-ask of a finished task).
- Private-key scrub replaces whole blocks (header through footer, any key type; truncated blocks still lose the header).
- Backoff after summarizer failures probes every 5th turn instead of locking compaction off for the session; `update_model()` resets the failure count.
- Summarizer transcript includes tool-call argument previews (file paths/commands), not just tool names.
- `should_compress` falls back to `estimate_messages_tokens_rough` when the provider omits `prompt_tokens`; backoff after 3 consecutive failures.
- Prompt delimiters are escaped in message content so `---END---` cannot break the prompt structure.

## Failure ladder (v2.5.0 — Codex-inspired, cross-reviewed)
The summarizer call is now a candidate CHAIN, and the failure mode degrades in stages instead of losing the session:

1. **Attempt order:** dedicated summarizer (when configured and its known window can hold the body in one pass), then the MAIN model via `main_runtime`. A summarizer whose known window is too small is skipped outright (logged); an unknown window (0) is always attempted.
2. **Body guard is chain-aware:** skip only when the body exceeds 80% of the BEST window in the chain — a 1M summarizer on a small-window main relaxes the guard that previously (and wrongly) used the main window alone; a small summarizer no longer receives doomed one-pass bodies. Window resolution: explicit `summary_context_length` config → Hermes' discovered-length cache (`get_cached_context_length`, a pure disk read — never a network probe in the hot path) → unknown.
3. **Mechanical rescue:** if the whole chain fails while the session sits at ≥95% of the window, compact ANYWAY — head + stub summary pointing at the already-archived transcript + tail. Failing open at that point means the next main call 400s and the session dies; the transcript is written before the summarizer is ever called, so the rescue always has material. The rescue still counts as an LLM failure for backoff.
4. **Urgency punch-through:** `should_compress` never lets backoff suppress a ≥95%-of-window turn — that turn's `compress()` is the only place the fallback and rescue exist.
5. **Fail open (unchanged):** below the urgency line, a failed chain returns messages unchanged, increments backoff, probes every 5th turn.
6. **Post-compact floor warning:** if the assembled output still sits at/above the fire threshold, warn immediately — the un-trimmable floor (system prompt + preserved head/tail + summary) exceeds the trigger and compaction would re-fire every turn. This is the 2026-09-01 production incident (threshold 49,152 below a ~92K system-prompt floor) made self-announcing: it now surfaces on the first compaction instead of from log forensics.

Design cross-check: the rescue shape (shrink-and-retry near overflow instead of fail-open) and the aux→main fallback are patterned on Codex's overflow handling; Codex's `BodyAfterPrefix` trigger scoping and recompact-on-model-change were evaluated and rejected (our threshold recomputes on `update_model()` already, and the floor problem is guarded by the warning above).

## Fixes (v2.5.1 — Astra review)
Five verified findings, each with a regression check ([26]–[30]):
1. **The overflow guard bypassed the rescue.** A body exceeding 80% of every candidate window returned messages unchanged even at 96% session usage — the mechanical rescue was unreachable exactly when needed. The guard now routes oversized bodies into the rescue at ≥95% of the window (and does not count it as an LLM failure — no call was attempted); below the urgency line it still skips.
2. **Repeated messages silently disappeared.** The tail dedup (`tm not in head`) compared dict CONTENT, not position — a recent "continue" whose early twin sat in the head was dropped (breaking alternation and losing a live instruction). Head/tail are disjoint slices by construction, so the dedup is simply gone; last-user-message membership is now positional (index-based), never equality-based.
3. **`preserve_last_n: 0` skipped tool-pair repair.** Sanitization ran only under `if tail:` — a head assistant whose tool results were summarized away kept unanswered `tool_calls` (protocol 400). The head is now sanitized on its own when the tail is empty.
4. **PKCS#8 keys survived redaction.** The regex required a type word (`RSA`/`EC`/...) before `PRIVATE KEY`, so plain `-----BEGIN PRIVATE KEY-----` passed intact, and truncated typed blocks kept their base64 body after the header was removed. The pattern now makes the type optional in BEGIN and END, and an unterminated block redacts through to the next `-----BEGIN` line or end of input (over-redaction is safe; leakage is not).
5. **Transcript filenames could collide.** One-second timestamps opened in `w` mode: two writes within a second (or concurrent sessions sharing the fallback dir) silently overwrote an archive. Writes now use `tempfile.mkstemp` (timestamp prefix + unique suffix, exclusive create).

## Provenance notes

- ZCode behavior observed in its shipped application (v3.7.6, Aug 2026) and its on-disk state (`~/.zcode/cli/agents/<session>/<agent>/transcript.jsonl`, `~/.zcode/cli/memories/projects/<project>/`).
- VERIFIED against live ZCode artifacts (2026-08-26: a 2026-07-03 compaction summary recovered from ZCode's own session database, plus a live continued session): full-rewrite summary, numbered-section format, delivery as a **user-role message**, preserved recent tail, and an append-only store that never deletes old turns — all confirmed.
- Two deliberate Hermes-specific divergences: **(1)** ZCode does NOT inject a transcript pointer into the summary — the pointer here is our extension, the Hermes-native way to give the model re-read access (ZCode's own store serves that role internally). **(2)** ZCode never prunes its store; our `transcript_retain: 2` pruning is safe because Hermes' `in_place` soft-archive already retains every pre-compaction turn in the session database.
- ZCode's section list (Problem Solving / All user messages / Pending Tasks) differs from ours in 3 of 9 slots; v2.2 adopts ZCode's "All user messages" verbatim-voice section.
- ZCode is closed-source and evolving; its compaction may change between versions. This engine replicates the design as of the version noted above.
