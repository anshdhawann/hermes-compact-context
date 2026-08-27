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
  threshold_tokens = context_length × threshold_percent (default 0.20, configurable via compact-context.threshold_percent)
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
5. **Assembly** — `[system+note] [head] [summary message] [tail] [last user message]`. The summary role is chosen to avoid consecutive same-role messages; if the message after the summary would repeat the summary's role, a synthetic boundary marker is inserted with the OPPOSITE role of the summary (not hardcoded — a hardcoded `user` marker produced three consecutive user messages whenever the summary itself was user-role). Orphaned `tool` messages (no active `tool_call_id`) are filtered.
6. **Metadata** — the summary message is marked `_compressed_summary: true` (underscore keys are stripped by wire sanitizers before reaching the API).

## Why role alternation matters

OpenAI-format backends reject adjacent messages with the same role. The summary message's role must differ from both its predecessor (last head message) and successor (first tail message or the appended last user message). The engine resolves this with the summary-role flip; when the successor would still repeat the summary's role, a synthetic boundary marker is inserted with the opposite role of the summary.

## Token accounting

- `update_model()` recomputes `threshold_tokens = context_length × threshold_percent` on every model switch — `threshold_percent` is config-readable (`compact-context.threshold_percent`, default 0.20), so the percent is the single governing knob.
- `should_compress()` compares `last_prompt_tokens` (from provider usage). If the provider omits `prompt_tokens`, compression never fires.
- `compression.threshold` in config.yaml governs ONLY the built-in compressor, not this engine.

## Testing

`tests/test_compact_engine.py` runs without network (the summarizer LLM is stubbed). It verifies: transcript written with all messages, pointer injected, tail preserved, head preserved with no body leak, final user message present, 10-section prompt + focus topic + truncation marker, summarizer call config, strict role alternation, and a regression case for the boundary marker (tool-ended head → user-role summary + user-started tail → marker must flip to `assistant`).

```bash
python3 tests/test_compact_engine.py
# REPO path: defaults to ~/.hermes/hermes-agent, override with HERMES_REPO
```

## Hardening (v2.1)
- `threshold_percent` is config-driven (was class-attr only).
- `transcript_retain` keeps the N most recent transcript files (default 2) to bound disk usage.
- Post-summary secret scrub is code-enforced (sk-*, sbp_*, gho_*, Bearer, private keys, key=value) — not just a prompt instruction.
- Body-too-large guard (80% of context_length) skips a turn when the body itself would overflow the summarizer.
- `should_compress` falls back to `estimate_messages_tokens_rough` when the provider omits `prompt_tokens`; backoff after 3 consecutive failures.
- Prompt delimiters are escaped in message content so `---END---` cannot break the prompt structure.

## Provenance notes

- ZCode behavior observed in its shipped application (v3.7.6, Aug 2026) and its on-disk state (`~/.zcode/cli/agents/<session>/<agent>/transcript.jsonl`, `~/.zcode/cli/memories/projects/<project>/`).
- VERIFIED against live ZCode artifacts (2026-08-26: a 2026-07-03 compaction summary recovered from ZCode's own session database, plus a live continued session): full-rewrite summary, numbered-section format, delivery as a **user-role message**, preserved recent tail, and an append-only store that never deletes old turns — all confirmed.
- Two deliberate Hermes-specific divergences: **(1)** ZCode does NOT inject a transcript pointer into the summary — the pointer here is our extension, the Hermes-native way to give the model re-read access (ZCode's own store serves that role internally). **(2)** ZCode never prunes its store; our `transcript_retain: 2` pruning is safe because Hermes' `in_place` soft-archive already retains every pre-compaction turn in the session database.
- ZCode's section list (Problem Solving / All user messages / Pending Tasks) differs from ours in 3 of 9 slots; v2.2 adopts ZCode's "All user messages" verbatim-voice section.
- ZCode is closed-source and evolving; its compaction may change between versions. This engine replicates the design as of the version noted above.
