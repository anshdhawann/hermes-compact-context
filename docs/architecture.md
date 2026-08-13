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
  threshold_tokens = context_length × threshold_percent (default 0.20)
  → compress(messages, current_tokens, focus_topic)
session end → on_session_end(session_id, messages)
```

The engine swaps in via config: `context.engine: compact-context`. Hermes discovers it through the plugin system (`register()` → `ctx.register_context_engine`).

## Compress flow

1. **Boundaries** — `head` = system prompt + first `preserve_first_n` non-system messages; `tail` = last `preserve_last_n` messages; `body` = everything between.
2. **Transcript archive** — the FULL message list (head + body + tail, tool outputs included) is written as JSONL to `~/.hermes/sessions/<id>/compaction_transcript_<ts>.jsonl` (or `compact-context.transcript_dir` override). The path is injected into the summary message.
3. **Summarization prompt** — body messages are formatted as a dense text transcript (tool outputs >4K chars truncated with a marker pointing at the on-disk transcript). The prompt requests an `<analysis>` + `<summary>` reply with 9 sections. `focus_topic` (from `/compress [topic]`) is forwarded and prioritized.
4. **Summary call** — routed via `call_llm(task="compression")`; a dedicated summarizer model (config `compact-context.model`/`provider`) overrides the main runtime when set. `max_tokens = target_tokens × 1.5`. On any failure the engine returns messages unchanged (graceful degradation).
5. **Assembly** — `[system+note] [head] [summary message] [tail] [last user message]`. The summary role is chosen to avoid consecutive same-role messages; a synthetic user boundary marker is inserted if needed. Orphaned `tool` messages (no active `tool_call_id`) are filtered.
6. **Metadata** — the summary message is marked `_compressed_summary: true` (underscore keys are stripped by wire sanitizers before reaching the API).

## Why role alternation matters

OpenAI-format backends reject adjacent messages with the same role. The summary message's role must differ from both its predecessor (last head message) and successor (first tail message or the appended last user message). The engine resolves this with the summary-role flip plus a synthetic user boundary message.

## Token accounting

- `update_model()` recomputes `threshold_tokens = context_length × threshold_percent` on every model switch — the percent is the single governing knob.
- `should_compress()` compares `last_prompt_tokens` (from provider usage). If the provider omits `prompt_tokens`, compression never fires.
- `compression.threshold` in config.yaml governs ONLY the built-in compressor, not this engine.

## Testing

`tests/test_compact_engine.py` runs without network (the summarizer LLM is stubbed). It verifies: transcript written with all messages, pointer injected, tail preserved, head preserved with no body leak, final user message present, 9-section prompt + focus topic + truncation marker, summarizer call config, strict role alternation.

```bash
python3 tests/test_compact_engine.py
# REPO path: defaults to ~/.hermes/hermes-agent, override with HERMES_REPO
```

## Provenance notes

- ZCode behavior observed in its shipped application (v3.7.6, Aug 2026) and its on-disk state (`~/.zcode/cli/agents/<session>/<agent>/transcript.jsonl`, `~/.zcode/cli/memories/projects/<project>/`).
- Observed-but-unverified: live end-to-end capture of a post-compaction context (pointer line present in the app, not captured from a live session on record).
- ZCode is closed-source and evolving; its compaction may change between versions. This engine replicates the design as of the version noted above.
