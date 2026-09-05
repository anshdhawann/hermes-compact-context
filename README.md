# hermes-compact-context

ZCode-style aggressive context compression for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) — a drop-in context engine that replaces the built-in conservative compressor with a full-rewrite compaction: **one ~7K structured summary for the entire conversation, instead of incremental trimming.**

## Why

The built-in Hermes compressor protects the last ~20 messages verbatim and only summarizes the middle — on a 284K-token session it goes 284K → ~280K. This engine goes **284K → ~7K** while keeping everything retrievable.

## How it works

1. **Full rewrite** — the whole conversation is summarized in one pass by a dedicated large-context model. No protected middle, no iterative re-compression.
2. **10-section handoff summary** — chronological analysis over every message: requests & intent, technical concepts, files & full code snippets, errors & fixes, user preferences, all user messages (verbatim), security constraints (preserved VERBATIM), key decisions, current work, optional next step.
3. **Transcript archive + pointer** — the full pre-compaction conversation is written to JSONL on disk and the path is injected into the summary, so the model can re-read exact details on demand. *Context shrinks; information does not disappear.*
4. **Verbatim tail** — the last N messages stay untouched.
5. **Resume semantics** — the model picks up the last task "as if the break never happened."
6. **Invisible summary** — the summary row is persisted `display_kind="hidden"`: the model sees it in context, every transcript surface renders nothing (ZCode-style: main thread stays clean, full chat lives in the archive).

## Install

```bash
bash install.sh   # copies the plugin to ~/.hermes/plugins/compact-context

hermes config set context.engine compact-context
# ⚠️ plugins.enabled must be set as a full YAML list — NEVER use '+compact-context'.
# The '+name' syntax replaces the whole list with a string, which silently
# disables EVERY plugin (the loader only accepts a list). Keep any plugins
# you already have enabled:
hermes config set plugins.enabled '["compact-context"]'
# ⚠️ REQUIRED: persist compacted messages in the SAME session. Without this,
# Hermes rotation mode spawns a child session that reloads the FULL parent
# history, re-triggers compression, and loops forever ("No changes from
# compression" every turn, tokens never shrink):
hermes config set compression.in_place true
# /reset to activate
```

## Configuration

```yaml
compact-context:
  target_tokens: 7000       # summary size
  preserve_first_n: 3       # head messages kept verbatim
  preserve_last_n: 6        # tail messages kept verbatim
  threshold_percent: 0.20   # trigger at 20% of context window (~200K on 1M model)
  threshold_tokens: 0       # fixed trigger in tokens; set BOTH knobs to fire at min(percent×window, fixed)
  transcript_enabled: true  # archive + pointer injection
  transcript_retain: 2      # newest N full archives + consolidated root; old paths become redirects (0 = keep all)
  model: zai/GLM-5.2        # dedicated summarizer — MUST fit the full conversation (1M window recommended)
  provider: opencode-go
  summary_context_length: 0 # summarizer's window in tokens (0 = auto: Hermes' discovered-length cache)
compression:
  in_place: true            # REQUIRED — see Install
```

**Important:** the summarizer reads the entire conversation in one pass, so its context window must be at least as large as your session. A 1M-context model (e.g. GLM-5.2) is recommended.

**Failure ladder** — retry the dedicated summarizer with the main model, pinning the main model/provider/endpoint/API mode. If the chain fails at ≥95% of the context window and a transcript archive was successfully written, use an archive handoff. Without a verified archive, fail open with the original messages. Below the urgency line, failures back off with periodic probes. Truncated responses (`finish_reason="length"`) count as failures.

**Output budget** — measure the complete output with Hermes' token estimator and reserve response headroom. With a verified archive, trim recent history as needed; if the preserved non-system head is still too large, move it to the archive too. An oversized latest request is shortened with a pointer to its full original text. System and developer instructions are never truncated: if those plus a minimal archive handoff cannot fit, compression raises a clear error before Hermes persists a result. With no verified archive or usable estimate, further destructive trimming is refused and the original messages are returned. These checks use a rough estimate, not a guarantee of the provider's exact token count.

**Archive retrieval** — retention keeps the newest N full transcripts plus a consolidated root. Older paths become small JSON redirect files containing `_consolidated_into` and instructions to read the root, so existing summary pointers stay usable. Session identifiers are hashed in full; different sessions sharing a directory cannot cross-prune. Archives created by older plugin versions are left untouched because their ambiguous filename prefixes cannot be safely assigned to a session. Retention bounds full snapshot count, not total bytes: original history and redirect pointers remain available.

The summarizer guard measures the actual formatted request plus its output reserve. Tool output is shortened in that prompt; the full output remains in the archive. A request too large for every known summarizer window is skipped below the urgency line and routed to archive rescue when urgent. A post-compaction warning reports when the result still exceeds the configured trigger even though it fits the model budget.


Trigger tuning: `threshold_tokens` alone = fixed mode (fires at exactly N tokens); `threshold_percent` alone = relative (default 0.20). Set **both** and they compose as **min()** — e.g. `threshold_percent: 0.8` + `threshold_tokens: 200000` fires at min(80% of window, 200K): ride the window on small models, but never past 200K on big ones. min() is always safely below the window (the percent is validated to 0.05–0.95), so it can never trip the overflow guard. A fixed-only value ≥ 95% of the context window can never fire in time, so it's ignored with a warning and the percent rule is used instead — re-checked on every model switch, since the window can change. The built-in `compression.threshold` config does NOT govern plugin engines.

## How it compares

| Tool | Mechanism | Transcript archive + pointer |
|---|---|---|
| **hermes-compact-context** | full rewrite → ~7K | ✅ |
| Hermes built-in compressor | conservative middle-summarize | ❌ |
| Claude Code auto-compact | paging + selective clearing + summarize | ❌ |
| Codex CLI /compact | checkpoint handoff summary | ❌ |
| OpenCode | overflow summarization + tool-output pruning | ❌ |
| ZCode /compact | full rewrite + transcript archive | ✅ (closed-source) |

## Attribution

Inspired by the compaction design of [ZCode](https://zcode.z.ai) (Z.AI) — the best long-session compaction I've used; sessions effectively last forever. This project is an independent, original implementation. It is **not affiliated with, endorsed by, or sponsored by Z.AI / ZCode.**

See [docs/architecture.md](docs/architecture.md) for the design details and provenance.

## License

MIT
