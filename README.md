# hermes-compact-context

ZCode-style aggressive context compression for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) — a drop-in context engine that replaces the built-in conservative compressor with a full-rewrite compaction: **one ~7K structured summary for the entire conversation, instead of incremental trimming.**

## Why

The built-in Hermes compressor protects the last ~20 messages verbatim and only summarizes the middle — on a 284K-token session it goes 284K → ~280K. This engine goes **284K → ~7K** while keeping everything retrievable.

## How it works

1. **Full rewrite** — the whole conversation is summarized in one pass by a dedicated large-context model. No protected middle, no iterative re-compression.
2. **10-section handoff summary** — chronological analysis over every message: requests & intent, technical concepts, files & full code snippets, errors & fixes, user preferences, security constraints (preserved VERBATIM), key decisions, current work, optional next step.
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
  transcript_enabled: true  # archive + pointer injection
  transcript_retain: 2      # keep N most recent transcript files (0 = keep all)
  model: zai/GLM-5.2        # dedicated summarizer — MUST fit the full conversation (1M window recommended)
  provider: opencode-go
compression:
  in_place: true            # REQUIRED — see Install
```

**Important:** the summarizer reads the entire conversation in one pass, so its context window must be at least as large as your session. A 1M-context model (e.g. GLM-5.2) is recommended; if the summary call fails, the engine keeps messages unchanged and logs a warning.

Trigger tuning: `compact-context.threshold_percent` (default 0.20 → fires at ~200K on a 1M-window model). The built-in `compression.threshold` config does NOT govern plugin engines.

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
