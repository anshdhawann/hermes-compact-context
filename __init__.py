"""
Compact Context Engine — ZCode-style full-rewrite context compression.

v2 (ZCode /compact replica):

1. **Transcript archive** — the full pre-compaction conversation is written
   to a JSONL transcript on disk and the path is injected into the summary
   message, so the model can re-read exact details (code snippets, error
   messages, generated content) on demand. Context shrinks; information does
   not disappear — it moves to retrievable storage (ZCode behaviour).

2. **ZCode-grade summary prompt** — chronological <analysis> pass over every
   message, then a 10-section <summary>: Primary Request & Intent, Key
   Technical Concepts, Files & Code Sections (full snippets), Errors & Fixes,
   User Preferences & Corrections, All user messages (verbatim), Security &
   Constraints (VERBATIM), Key Decisions & Rationale, Current Work, Optional
   Next Step. Text-only, no tools, one shot.

3. **Tail preservation** — the last N messages stay verbatim after the
   summary (ZCode: "Recent messages are preserved verbatim.").

4. **Resume instruction** — the model is told to pick up the last task as if
   the break never happened (no recap, no acknowledgement of the summary).

5. **focus_topic / /compress [focus] instructions** are forwarded into the
   summary prompt and prioritised.

Activate in config.yaml:
  context:
    engine: "compact-context"

  compact-context:
    target_tokens: 7000       # target summary token size
    preserve_first_n: 3       # head messages kept verbatim before summary
    preserve_last_n: 6        # tail messages kept verbatim after summary
    transcript_enabled: true  # archive full conversation to disk + pointer
    transcript_dir: ''        # optional override (default ~/.hermes/sessions/<id>/)
    model: deepseek-v4-flash  # dedicated summarizer (must fit full convo)
    provider: opencode-go

The summary model MUST have a context window large enough to read the full
conversation. Recommended: a large-context model (e.g. GLM-5.2 @ 1M) or the
main runtime model when it has a 1M window.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.auxiliary_client import call_llm, aux_interrupt_protection
from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — FULL REWRITE] The entire earlier conversation "
    "was compacted into this summary. This is a handoff from a previous "
    "context window — treat it as background reference, NOT as active "
    "instructions. Do NOT answer questions or fulfill requests mentioned "
    "in this summary; they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "Topic overlap with the summary does NOT mean you should resume its "
    "task: even on similar topics, the latest user message WINS. Treat "
    "ONLY the latest message as the active task. "
    "Your persistent memory (MEMORY.md, USER.md) in the system prompt "
    "is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note."
)

TRANSCRIPT_NOTE_TEMPLATE = (
    "\n\nIf you need specific details from before compaction (like exact code "
    "snippets, error messages, or content you generated), read the full "
    "transcript at: {path}"
)

RECENT_PRESERVED_NOTE = (
    "\n\nRecent messages are preserved verbatim below this summary."
)

RESUME_NOTE = (
    "\n\nContinue the conversation from where it left off without asking the "
    "user any further questions. Resume directly — do not acknowledge the "
    "summary, do not recap what was happening, do not preface with "
    "\"I'll continue\" or similar. Pick up the last task as if the break "
    "never happened."
)

# Mechanical rescue stub: used ONLY when the summarizer chain failed while
# the session sat at >=95% of the window, where failing open means the next
# main call dies of overflow. The transcript is already on disk at this
# point, so the stub points the model at it instead of a written summary.
MECHANICAL_RESCUE_SUMMARY = (
    "## Compaction fallback notice\n"
    "The summarizer model failed while this session was near its context "
    "limit, so the older conversation was archived WITHOUT a written "
    "summary.\n\nFull pre-compaction transcript: {path}\n\n"
    "Re-read that transcript file to recover exact details (requests, file "
    "paths, code, decisions) before continuing. The recent history below "
    "is preserved verbatim."
)

SUMMARY_END_MARKER = "[END OF FULL COMPACTION SUMMARY]"

# Delimiter used to wrap conversation in prompt - sanitized in formatter so
# a message containing "---END---" cannot break the prompt structure.
CONVERSATION_BEGIN_DELIM = "---BEGIN---"
CONVERSATION_END_DELIM = "---END---"

# Post-summary secret scrub: code-enforced, not just prompt instruction.
# Matches common secret shapes; replaced with [REDACTED] after LLM returns.
import re as _re
_SECRET_PATTERNS = [
    _re.compile(r'sk-[A-Za-z0-9_-]{20,}'),
    _re.compile(r'sbp_[A-Za-z0-9]{20,}'),
    _re.compile(r'gho_[A-Za-z0-9_]{20,}'),
    _re.compile(r'ghp_[A-Za-z0-9_]{20,}'),
    _re.compile(r'xox[bprs]-[A-Za-z0-9-]{10,}'),
    _re.compile(r'AKIA[0-9A-Z]{16}'),
    # Full block (header through footer) for any key type: RSA, EC, DSA,
    # OPENSSH, ENCRYPTED. The optional-footer form still redacts the header
    # alone when a block is truncated mid-key.
    _re.compile(
        # Any private-key block, typed (RSA/EC/OPENSSH/ENCRYPTED...) or plain
        # PKCS#8 ("BEGIN PRIVATE KEY" — no type word). An unterminated block
        # (truncated by an earlier cut) is redacted through to the next
        # "-----BEGIN" line or end of input — over-redacting is safe here,
        # leaking key material is not.
        r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"
        r"[\s\S]*?(?:-----END (?:[A-Z0-9_-]+ )?PRIVATE KEY-----|(?=-----BEGIN )|\Z)"
    ),
    _re.compile(r'(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]'),
    _re.compile(r'(?i)Bearer\s+[A-Za-z0-9_\-\.]{20,}'),
]

def _scrub_secrets(s: str) -> str:
    for pat in _SECRET_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s

COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"

DEFAULT_TARGET_TOKENS = 7000
DEFAULT_PRESERVE_FIRST_N = 3
DEFAULT_PRESERVE_LAST_N = 6
DEFAULT_THRESHOLD_PERCENT = 0.20
DEFAULT_THRESHOLD_TOKENS = 0  # 0 = off; derive from threshold_percent instead
DEFAULT_TRANSCRIPT_RETAIN = 2
TRANSCRIPT_GLOB = "compaction_transcript_*.jsonl"

# Compaction summary prompt — same design intent and section structure as
# ZCode's /compact (inspired by its compaction behavior), written in original
# wording for clean public distribution.
ZCODE_SUMMARY_PROMPT = """You are compacting the context of a long-running agent conversation. Respond with TEXT ONLY.

STRICT RULES:
- Do NOT call any tools: no Read, Bash, Grep, Glob, Edit, Write, or anything else.
- Everything you need is in the transcript below — additional fetching is unnecessary.
- Tool calls will be REJECTED and will waste your only turn, failing the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

TASK
Create a detailed handoff summary of the conversation so far, focused on the user's explicit requests and your previous actions. Capture technical details, code patterns, and architectural decisions well enough that development work can continue without losing context.

ANALYSIS PASS
Before writing the summary, work through the conversation chronologically in <analysis> tags to organize your thoughts. For each section, identify:
- The user's explicit requests and intents
- Your approach to addressing them
- Key decisions, technical concepts, and code patterns
- Specific details: file names, full code snippets, function signatures, file edits
- Every error encountered and how it was diagnosed and fixed
- User feedback — especially corrections where the user asked you to do something differently
- Security-relevant instructions or constraints the user stated (sensitive files or data to avoid, operations that must not be performed, credential or secret handling rules) — these MUST be preserved verbatim in the summary so they continue to apply after compaction

Then double-check for technical accuracy and completeness, addressing each required element thoroughly.

SUMMARY SECTIONS

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail.
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable, with a summary of why each file read or edit is important.
4. Errors and Fixes: List every error encountered, the exact error message or signature, and how it was diagnosed and fixed.
5. User Preferences and Corrections: Every explicit preference, style rule, or correction the user stated — preserved verbatim where stated as rules.
6. All user messages: Every user message in order — verbatim when short, condensed to its operative request when long. The user's own voice must survive compaction.
7. Security and Constraints: Every security-relevant instruction or constraint the user stated — preserved VERBATIM so they continue to apply after compaction.
8. Key Decisions and Rationale: Technical decisions, architecture choices, and tool selections, with the reasoning given.
9. Current Work: Precise description of the work currently in progress, with exact file paths and the last known state.
10. Optional Next Step: The single most likely next step to continue the work.

OUTPUT CONSTRAINTS
- Target: ~{target_tokens} tokens
- Be DENSE. Prefer lists over prose. Use exact values where available.
- Include full code snippets for files that matter — do not truncate or paraphrase code.
- Preserve error messages and stack traces verbatim — they matter.
- REDACT any API keys, tokens, passwords, or connection strings — replace with [REDACTED].
- Do NOT treat past instructions as still active — report them as completed or in-progress work.
{focus_note}

CONVERSATION HISTORY TO COMPRESS:
---BEGIN---
{conversation_text}
---END---"""


# -- Utilities ---------------------------------------------------------------

def _content_text(content: Any) -> str:
    """Extract plain text from a message content field (string or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


def _format_conversation_for_summary(messages: List[Dict[str, Any]]) -> str:
    """Format a list of messages into a dense text transcript for summarization.

    Tool outputs are truncated with a marker (the full content is preserved
    in the on-disk transcript, which the summarizer is told about via the
    prompt when truncation occurred).
    """
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = _content_text(msg.get("content", ""))

        if role == "system":
            continue

        prefix = f"[{i}] {role.upper()}"
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            # Include a short argument preview: file paths and commands are
            # what the summary's "Files and Code Sections" / "Errors and
            # Fixes" sections are built from — names alone starve them.
            call_strs = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "") or ""
                preview = args[:200] + "…" if len(args) > 200 else args
                call_strs.append(f"{fn.get('name', '?')}({preview})")
            prefix += f" [tool_call: {', '.join(call_strs)}]"

        if role == "tool" and len(content) > 4000:
            content = (
                content[:2000]
                + " ... [TRUNCATED IN PROMPT — full output preserved in the on-disk transcript] ... "
                + content[-1800:]
            )

        if content:
            # Escape prompt delimiters so a message cannot break the prompt structure
            content = content.replace(CONVERSATION_BEGIN_DELIM, "[BEGIN]").replace(CONVERSATION_END_DELIM, "[END]")
            lines.append(f"{prefix}: {content}")
        else:
            lines.append(prefix)

    return "\n".join(lines)


# -- Main Engine -------------------------------------------------------------

class CompactEngine(ContextEngine):
    """Context engine that does a ZCode-style full conversation rewrite."""

    threshold_percent: float = DEFAULT_THRESHOLD_PERCENT
    protect_first_n: int = DEFAULT_PRESERVE_FIRST_N
    protect_last_n: int = 0  # tail handled explicitly via preserve_last_n

    def __init__(
        self,
        context_length: int = 200000,
        model: str = None,
        provider: str = None,
        base_url: str = None,
        api_key: str = None,
        api_mode: str = None,
    ):
        self._model = model
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key
        self._api_mode = api_mode

        # Token tracking
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.threshold_tokens: int = 0
        self.threshold_tokens_cfg: int = DEFAULT_THRESHOLD_TOKENS
        self._explicit_percent: bool = False
        self.context_length: int = context_length
        self.compression_count: int = 0
        self._consecutive_failures: int = 0

        # Dedicated summarizer model/provider (read from config, may be None).
        self._summary_model: Optional[str] = None
        self._summary_provider: Optional[str] = None
        # Explicit summarizer context window from config (0 = unknown; then
        # Hermes' discovered-length cache is consulted — see _summary_window).
        self._summary_context_length: int = 0

        # Transcript archive (ZCode replica)
        self._session_id: Optional[str] = None
        self.transcript_enabled: bool = True
        self.transcript_dir: str = ""

        # Read compact config
        self.target_tokens: int = DEFAULT_TARGET_TOKENS
        self.preserve_last_n: int = DEFAULT_PRESERVE_LAST_N
        # Sane default even if _load_config fails (hermes_cli missing):
        self._recompute_threshold()
        self._load_config()

    def _recompute_threshold(self) -> None:
        """Resolve the effective fire threshold.

        Fixed beats percent. When BOTH are explicitly configured they
        compose as min() — ride the percent, never past the fixed cap;
        min() is always under 95% of the window (percent is validated to
        0.05-0.95), so the overflow guard below cannot trigger for it.
        A fixed-only value must stay under 95% of the context window,
        else it could never fire before overflow — in that case fall
        back to the percent rule (re-checked on every model switch,
        since the window can change).
        """
        cfg = self.threshold_tokens_cfg
        pct = int(self.context_length * self.threshold_percent)
        if cfg > 0 and self._explicit_percent:
            self.threshold_tokens = min(cfg, pct)
        elif 0 < cfg < self.context_length * 0.95:
            self.threshold_tokens = cfg
        else:
            if cfg > 0:
                logger.warning(
                    "threshold_tokens=%d >= 95%% of context window (%d); "
                    "falling back to threshold_percent=%.2f",
                    cfg, self.context_length, self.threshold_percent,
                )
            self.threshold_tokens = pct

    def _load_config(self):
        """Read compact-context-specific config from config.yaml.

        Reads the dedicated ``compact-context:`` section (isolated from the
        shared ``auxiliary.compression`` block the built-in compressor uses),
        falling back to the legacy ``compact:`` section for back-compat.
        """
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            # Prefer the dedicated section; fall back to legacy `compact:`.
            compact_cfg = cfg.get("compact-context", {})
            if not isinstance(compact_cfg, dict) or not compact_cfg:
                compact_cfg = cfg.get("compact", {}) or {}
            if isinstance(compact_cfg, dict):
                self.target_tokens = int(compact_cfg.get("target_tokens", DEFAULT_TARGET_TOKENS))
                self.protect_first_n = int(compact_cfg.get("preserve_first_n", DEFAULT_PRESERVE_FIRST_N))
                self.preserve_last_n = int(compact_cfg.get("preserve_last_n", DEFAULT_PRESERVE_LAST_N))
                self.transcript_enabled = bool(compact_cfg.get("transcript_enabled", True))
                self.transcript_dir = str(compact_cfg.get("transcript_dir", "") or "")
                # Tunable trigger; was previously a hardcoded class attr (0.20).
                # Explicit = key present AND valid (0.05-0.95). The membership
                # check matters: get()'s default (0.20) also passes the range
                # check, which used to mark the percent "explicit" on every
                # config load and silently cap fixed-only thresholds to
                # min(fixed, 20% of window).
                self._explicit_percent = False
                if "threshold_percent" in compact_cfg:
                    try:
                        tp = float(compact_cfg["threshold_percent"])
                        if 0.05 <= tp <= 0.95:
                            self.threshold_percent = tp
                            self._explicit_percent = True
                    except (TypeError, ValueError):
                        pass
                try:
                    self.transcript_retain = int(compact_cfg.get("transcript_retain", DEFAULT_TRANSCRIPT_RETAIN))
                except (TypeError, ValueError):
                    self.transcript_retain = DEFAULT_TRANSCRIPT_RETAIN
                # Fixed trigger: absolute token count that overrides the
                # percent rule (0 = off). Validated in _recompute_threshold.
                try:
                    self.threshold_tokens_cfg = int(compact_cfg.get("threshold_tokens", 0) or 0)
                except (TypeError, ValueError):
                    self.threshold_tokens_cfg = 0
                # Recompute threshold after config load
                self._recompute_threshold()
                # Dedicated summarizer model for this engine. If set, it
                # overrides the main agent's model when summarizing — needed
                # because the summarizer must read the FULL conversation in
                # one pass, so it needs a window large enough (e.g. GLM-5.2 @ 1M).
                cfg_model = compact_cfg.get("model")
                cfg_provider = compact_cfg.get("provider")
                if cfg_model:
                    self._summary_model = cfg_model
                if cfg_provider:
                    self._summary_provider = cfg_provider
                try:
                    self._summary_context_length = int(compact_cfg.get("summary_context_length", 0) or 0)
                except (TypeError, ValueError):
                    self._summary_context_length = 0
                logger.info(
                    "Compact engine config: target_tokens=%d, preserve_first_n=%d, "
                    "preserve_last_n=%d, threshold_percent=%.2f, threshold_tokens_cfg=%d "
                    "(fires at %d tokens), "
                    "transcript_enabled=%s, "
                    "summary_model=%s, summary_provider=%s, summary_window=%d",
                    self.target_tokens, self.protect_first_n,
                    self.preserve_last_n, self.threshold_percent,
                    self.threshold_tokens_cfg,
                    self.threshold_tokens,
                    self.transcript_enabled,
                    self._summary_model, self._summary_provider,
                    self._summary_context_length,
                )
        except Exception:
            logger.debug("Could not read compact-context config, using defaults")

    def _summary_window(self) -> int:
        """Known context window of the dedicated summarizer (0 = unknown).

        Explicit ``summary_context_length`` config wins; else Hermes'
        discovered-length cache (a pure disk read — the compaction hot path
        must never probe the network). Used by the body guard and per-attempt
        routing: a summarizer with a BIGGER window than main relaxes the
        guard; a smaller one is skipped outright for bodies it cannot read.
        """
        if self._summary_context_length > 0:
            return self._summary_context_length
        if self._summary_model:
            try:
                from agent.model_metadata import get_cached_context_length
                hit = get_cached_context_length(self._summary_model, "")
                if hit and int(hit) > 0:
                    return int(hit)
            except Exception:
                pass
        return 0

    @property
    def name(self) -> str:
        return "compact-context"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0) or (
            self.last_prompt_tokens + self.last_completion_tokens
        )

    def should_compress(self, prompt_tokens: int = None, messages: List[Dict[str, Any]] = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens <= 0 and messages is not None:
            try:
                tokens = estimate_messages_tokens_rough(messages)
            except Exception:
                tokens = 0
        if tokens <= 0:
            return False
        # Urgency punch-through: at >=95% of the window the next main call can
        # 400 on overflow, so compress() MUST run — its failure path is the
        # only place the main-model fallback and the mechanical rescue live.
        # Backoff never suppresses this (the rescue needs no LLM at all).
        if self.context_length > 0 and tokens >= self.context_length * 0.95:
            return True
        # Backoff after repeated summarizer failures to avoid spam — but
        # probe every 5th turn instead of dying for the rest of the session:
        # while suppressed, compress() never runs, so nothing but a probe
        # can ever reset the counter.
        if self._consecutive_failures >= 3 and self._consecutive_failures % 5 != 0:
            self._consecutive_failures += 1
            logger.info("Compact: suppressed by backoff (%d consecutive failures)", self._consecutive_failures)
            return False
        return tokens >= self.threshold_tokens

    # -- Transcript archive --------------------------------------------------

    def _write_transcript(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Write the full pre-compaction conversation to a JSONL transcript.

        Returns the file path (or None when disabled/failed). The path is
        injected into the summary message so the model can re-read exact
        details on demand — the core ZCode "never forgets" mechanism.
        """
        if not self.transcript_enabled:
            return None
        try:
            if self.transcript_dir:
                base = Path(self.transcript_dir)
            elif self._session_id:
                base = Path.home() / ".hermes" / "sessions" / self._session_id
            else:
                base = Path.home() / ".hermes" / "cache" / "compaction_transcripts"
            base.mkdir(parents=True, exist_ok=True)
            # Exclusive-create + unique name: two writes inside one second
            # (or concurrent sessions sharing this fallback dir) must never
            # silently overwrite an earlier archive.
            fd, unique_name = tempfile.mkstemp(
                prefix=f"compaction_transcript_{int(time.time())}_", suffix=".jsonl", dir=str(base))
            path = Path(unique_name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for msg in messages:
                    record = {}
                    for key in ("role", "content", "tool_calls", "tool_name",
                                "tool_call_id", "timestamp"):
                        if key in msg and msg.get(key) is not None:
                            record[key] = msg[key]
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            logger.info("Compact: transcript archived to %s", path)
            return str(path)
        except Exception as e:
            logger.warning("Compact: transcript write failed: %s", e)
            return None

    def _prune_old_transcripts(self, current_path: Optional[str]) -> None:
        """Keep only the N most recent transcript files."""
        try:
            retain = getattr(self, "transcript_retain", DEFAULT_TRANSCRIPT_RETAIN)
            if retain <= 0:
                return
            # Resolve base dir from current_path or config
            if current_path:
                base = Path(current_path).parent
            elif self.transcript_dir:
                base = Path(self.transcript_dir)
            elif self._session_id:
                base = Path.home() / ".hermes" / "sessions" / self._session_id
            else:
                base = Path.home() / ".hermes" / "cache" / "compaction_transcripts"
            if not base.exists():
                return
            files = sorted(base.glob(TRANSCRIPT_GLOB), key=lambda p: p.stat().st_mtime)
            # Keep newest `retain` files
            for old in files[:-retain]:
                try:
                    old.unlink()
                    logger.info("Compact: pruned old transcript %s", old)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Compact: transcript prune skipped: %s", e)

    # -- Compression ---------------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Full-rewrite compression: preserve head + recent tail, summarize
        everything else (ZCode /compact replica).

        The ``force`` kwarg is accepted for signature parity with the built-in
        ContextCompressor — the host's manual /compress path passes it. The
        engine always summarizes when called (manual compression is inherently
        forced), so the flag is accepted but not gating here.

        Returns:
            [system] + [protected head messages] + [summary message] + [recent tail] + [last user message]
        """
        n_messages = len(messages)
        display_tokens = current_tokens if current_tokens else (
            self.last_prompt_tokens or estimate_messages_tokens_rough(messages)
        )

        # Determine head boundary
        head_size = self._compute_head_size(messages)
        if n_messages <= head_size + 2:
            logger.info("Compact: only %d messages, skipping", n_messages)
            return messages

        # Tail: last N messages preserved verbatim (ZCode "recent messages")
        tail = []
        body_end = None
        if self.preserve_last_n > 0:
            tail_start = max(head_size, n_messages - self.preserve_last_n)
            tail = messages[tail_start:]
            body_end = tail_start

        head = messages[:head_size]
        body = messages[head_size:body_end] if body_end is not None else messages[head_size:]

        # Repair tool pairing across the head/tail boundary BEFORE assembly:
        # (a) a tail tool whose calling assistant was summarized into the body
        # must not influence the boundary-marker decision and only then get
        # deleted (that left the summary adjacent to a same-role message —
        # 400), and (b) a head assistant whose tool results fell into the body
        # must lose those calls here (unanswered tool_calls — 400). A tool in
        # head always has its parent assistant before it in head, so head
        # never loses messages; only tail can (plus stripped calls).
        # Repair tool pairing INDEPENDENTLY on head and tail, before assembly.
        # A transaction is preserved only when it lives entirely inside ONE
        # side: a pair straddling the seam (assistant in head, result in
        # tail — e.g. split parallel tool calls) is kept by a joint pass,
        # and the summary message is then inserted BETWEEN pending
        # tool_calls and their remaining results, which the API rejects.
        # Each half of a straddling transaction is dropped (the transcript
        # archive keeps the content). Head tools always have their parent
        # assistant in head; tail tools whose parent sits in head or body
        # are orphans relative to the emitted tail and are dropped;
        # unanswered head calls are stripped.
        head = self._sanitize_tool_pairs(head)
        if tail:
            tail = self._sanitize_tool_pairs(tail)

        if not body:
            logger.info("Compact: nothing to summarize (only head+tail), skipping")
            return messages

        # Archive the FULL conversation before it is summarized away.
        transcript_path = self._write_transcript(messages)
        if transcript_path:
            self._prune_old_transcripts(transcript_path)

        # Guard: the summarizer must read the whole body in ONE pass. Skip
        # when the body exceeds ~80% of the best window in the candidate
        # chain — EXCEPT at >=95% of the window, where returning unchanged
        # means the next main call 400s on overflow: route into the
        # mechanical rescue instead (the transcript is already on disk by
        # this point).
        body_tokens_est = 0
        body_unreadable = False
        try:
            body_tokens_est = estimate_messages_tokens_rough(body)
            guard_window = max(self.context_length, self._summary_window())
            guard_limit = int(guard_window * 0.80)
            if body_tokens_est > guard_limit:
                if display_tokens >= self.context_length * 0.95:
                    body_unreadable = True
                    logger.warning(
                        "Compact: body ~%d tokens exceeds every candidate window "
                        "(guard %d) at %d tokens — mechanical rescue",
                        body_tokens_est, guard_limit, display_tokens,
                    )
                else:
                    logger.warning(
                        "Compact: body ~%d tokens exceeds guard (%d = 80%% of %d) — skipping this turn",
                        body_tokens_est, guard_limit, guard_window,
                    )
                    return messages
        except Exception:
            pass

        summary = None
        emergency = False        # rescue mode: the output MUST fit the window
        tail_removed = 0         # messages trimmed off the tail's front (rescue)
        last_err = "body unreadable in one pass by every candidate window"
        if not body_unreadable:
            logger.info(
                "Compact triggered (%d tokens >= %d threshold): "
                "summarizing %d turns into ~%d tokens (head=%d, tail=%d)",
                display_tokens, self.threshold_tokens,
                len(body), self.target_tokens, len(head), len(tail),
            )

            conversation_text = _format_conversation_for_summary(body)

            # Build the summarization prompt
            focus_note = ""
            if focus_topic:
                focus_note = (
                    f"\nADDITIONAL SUMMARIZATION INSTRUCTIONS (user-supplied):\n"
                    f"Focus topic: \"{focus_topic}\"\n"
                    f"This compaction should PRIORITISE preserving all information "
                    f"related to the focus topic above, while still capturing the "
                    f"other required sections."
                )

            prompt = ZCODE_SUMMARY_PROMPT.format(
                target_tokens=self.target_tokens,
                focus_note=focus_note,
                conversation_text=conversation_text,
            )

            summary, last_err = self._attempt_summary_chain(prompt, body_tokens_est)

        if summary is not None:
            # Code-enforced redaction (prompt says REDACT, but we enforce it)
            summary = _scrub_secrets(summary)
            self._consecutive_failures = 0
        elif display_tokens >= self.context_length * 0.95:
            # Mechanical rescue at the edge of overflow: the LLM chain is
            # down (or the body is unreadable in one pass anywhere) but the
            # transcript is ALREADY on disk — compact anyway with a stub
            # pointing at it. Failing open here means the next main call
            # 400s and the session dies.
            emergency = True
            if not body_unreadable:
                # An unreadable body is a configuration problem, not an LLM
                # failure — don't pollute the backoff counter with it.
                self._consecutive_failures += 1
                logger.warning(
                    "Compact: summarizer chain failed near the context limit (%s) — "
                    "mechanical rescue, full transcript at %s",
                    last_err, transcript_path or "(archive unavailable)",
                )
            summary = MECHANICAL_RESCUE_SUMMARY.format(
                path=transcript_path or
                "(transcript unavailable — older turns could not be archived)",
            )
        else:
            logger.warning("Compact: LLM summary failed: %s — keeping messages unchanged", last_err)
            self._consecutive_failures += 1
            return messages

        if emergency:
            # The rescue contract is that the OUTPUT FITS. A huge preserved
            # tail (one big tool result) can exceed the window on its own —
            # trim oldest tail messages (re-sanitizing so tool pairing stays
            # valid) until head + stub + tail fits ~95% of the window.
            # Content cut here lives in the transcript archive. If even
            # head+stub alone overflows, nothing more can be cut — the
            # post-assembly check below reports it.
            try:
                budget = int(self.context_length * 0.95)
                stub_tokens = estimate_messages_tokens_rough([{"role": "user", "content": summary}])
                while tail and (
                    estimate_messages_tokens_rough(head) + stub_tokens
                    + estimate_messages_tokens_rough(tail) > budget
                ):
                    _prev_len = len(tail)
                    tail = self._sanitize_tool_pairs(tail[1:])
                    tail_removed += _prev_len - len(tail)
            except Exception:
                pass

        # Assemble the compressed message list
        compressed = []
        for i, msg in enumerate(head):
            m = msg.copy()
            # Append compaction note to system prompt
            if i == 0 and m.get("role") == "system":
                note = (
                    "\n\n[Note: The conversation history has been fully compacted. "
                    "A comprehensive summary replaces all prior turns. "
                    "Your persistent memory (MEMORY.md, USER.md) remains fully authoritative.]"
                )
                existing = _content_text(m.get("content", ""))
                if note not in existing:
                    if isinstance(m.get("content"), str):
                        m["content"] = existing + note
            compressed.append(m)

        # Determine summary role (avoid consecutive same-role)
        last_head_role = head[-1].get("role", "user") if head else "user"
        summary_role = "assistant" if last_head_role == "user" else "user"

        summary_text = SUMMARY_PREFIX + "\n\n" + summary + "\n\n" + SUMMARY_END_MARKER
        if transcript_path:
            summary_text += TRANSCRIPT_NOTE_TEMPLATE.format(path=transcript_path)
        if tail:
            summary_text += RECENT_PRESERVED_NOTE
        summary_text += RESUME_NOTE

        compressed.append({
            "role": summary_role,
            "content": summary_text,
            COMPRESSED_SUMMARY_METADATA_KEY: True,
            # Persist the summary invisible in every transcript surface
            # (the desktop renderer maps display_kind='hidden' to null),
            # while it stays in the model's context — mirrors ZCode's
            # "model sees compacted context, UI shows the archive".
            # archive_and_compact() inserts rows as-is, so this stamp
            # must live on the dict itself.
            "display_kind": "hidden",
        })

        # Maintain strict role alternation across the compaction boundary.
        # The summary role is chosen opposite to head[-1]; if the message that
        # follows (tail[0] or the last user message) would repeat it, insert a
        # synthetic user boundary marker so the API never sees two same-role
        # messages in a row.
        # Locate the last user message once, up front — it is used both for
        # the boundary check here and appended at the end if not already
        # preserved in head/tail. (Previously assigned only after this point,
        # which crashed on empty-tail sessions.)
        # Membership is POSITIONAL, never dict-equality: `x not in head`
        # compares contents, so a repeated message (a second "continue")
        # whose earlier twin lives in the head was silently dropped.
        last_user_msg = self._find_last_user_message(messages)
        last_user_idx = None
        for _i in range(len(messages) - 1, -1, -1):
            if messages[_i].get("role") == "user":
                last_user_idx = _i
                break
        _tail_start = max(head_size, (n_messages - self.preserve_last_n)
                          if self.preserve_last_n > 0 else n_messages)
        _lu_in_head = last_user_idx is not None and last_user_idx < head_size
        # tail_removed: emergency trims cut the tail's front, shifting the
        # surviving tail's start forward by that many ORIGINAL indices.
        _lu_in_tail = last_user_idx is not None and last_user_idx >= _tail_start + tail_removed
        if tail or (last_user_msg is not None and not _lu_in_head and not _lu_in_tail):
            _next_role = (tail[0].get("role") if tail else None) or (last_user_msg.get("role") if last_user_msg else None)
            if _next_role == summary_role:
                compressed.append({
                    # The marker sits between the summary and the next message,
                    # both of which carry summary_role — so the marker must take
                    # the OPPOSITE role, or the API sees three same-role messages.
                    "role": "assistant" if summary_role == "user" else "user",
                    "content": (
                        "[Compaction boundary: the summary above replaces the earlier "
                        "conversation; the messages below are preserved recent history. "
                        "Continue with the task.]"
                    ),
                })

        # Append the recent tail verbatim. Head and tail are disjoint slices
        # (tail_start >= head_size by construction), so no dedup is needed —
        # and the old content-equality dedup DROPPED legitimate repeated
        # messages whose twin happened to live in the head.
        for tm in tail:
            compressed.append(tm.copy())

        # Append the last user message only when it repairs the list or the
        # session genuinely ends on it. After an assistant answer it would
        # read as a fresh re-ask of an already-completed task; ending on the
        # assistant's reply is valid and lets the model continue instead.
        # (tail ending on a dangling tool MUST still get the append — a bare
        # trailing tool message is a protocol 400.)
        if last_user_msg is not None and not _lu_in_head and not _lu_in_tail:
            tail_end_role = tail[-1].get("role") if tail else None
            session_ends_on_user = bool(messages) and messages[-1] is last_user_msg
            if session_ends_on_user or tail_end_role in (None, "tool"):
                compressed.append(last_user_msg.copy())

        self.compression_count += 1

        logger.info(
            "Compact complete: %d messages -> %d messages (%.1f%% reduction)",
            n_messages, len(compressed),
            (1 - len(compressed) / max(n_messages, 1)) * 100,
        )

        if emergency:
            # Verify the rescue actually fits — a rescued output above the
            # window just moves the 400 one turn later. The trim above should
            # have handled the tail; if head + stub alone still exceeds the
            # window, nothing more can be cut — report it.
            try:
                final_est = estimate_messages_tokens_rough(compressed)
                if self.context_length > 0 and final_est > self.context_length:
                    logger.error(
                        "Compact: rescue output ~%d tokens still exceeds the context "
                        "window (%d) — system prompt + head floor too large to shrink",
                        final_est, self.context_length)
            except Exception:
                pass

        # Floor warning: if even the compacted list sits at/above the fire
        # threshold, the un-trimmable floor (system prompt + preserved head
        # and tail + summary) is too big for the configured threshold and
        # compaction will re-trigger EVERY turn. Better caught here, on the
        # first compaction, than rediscovered from a log full of
        # "summarizing 1 turns" lines.
        try:
            out_tokens = estimate_messages_tokens_rough(compressed)
            if self.threshold_tokens > 0 and out_tokens >= self.threshold_tokens:
                logger.warning(
                    "Compact: post-compaction size ~%d tokens still >= threshold %d — "
                    "un-trimmable floor too large (system prompt + preserve_first_n=%d "
                    "+ preserve_last_n=%d + summary); raise the threshold or lower "
                    "preserve_* or compaction will re-trigger every turn",
                    out_tokens, self.threshold_tokens,
                    self.protect_first_n, self.preserve_last_n,
                )
        except Exception:
            pass
        return compressed

    def _attempt_summary_chain(self, prompt: str, body_tokens_est: int) -> Tuple[Optional[str], str]:
        """Run the summarizer candidate chain. Returns (summary | None, last_err).

        Dedicated summarizer first (when configured AND its known window can
        hold the body in one pass), then the MAIN model via main_runtime —
        a summarizer too small for the body is skipped outright, and a
        summarizer failure falls back to main instead of failing open.
        """
        call_kwargs = {
            "task": "compression",
            "main_runtime": {
                "model": self._model,
                "provider": self._provider,
                "base_url": self._base_url,
                "api_key": self._api_key,
                "api_mode": self._api_mode,
            },
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self.target_tokens * 1.5),
        }
        summary_window = self._summary_window()
        attempts = []
        if self._summary_model and (summary_window <= 0 or body_tokens_est <= summary_window * 0.80):
            attempts.append(dict(call_kwargs, model=self._summary_model))
            if self._summary_provider:
                attempts[-1]["provider"] = self._summary_provider
        elif self._summary_model:
            logger.info(
                "Compact: body ~%d exceeds summarizer window %d — going straight to the main model",
                body_tokens_est, summary_window,
            )
        if body_tokens_est <= self.context_length * 0.80:
            main_attempt = dict(call_kwargs)
            # Pin the MAIN route explicitly. Without explicit args, call_llm
            # resolves task='compression' from the auxiliary.compression
            # config BEFORE the main runtime — with that config pointing at
            # the same summarizer that just failed, the "fallback" would
            # silently retry the identical route.
            if self._model:
                main_attempt["model"] = self._model
            if self._provider:
                main_attempt["provider"] = self._provider
            if self._base_url:
                main_attempt["base_url"] = self._base_url
            if self._api_key:
                main_attempt["api_key"] = self._api_key
            attempts.append(main_attempt)  # main model, pinned when known

        summary = None
        last_err = "no attempt made"
        for i, attempt_kwargs in enumerate(attempts):
            label = f"attempt {i + 1}/{len(attempts)}"
            try:
                with aux_interrupt_protection():
                    response = call_llm(**attempt_kwargs)
                candidate = response.choices[0].message.content
                if candidate and candidate.strip():
                    summary = candidate
                    if i > 0:
                        logger.info(
                            "Compact: main-model fallback succeeded after %s failed",
                            attempts[0].get("model", "main"),
                        )
                    break
                last_err = f"{label}: empty summary"
            except Exception as e:
                last_err = f"{label}: {e}"
            logger.info("Compact: summarizer %s failed (%s)", label, last_err)
        return summary, last_err

    # -- Helpers -------------------------------------------------------------

    def _compute_head_size(self, messages: List[Dict[str, Any]]) -> int:
        """Compute how many messages to preserve as the protected head."""
        head_count = 0
        non_system_count = 0
        for msg in messages:
            if msg.get("role") == "system":
                head_count += 1
                continue
            non_system_count += 1
            if non_system_count > self.protect_first_n:
                break
            head_count += 1
        return head_count

    def _find_last_user_message(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg
        return None

    @staticmethod
    def _sanitize_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Repair tool pairing in BOTH directions.

        Drops tool messages whose calling assistant is absent, and strips
        tool_calls from assistants whose tool results are absent (partial
        parallel-call splits included). Either shape left in the list is an
        immediate 400 on OpenAI-format backends. An assistant left with no
        calls and no content gets a placeholder so it stays a valid message.
        """
        answered_call_ids = {
            m.get("tool_call_id") for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        active_call_ids = set()  # call ids still referenced by a kept assistant
        cleaned = []
        for msg in messages:
            m = msg.copy()
            if m.get("role") == "assistant" and m.get("tool_calls"):
                valid_calls = [tc for tc in m["tool_calls"] if tc.get("id") in answered_call_ids]
                if valid_calls:
                    m["tool_calls"] = valid_calls
                    active_call_ids.update(tc.get("id") for tc in valid_calls)
                else:
                    m.pop("tool_calls", None)
                    if not m.get("content"):
                        m["content"] = "[Completed earlier actions]"
            elif m.get("role") == "tool":
                if m.get("tool_call_id") not in active_call_ids:
                    continue
            cleaned.append(m)
        return cleaned

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self._model = model
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key
        self._api_mode = api_mode
        # A model switch is a fresh summarizer config — give it a fresh
        # backoff state instead of staying suppressed from the old one.
        self._consecutive_failures = 0
        self.context_length = context_length
        self._load_config()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        logger.info("Compact engine started for session %s", session_id)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        logger.info("Compact engine ending session %s", session_id)

    def on_session_reset(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self._consecutive_failures = 0
        self._session_id = None


# -- Plugin Registration -----------------------------------------------------

def register(ctx):
    """Register the compact engine with the Hermes plugin system."""
    engine = CompactEngine()
    ctx.register_context_engine(engine)
    logger.info("Compact context engine registered")
