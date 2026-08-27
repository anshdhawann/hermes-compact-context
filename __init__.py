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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    _re.compile(r'-----BEGIN (?:RSA )?PRIVATE KEY-----'),
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
            tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            prefix += f" [tool_call: {', '.join(tool_names)}]"

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
        self.threshold_tokens: int = int(context_length * self.threshold_percent)
        self.context_length: int = context_length
        self.compression_count: int = 0
        self._consecutive_failures: int = 0

        # Dedicated summarizer model/provider (read from config, may be None).
        self._summary_model: Optional[str] = None
        self._summary_provider: Optional[str] = None

        # Transcript archive (ZCode replica)
        self._session_id: Optional[str] = None
        self.transcript_enabled: bool = True
        self.transcript_dir: str = ""

        # Read compact config
        self.target_tokens: int = DEFAULT_TARGET_TOKENS
        self.preserve_last_n: int = DEFAULT_PRESERVE_LAST_N
        self._load_config()

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
                try:
                    tp = float(compact_cfg.get("threshold_percent", self.threshold_percent))
                    if 0.05 <= tp <= 0.95:
                        self.threshold_percent = tp
                except (TypeError, ValueError):
                    pass
                try:
                    self.transcript_retain = int(compact_cfg.get("transcript_retain", DEFAULT_TRANSCRIPT_RETAIN))
                except (TypeError, ValueError):
                    self.transcript_retain = DEFAULT_TRANSCRIPT_RETAIN
                # Recompute threshold after config load
                self.threshold_tokens = int(self.context_length * self.threshold_percent)
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
                logger.info(
                    "Compact engine config: target_tokens=%d, preserve_first_n=%d, "
                    "preserve_last_n=%d, threshold_percent=%.2f (fires at %d tokens), "
                    "transcript_enabled=%s, "
                    "summary_model=%s, summary_provider=%s",
                    self.target_tokens, self.protect_first_n,
                    self.preserve_last_n, self.threshold_percent,
                    self.threshold_tokens,
                    self.transcript_enabled,
                    self._summary_model, self._summary_provider,
                )
        except Exception:
            logger.debug("Could not read compact-context config, using defaults")

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
        # Backoff after repeated summarizer failures to avoid spam
        if self._consecutive_failures >= 3:
            logger.info("Compact: suppressed by backoff (%d consecutive failures)", self._consecutive_failures)
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens <= 0 and messages is not None:
            try:
                tokens = estimate_messages_tokens_rough(messages)
            except Exception:
                tokens = 0
        if tokens <= 0:
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
            path = base / f"compaction_transcript_{int(time.time())}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
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

        if not body:
            logger.info("Compact: nothing to summarize (only head+tail), skipping")
            return messages

        # Archive the FULL conversation before it is summarized away.
        transcript_path = self._write_transcript(messages)
        if transcript_path:
            self._prune_old_transcripts(transcript_path)

        # Guard: if body itself exceeds ~80% of context window, summarizing will
        # fail (summarizer must read full body in one pass). Skip and back off.
        try:
            body_tokens_est = estimate_messages_tokens_rough(body)
            # Use summarizer window if known, else main context_length as proxy
            # 1M is the advertised summarizer window; cap guard at 800k.
            guard_limit = int(self.context_length * 0.80)
            if body_tokens_est > guard_limit:
                logger.warning(
                    "Compact: body ~%d tokens exceeds guard (%d = 80%% of %d) — skipping this turn",
                    body_tokens_est, guard_limit, self.context_length,
                )
                return messages
        except Exception:
            pass

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

        # Call the LLM for summarization
        try:
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
            # Prefer the dedicated summarizer model if configured. The
            # summarizer must read the full conversation in one pass, so it
            # needs a window large enough (e.g. GLM-5.2 @ 1M). When set,
            # explicit args override main_runtime in call_llm.
            if self._summary_model:
                call_kwargs["model"] = self._summary_model
            if self._summary_provider:
                call_kwargs["provider"] = self._summary_provider
            with aux_interrupt_protection():
                response = call_llm(**call_kwargs)

            summary = response.choices[0].message.content
            if not summary or not summary.strip():
                logger.warning("Compact: LLM returned empty summary, keeping messages unchanged")
                self._consecutive_failures += 1
                return messages
            # Code-enforced redaction (prompt says REDACT, but we enforce it)
            summary = _scrub_secrets(summary)
            self._consecutive_failures = 0

        except Exception as e:
            logger.warning("Compact: LLM summary failed: %s — keeping messages unchanged", e)
            self._consecutive_failures += 1
            return messages

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
        last_user_msg = self._find_last_user_message(messages)
        if tail or (last_user_msg is not None and last_user_msg not in head and last_user_msg not in tail):
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

        # Append the recent tail verbatim (if not already in head)
        for tm in tail:
            if tm not in head:
                compressed.append(tm.copy())

        # Append the last user message (if not already in head or tail)
        if last_user_msg is not None and last_user_msg not in head and last_user_msg not in tail:
            compressed.append(last_user_msg.copy())

        self.compression_count += 1

        # Sanitize orphaned tool pairs
        compressed = self._sanitize_tool_pairs(compressed)

        logger.info(
            "Compact complete: %d messages -> %d messages (%.1f%% reduction)",
            n_messages, len(compressed),
            (1 - len(compressed) / max(n_messages, 1)) * 100,
        )
        return compressed

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
        """Remove orphaned tool messages whose tool_call_id has no parent."""
        active_call_ids = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    cid = tc.get("id", "")
                    if cid:
                        active_call_ids.add(cid)

        cleaned = []
        for msg in messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                if tid and tid not in active_call_ids:
                    continue
            cleaned.append(msg)
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
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
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
