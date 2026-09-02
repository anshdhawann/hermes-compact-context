"""Functional test for the ZCode-replica compact-context plugin."""
import importlib.util
import json
import os
import sys
import tempfile
from contextlib import contextmanager

REPO = os.environ.get("HERMES_REPO", os.path.expanduser("~/.hermes/hermes-agent"))
PLUGIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "__init__.py")
sys.path.insert(0, REPO)

# Capture what the summarizer would receive
captured_prompt = {}

@contextmanager
def _noop_protect():
    yield

class FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()

class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]

def fake_call_llm(**kwargs):
    captured_prompt["kwargs"] = kwargs
    return FakeResponse("## Goal\nTest compaction\n## Completed Work\n- did things")

spec = importlib.util.spec_from_file_location("compact_ctx", PLUGIN)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.call_llm = fake_call_llm
mod.aux_interrupt_protection = _noop_protect

engine = mod.CompactEngine(context_length=1_000_000)
engine.on_session_start("test-sess-abc")
engine.transcript_enabled = True
import tempfile as _tf
engine.transcript_dir = _tf.mkdtemp(prefix="compact-test-")
engine.preserve_last_n = 6

# Build a realistic ALTERNATING conversation: system + 3 head exchanges + 8 body turns + 6 tail
messages = [{"role": "system", "content": "You are Hermes."}]
messages += [{"role": "user", "content": "head-q0"}]
messages += [{"role": "assistant", "content": "head-a0"}]
messages += [{"role": "user", "content": "head-q1"}]
messages += [{"role": "assistant", "content": f"body-a{i}", "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]} for i in range(8)]
messages += [{"role": "tool", "tool_call_id": f"call_{i}", "content": "TOOLOUT" * 2000} for i in range(8)]
for i in range(3):
    messages.append({"role": "user", "content": f"tail-q{i}"})
    messages.append({"role": "assistant", "content": f"tail-a{i}"})
messages.append({"role": "user", "content": "FINAL USER MESSAGE - continue now"})

orig_count = len(messages)
result = engine.compress(messages, current_tokens=250_000, focus_topic="grantit pipeline")

print(f"messages: {orig_count} -> {len(result)}")
roles = [m.get("role") for m in result]
print("roles:", roles)

# 1. Transcript written + path injected
transcripts = os.listdir(engine.transcript_dir)
print("\n[1] transcript files:", transcripts)
assert len(transcripts) == 1, "transcript not written"
tpath = os.path.join(engine.transcript_dir, transcripts[0])
with open(tpath) as f:
    tlines = f.readlines()
assert len(tlines) == orig_count, f"transcript has {len(tlines)} lines, expected {orig_count}"
print(f"    transcript has all {len(tlines)} messages ✓")

# 2. Summary message contains transcript path + resume note + tail note
summary_msg = next(m for m in result if m.get("_compressed_summary"))
stext = summary_msg["content"]
assert tpath in stext, "transcript path missing from summary"
assert summary_msg.get("display_kind") == "hidden", (
    "summary must persist display_kind='hidden' (invisible in transcript, "
    "present in model context)"
)
assert "Pick up the last task as if the break never happened" in stext, "resume note missing"
assert "Recent messages are preserved verbatim" in stext, "recent-preserved note missing"
print("[2] summary contains transcript path + resume + preserved-tail notes ✓")

# 3. Tail preserved verbatim — last 6 messages = tail-a0, tail-q1, tail-a1, tail-q2, tail-a2, FINAL
tail_expect = ["tail-a0", "tail-q1", "tail-a1", "tail-q2", "tail-a2", "FINAL USER MESSAGE - continue now"]
tail_in_result = [m for m in result if m.get("content") in tail_expect]
assert len(tail_in_result) == 6, f"tail not fully preserved: {len(tail_in_result)}/6"
print(f"[3] tail preserved verbatim ({len(tail_in_result)}/6) ✓")

# 4. Head preserved AND no body message leaks into head (head = system + first 3 non-system)
head_in_result = [m for m in result if m.get("content") in ("head-q0", "head-a0", "head-q1")]
assert len(head_in_result) == 3, "head not preserved"
assert not any(m.get("content") == "head-q2" for m in result), "unexpected head-q2"
leaked = [m for m in result if isinstance(m.get("content"), str) and m["content"].startswith("body-a")]
assert not leaked, f"head leak: {[m['content'] for m in leaked]}"
print("[4] head (system + first 3 non-system) preserved, no body leak ✓")

# 5. Final user message present
assert any(m.get("content") == "FINAL USER MESSAGE - continue now" for m in result), "final user msg missing"
print("[5] final user message present ✓")

# 6. Prompt is the ZCode 10-section template + focus note + truncated tool marker
prompt = captured_prompt["kwargs"]["messages"][0]["content"]
for sec in ["Primary Request and Intent", "Files and Code Sections", "Errors and Fixes",
            "Security and Constraints", "All user messages", "Current Work", "Optional Next Step",
            "Respond with TEXT ONLY", "grantit pipeline",
            "TRUNCATED IN PROMPT"]:
    assert sec in prompt, f"prompt missing: {sec}"
print("[6] ZCode 10-section prompt + focus topic + truncation marker ✓")

# 7. Summarizer config passed
kw = captured_prompt["kwargs"]
assert kw["task"] == "compression" and kw["max_tokens"] == int(7000 * 1.5)
print(f"[7] summarizer call: task={kw['task']}, max_tokens={kw['max_tokens']} ✓")

# 8. Strict role alternation across the whole result
for i in range(1, len(result)):
    assert result[i]["role"] != result[i-1]["role"], f"role alternation broken at {i}: {result[i-1]['role']} -> {result[i]['role']}"
print("[8] strict role alternation maintained ✓")

# 9. REGRESSION: head ends with a tool message → summary role is "user";
#    tail starts with a user message → boundary marker is inserted between
#    two user-role messages. The marker must flip to "assistant" (a
#    hardcoded "user" marker produced three consecutive user messages,
#    which OpenAI-format backends reject).
messages_b = [{"role": "system", "content": "You are Hermes."}]
messages_b += [{"role": "user", "content": "headB-q0"}]
messages_b += [{"role": "assistant", "content": "headB-a0", "tool_calls": [{"id": "call_b0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}]
messages_b += [{"role": "tool", "tool_call_id": "call_b0", "content": "headB-tool-out"}]  # head ends with tool
for i in range(6):
    messages_b.append({"role": "user", "content": f"bodyB-q{i}"})
    messages_b.append({"role": "assistant", "content": f"bodyB-a{i}"})
for i in range(3):
    messages_b.append({"role": "user", "content": f"tailB-q{i}"})   # tail starts with user
    messages_b.append({"role": "assistant", "content": f"tailB-a{i}"})
# session ends on an assistant reply (manual /compress after an exchange) —
# this makes the last-6 tail start with a user message

engine.transcript_dir = _tf.mkdtemp(prefix="compact-test-b-")
result_b = engine.compress(messages_b, current_tokens=250_000)

roles_b = [m.get("role") for m in result_b]
print("\n[9] scenario B roles:", roles_b)
summary_b = next(m for m in result_b if m.get("_compressed_summary"))
assert summary_b["role"] == "user", f"scenario B summary should be user-role, got {summary_b['role']}"
markers_b = [m for m in result_b if isinstance(m.get("content"), str) and m["content"].startswith("[Compaction boundary:")]
assert len(markers_b) == 1, f"expected 1 boundary marker, got {len(markers_b)}"
assert markers_b[0]["role"] == "assistant", (
    f"boundary marker must flip to assistant when summary is user-role; got {markers_b[0]['role']}"
)
for i in range(1, len(result_b)):
    assert result_b[i]["role"] != result_b[i-1]["role"], (
        f"scenario B role alternation broken at {i}: {result_b[i-1]['role']} -> {result_b[i]['role']}"
    )
print("[9] regression: tool-ended head + user-started tail keeps strict alternation ✓")

import shutil
shutil.rmtree(engine.transcript_dir, ignore_errors=True)

# 10. Fixed threshold_tokens overrides the percent rule (fixed-ONLY mode:
# pin _explicit_percent off so ambient ~/.hermes/config.yaml can't flip us
# into min() composition)
e2 = mod.CompactEngine(context_length=200_000)
e2._explicit_percent = False
pct_default = e2.threshold_tokens  # percent-derived, whatever config says
e2.threshold_tokens_cfg = 150_000
e2._recompute_threshold()
assert e2.threshold_tokens == 150_000, f"fixed override ignored: {e2.threshold_tokens}"
assert e2.should_compress(prompt_tokens=150_000) is True
assert e2.should_compress(prompt_tokens=149_999) is False
print(f"\n[10] fixed threshold: fires at exactly 150K (percent rule would be {pct_default}) ✓")

# 11. Fixed value too big for the window falls back to percent (re-checked on
# every recompute, i.e. every model switch — the window can change under a fixed value)
e2.threshold_tokens_cfg = 195_000  # >= 0.95 * 200K
e2._recompute_threshold()
assert e2.threshold_tokens == pct_default, (
    f"oversized fixed threshold should fall back to percent ({pct_default}), got {e2.threshold_tokens}"
)
e2.threshold_tokens_cfg = 150_000
e2.context_length = 128_000  # switch to a window the fixed value no longer fits
e2._recompute_threshold()
assert e2.threshold_tokens == int(128_000 * e2.threshold_percent), (
    "fixed 150K invalid on 128K window — should re-derive from percent"
)
e2.context_length = 1_000_000
e2._recompute_threshold()
assert e2.threshold_tokens == 150_000, "fixed 150K valid again on 1M window"
print("[11] oversized/outgrown fixed threshold safely falls back to percent ✓")

# 12. Both knobs explicitly set -> compose as min(): ride the percent, capped by the fixed
# (user formula: min(0.8 * context_length, 200K))
e2.threshold_percent = 0.8
e2._explicit_percent = True
e2.threshold_tokens_cfg = 200_000
e2.context_length = 1_000_000
e2._recompute_threshold()
assert e2.threshold_tokens == 200_000, f"big window: fixed cap should win, got {e2.threshold_tokens}"
e2.context_length = 200_000  # 0.8*200K = 160K < 200K cap
e2._recompute_threshold()
assert e2.threshold_tokens == 160_000, f"small window: percent should win, got {e2.threshold_tokens}"
e2.context_length = 128_000  # 0.8*128K = 102.4K -- min() can never trip the 95% guard
e2._recompute_threshold()
assert e2.threshold_tokens == 102_400, f"tiny window: expected 102400, got {e2.threshold_tokens}"
# fixed-only mode (no explicit percent) is unchanged: v2.3.0 semantics
e2._explicit_percent = False
e2.context_length = 1_000_000
e2._recompute_threshold()
assert e2.threshold_tokens == 200_000, "fixed-only must not be capped by a non-explicit percent"
print("[12] both-set composes as min(); fixed-only semantics unchanged ✓")

print("\nALL CHECKS PASSED ✅")
