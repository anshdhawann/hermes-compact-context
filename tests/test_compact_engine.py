"""Functional test for the ZCode-replica compact-context plugin."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = os.environ.get("HERMES_REPO", os.path.expanduser("~/.hermes/hermes-agent"))
PLUGIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "__init__.py")
sys.path.insert(0, REPO)

# Capture what the summarizer would receive
captured_prompt = {}

@contextmanager
def _noop_protect():
    yield

# Portable + deterministic environment, installed BEFORE the plugin loads:
# 1) If no Hermes install exists (CI, contributor machines), stub the three
#    `agent.*` modules the plugin imports — the suite must run anywhere.
try:
    import agent  # noqa: F401 — real Hermes install present
except ModuleNotFoundError:
    import types as _types
    _agent = _types.ModuleType("agent"); _agent.__path__ = []
    _aux = _types.ModuleType("agent.auxiliary_client")
    _aux.call_llm = lambda **kw: (_ for _ in ()).throw(RuntimeError("stubbed call_llm"))
    _aux.aux_interrupt_protection = _noop_protect
    _ce = _types.ModuleType("agent.context_engine")
    class ContextEngine:  # minimal stand-in for the ABC
        pass
    _ce.ContextEngine = ContextEngine
    _mm = _types.ModuleType("agent.model_metadata")
    _mm.estimate_messages_tokens_rough = (
        lambda messages: sum(len(str(m.get("content", ""))) for m in messages) // 4
    )
    _agent.auxiliary_client = _aux; _agent.context_engine = _ce; _agent.model_metadata = _mm
    for _name, _mod in (("agent", _agent), ("agent.auxiliary_client", _aux),
                        ("agent.context_engine", _ce), ("agent.model_metadata", _mm)):
        sys.modules[_name] = _mod

# 2) Deterministic config, applied AFTER the plugin load below: never read
# the developer's real ~/.hermes/config.yaml (an ambient-config leak once
# masked a threshold bug). With a real Hermes install we override only
# load_config on the REAL module — agent.* imports other names from it
# (load_env etc.), so a full fake module breaks those imports. Without a
# Hermes install, install the fake module outright.
FAKE_CONFIG = {"compact-context": {
    "target_tokens": 7000, "preserve_first_n": 3, "preserve_last_n": 6,
}}

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

try:
    import hermes_cli.config as _hcc
    _hcc.load_config = lambda: FAKE_CONFIG
except ModuleNotFoundError:
    import types as _t2
    _hc = _t2.ModuleType("hermes_cli")
    _hcc = _t2.ModuleType("hermes_cli.config")
    _hcc.load_config = lambda: FAKE_CONFIG
    _hc.config = _hcc
    sys.modules["hermes_cli"] = _hc
    sys.modules["hermes_cli.config"] = _hcc

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

# 13. Config-load regression: an UNSET percent must not be "explicit" —
# get()'s 0.20 default passes the range check and used to cap fixed-only
# thresholds to min(fixed, 20% of window). Real _load_config path via fake config.
_cc = FAKE_CONFIG["compact-context"]
_cc["threshold_tokens"] = 300000
e13 = mod.CompactEngine(context_length=1_000_000)
assert e13._explicit_percent is False, "unset percent wrongly marked explicit"
assert e13.threshold_tokens == 300000, f"fixed-only capped: {e13.threshold_tokens}"
_cc["threshold_percent"] = 0.8
e13b = mod.CompactEngine(context_length=1_000_000)
assert e13b.threshold_tokens == 300000, "both-set: min(300K, 800K) should be 300K"
_cc["threshold_percent"] = 0.1
e13c = mod.CompactEngine(context_length=1_000_000)
assert e13c.threshold_tokens == 100000, "both-set: min(300K, 100K) should be 100K"
del _cc["threshold_tokens"], _cc["threshold_percent"]
print("[13] config load: fixed-only uncapped; both-set composes as min() ✓")

def _fresh_engine(**attrs):
    e = mod.CompactEngine(context_length=1_000_000)
    e.on_session_start("t-check")
    e.transcript_enabled = False
    for k, v in attrs.items():
        setattr(e, k, v)
    return e

# 14. Orphaned-tool tail (claim 2): tools deleted before the marker decision
# used to leave summary adjacent to a same-role message -> 400.
msgs14 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "tool_calls": [{"id": "cx", "type": "function",
        "function": {"name": "run", "arguments": "{}"}}]},
] + [{"role": "tool", "tool_call_id": "cx", "content": "OUT"} for _ in range(7)]
out14 = _fresh_engine(protect_first_n=2).compress(msgs14, current_tokens=250_000)
for i in range(1, len(out14)):
    assert out14[i]["role"] != out14[i-1]["role"], (
        f"[14] alternation broken at {i}: {out14[i-1]['role']}->{out14[i]['role']}"
    )
assert not any(m.get("role") == "tool" for m in out14), "[14] orphaned tools must be gone"
print(f"[14] orphaned-tool tail keeps strict alternation, roles={[m['role'] for m in out14]} ✓")

# 15. Head cut between assistant(tool_calls) and its tool output (claim 3):
# the unanswered call must be stripped, not shipped to the API.
msgs15 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"},
    {"role": "assistant", "tool_calls": [{"id": "ca", "type": "function",
        "function": {"name": "edit_file", "arguments": '{"path":"x.py"}'}}]},
    {"role": "tool", "tool_call_id": "ca", "content": "ok"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
]
out15 = _fresh_engine(protect_first_n=2, preserve_last_n=2).compress(msgs15, current_tokens=250_000)
answered = {m.get("tool_call_id") for m in out15 if m.get("role") == "tool"}
for m in out15:
    if m.get("tool_calls"):
        assert set(tc.get("id") for tc in m["tool_calls"]) <= answered, (
            f"[15] unanswered tool_calls survived: {[tc.get('id') for tc in m['tool_calls']]}"
        )
assert not out15[2].get("tool_calls"), "[15] head assistant must lose the unanswered call"
print("[15] unanswered tool_calls stripped at the head boundary ✓")

# 16. Secret scrub: whole private-key blocks (any type) redacted, truncated
# blocks still lose the header.
blob = ("pre -----BEGIN OPENSSH PRIVATE KEY-----\nAAAAB3NzaC1yc2EAAAAsecret1\n"
        "-----END OPENSSH PRIVATE KEY----- mid -----BEGIN EC PRIVATE KEY-----\n"
        "MHQCAQEsecret2\n-----END EC PRIVATE KEY----- post")
scrubbed = mod._scrub_secrets(blob)
assert "secret1" not in scrubbed and "secret2" not in scrubbed, "[16] key body leaked"
assert "REDACTED" in scrubbed
assert "BEGIN OPENSSH PRIVATE KEY" not in mod._scrub_secrets("x -----BEGIN OPENSSH PRIVATE KEY-----\nabc")
print("[16] private-key blocks fully redacted (OPENSSH/EC + truncated) ✓")

# 17. Backoff probes every 5th suppression instead of locking out forever,
# and a model switch resets the failure state. Mid-window tokens (v2.5.0:
# 10**9 would now PUNCH THROUGH as urgent >=95% of the window).
e17 = _fresh_engine()
e17._consecutive_failures = 3
assert e17.should_compress(prompt_tokens=500_000) is False   # 3 % 5
assert e17.should_compress(prompt_tokens=500_000) is False   # 4 % 5
assert e17.should_compress(prompt_tokens=500_000) is True    # 5 % 5 -> probe
e17._consecutive_failures = 7
e17.update_model("new-model", context_length=1_000_000)
assert e17._consecutive_failures == 0, "[17] update_model must reset backoff"
print("[17] backoff probes every 5th turn; update_model resets ✓")

# 18. Summarizer transcript includes tool-call arguments (file paths!).
conv18 = mod._format_conversation_for_summary([
    {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function",
        "function": {"name": "edit_file", "arguments": '{"path": "src/app.py"}'}}], "content": ""},
])
assert "edit_file" in conv18 and "src/app.py" in conv18, "[18] args missing from formatter"
print("[18] formatter includes tool-call arguments ✓")

# 19. No stale re-ask after an answered tail: when the tail ends on the
# assistant's answer, the old user message must NOT be re-appended after it.
msgs19 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2-stale-question"},
    {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
        "function": {"name": "run", "arguments": "{}"}}]},
] + [{"role": "tool", "tool_call_id": "c2", "content": "OUT"} for _ in range(4)] + [
    {"role": "assistant", "content": "a2-final-answer"},
]
out19 = _fresh_engine().compress(msgs19, current_tokens=250_000)
assert out19[-1]["role"] == "assistant" and out19[-1]["content"] == "a2-final-answer", (
    f"[19] stale re-ask: list ends on {out19[-1]}"
)
assert not any("u2-stale-question" in str(m.get("content", "")) for m in out19[3:]), (
    "[19] stale question re-injected after the summary"
)
for i in range(1, len(out19)):
    r_prev, r_i = out19[i-1]["role"], out19[i]["role"]
    if r_prev == "tool" and r_i == "tool":
        continue  # parallel tool results after one call are legal
    assert r_i != r_prev, f"[19] alternation broken at {i}: {r_prev}->{r_i}"
print(f"[19] answered tail: ends on the answer, no stale re-ask, roles={[m['role'] for m in out19]} ✓")

# -- v2.5.0: failure ladder (fallback -> rescue -> fail-open) ---------------

_cc = FAKE_CONFIG["compact-context"]
# Shared alternating conversation: head=[S,u0,a0,u1], body=[a1,u2],
# tail=[a2,u3,a3,u4,a4,u5] (n=12 > head+2).
msgs_chain = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
    {"role": "user", "content": "u5"},
]

# 20. Aux->main fallback: dedicated summarizer raises, the MAIN model
# rescues the SAME compaction instead of failing open.
_calls20 = []
def _flaky_llm20(**kwargs):
    _calls20.append(kwargs)
    if kwargs.get("model") == "stub/sum-model":
        raise RuntimeError("summarizer down")
    return FakeResponse("## Goal\nFallback summary OK")
_cc["model"] = "stub/sum-model"
_cc["provider"] = "stub-prov"
e20 = mod.CompactEngine(context_length=1_000_000)
e20.transcript_enabled = False  # keep test writes out of ~/.hermes
assert e20._summary_model == "stub/sum-model", "[20] summary model not loaded from config"
mod.call_llm = _flaky_llm20
out20 = e20.compress(msgs_chain, current_tokens=250_000)
assert len(_calls20) == 2, f"[20] expected summarizer+main attempts, got {len(_calls20)}"
assert _calls20[0].get("model") == "stub/sum-model" and "model" not in _calls20[1], (
    f"[20] chain order wrong: {[c.get('model') for c in _calls20]}"
)
assert any("Fallback summary OK" in str(m.get("content", "")) for m in out20), (
    "[20] main-fallback summary missing from output"
)
assert e20._consecutive_failures == 0, "[20] chain success must reset backoff"
del _cc["model"], _cc["provider"]
mod.call_llm = fake_call_llm
print("[20] summarizer failure falls back to main model in the same compaction ✓")

# 21. Mechanical rescue: the whole chain is down AND the session sits at
# >=95% of the window — compact anyway with a stub pointing at the
# already-archived transcript instead of letting the next main call 400.
def _dead_llm(**kwargs):
    raise RuntimeError("everything down")
_cc["model"] = "stub/sum-model"
e21 = mod.CompactEngine(context_length=1_000_000)
e21.on_session_start("t-rescue")
e21.transcript_enabled = True
e21.transcript_dir = tempfile.mkdtemp(prefix="compact-rescue-")
mod.call_llm = _dead_llm
out21 = e21.compress(msgs_chain, current_tokens=int(1_000_000 * 0.96))
# NOTE: output count can equal input count (head 4 + stub + marker + tail 6);
# the win is the ~960K-token body collapsing into the stub, not msg count.
assert out21 is not msgs_chain, "[21] rescue must still compact"
stub21 = [m for m in out21 if "Compaction fallback notice" in str(m.get("content", ""))]
assert stub21, "[21] rescue stub summary missing"
assert "compaction_transcript" in stub21[0]["content"], "[21] stub must point at the transcript"
assert not any("## Goal" in str(m.get("content", "")) for m in out21), "[21] no LLM summary expected"
assert e21._consecutive_failures == 1, "[21] rescue must still count the LLM failure"
assert e21.compression_count == 1
for i in range(1, len(out21)):
    assert out21[i]["role"] != out21[i-1]["role"], (
        f"[21] alternation broken at {i}: {out21[i-1]['role']}->{out21[i]['role']}"
    )
print(f"[21] mechanical rescue near overflow, roles={[m['role'] for m in out21]} ✓")

# 22. Fail-open preserved BELOW the urgency line: chain down at mid tokens
# returns messages unchanged (rescue is a last resort, not the default).
e22 = mod.CompactEngine(context_length=1_000_000)
e22.transcript_enabled = False
e22.on_session_start("t-failopen")
out22 = e22.compress(msgs_chain, current_tokens=250_000)
assert out22 is msgs_chain, "[22] non-urgent failure must return messages unchanged"
assert e22._consecutive_failures == 1
del _cc["model"]
mod.call_llm = fake_call_llm
print("[22] non-urgent summarizer failure still fails open (unchanged messages) ✓")

# 23. Urgency punch-through: backoff must never suppress a turn at >=95% of
# the window — compress() owns the main-model fallback and the rescue.
e23 = mod.CompactEngine(context_length=1_000_000)
e23._consecutive_failures = 4
assert e23.should_compress(prompt_tokens=500_000) is False, "[23] mid tokens must stay suppressed"
assert e23.should_compress(prompt_tokens=int(1_000_000 * 0.96)) is True, (
    "[23] urgent turn must punch through backoff"
)
print("[23] >=95%-of-window turns punch through backoff suppression ✓")

# 24. Chain-aware guard: a KNOWN bigger summarizer window relaxes the body
# guard (v2.4.1 would skip); a KNOWN smaller one routes straight to main.
_calls24 = []
def _tap_llm24(**kwargs):
    _calls24.append(kwargs)
    return FakeResponse("## Goal\nTap summary")
mod.call_llm = _tap_llm24
msgs24 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "big " + "X" * (260_000 * 4)},  # ~260K-token body message
    {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
]
_cc["model"] = "stub/sum-model"
_cc["summary_context_length"] = 1_000_000
_cc["preserve_last_n"] = 2  # keep the big message in the BODY, not the tail
e24a = mod.CompactEngine(context_length=200_000)  # main guard alone = 160K < body
e24a.transcript_enabled = False
out24a = e24a.compress(msgs24, current_tokens=250_000)
assert any("Tap summary" in str(m.get("content", "")) for m in out24a), (
    "[24a] big-window summarizer must NOT be blocked by the main-window guard"
)
assert _calls24[0].get("model") == "stub/sum-model", "[24a] summarizer attempt expected"
del _cc["summary_context_length"]
_cc["summary_context_length"] = 100_000          # 80K one-pass cap < 260K body
e24b = mod.CompactEngine(context_length=1_000_000)
e24b.transcript_enabled = False
out24b = e24b.compress(msgs24, current_tokens=250_000)
assert any("Tap summary" in str(m.get("content", "")) for m in out24b), "[24b] compaction must still happen"
assert len(_calls24) == 2 and "model" not in _calls24[1], (
    f"[24b] small-window summarizer must be skipped, went straight to main: {_calls24[1].get('model')}"
)
e24c = mod.CompactEngine(context_length=200_000)  # body exceeds BOTH windows
e24c.transcript_enabled = False
# 170K is above the guard but BELOW the 95%-of-window urgency line (190K):
# skip. (v2.5.1: at >=190K the same input routes into the rescue — see [26].)
out24c = e24c.compress(msgs24, current_tokens=170_000)
assert out24c is msgs24 and len(_calls24) == 2, "[24c] body exceeding both windows must be skipped, no calls"
del _cc["model"], _cc["summary_context_length"]
_cc["preserve_last_n"] = 6
mod.call_llm = fake_call_llm
print("[24] guard relaxes for a bigger summarizer window; routes around a smaller one ✓")

# 25. Post-compact floor warning: a threshold below the un-trimmable floor
# (system prompt + preserved head/tail + summary) re-triggers compaction
# every turn — must warn on the FIRST compaction (the 2026-09-01 mess).
import logging as _logging
_records25 = []
class _ListHandler(_logging.Handler):
    def emit(self, record):
        _records25.append(record.getMessage())
_h25 = _ListHandler()
mod.logger.addHandler(_h25)
try:
    _cc["threshold_tokens"] = 50  # absurdly low: any compacted output exceeds it
    e25 = mod.CompactEngine(context_length=1_000_000)
    e25.transcript_enabled = False
    assert e25.threshold_tokens == 50
    e25.compress(msgs_chain, current_tokens=250_000)
    assert any("still >= threshold" in r for r in _records25), "[25] floor warning not emitted"
finally:
    mod.logger.removeHandler(_h25)
    del _cc["threshold_tokens"]
print("[25] post-compact floor warning fires when output >= threshold ✓")

# -- v2.5.1: Astra review — guard/rescue, dup messages, tailless sanitize,
#    PKCS#8 keys, transcript collisions --------------------------------------

# 26. The overflow guard must not bypass the rescue: a body too big for
# EVERY candidate window at >=95% of the window compacts mechanically —
# no LLM call is even possible.
def _forbidden_llm(**kwargs):
    raise AssertionError("[26] LLM must not be called for an unreadable body")
mod.call_llm = _forbidden_llm
e26 = mod.CompactEngine(context_length=200_000)
e26.on_session_start("t-oversize")
e26.transcript_enabled = True
e26.transcript_dir = tempfile.mkdtemp(prefix="compact-oversize-")
msgs26 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "huge " + "X" * (170_000 * 4)},  # ~170K-token body message
    {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
    {"role": "user", "content": "u5"}, {"role": "assistant", "content": "a5"},
]
out26 = e26.compress(msgs26, current_tokens=195_000)  # 97.5% of the 200K window
assert out26 is not msgs26, "[26] guard must rescue, not fail open, near overflow"
stub26 = [m for m in out26 if "Compaction fallback notice" in str(m.get("content", ""))]
assert stub26 and "compaction_transcript" in stub26[0]["content"], "[26] rescue stub must point at the transcript"
assert e26._consecutive_failures == 0, "[26] unreadable body is not an LLM failure"
assert e26.compression_count == 1
assert not any("XXXX" in str(m.get("content", "")) for m in out26), "[26] oversized body must be dropped"
for i in range(1, len(out26)):
    assert out26[i]["role"] != out26[i-1]["role"], (
        f"[26] alternation broken at {i}: {out26[i-1]['role']}->{out26[i]['role']}"
    )
mod.call_llm = fake_call_llm
print(f"[26] oversized body at 97% of window -> mechanical rescue, roles={[m['role'] for m in out26]} ✓")

# 27. Repeated messages survive: the old content-equality dedup dropped a
# tail message whose identical twin lived in the head.
msgs27 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "continue"},       # head twin
    {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "continue"},       # tail copy — must survive
    {"role": "assistant", "content": "a3"},
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
    {"role": "user", "content": "u5"}, {"role": "assistant", "content": "a5"},
]
out27 = _fresh_engine().compress(msgs27, current_tokens=250_000)
n_continue = sum(1 for m in out27 if m.get("role") == "user" and m.get("content") == "continue")
assert n_continue == 2, f"[27] repeated 'continue' lost: found {n_continue}"
for i in range(1, len(out27)):
    assert out27[i]["role"] != out27[i-1]["role"], (
        f"[27] alternation broken at {i}: {out27[i-1]['role']}->{out27[i]['role']}"
    )
print("[27] repeated 'continue' in head and tail: both survive, alternation intact ✓")

# 28. preserve_last_n=0: sanitization used to be skipped entirely (it ran
# only under `if tail:`) — a head assistant kept tool_calls whose results
# were summarized away (unanswered calls = protocol 400).
msgs28 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"},
    {"role": "assistant", "tool_calls": [{"id": "cx", "type": "function",
        "function": {"name": "run", "arguments": "{}"}}], "content": ""},
    {"role": "tool", "tool_call_id": "cx", "content": "OUT"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
]
out28 = _fresh_engine(protect_first_n=2, preserve_last_n=0).compress(msgs28, current_tokens=250_000)
assert not any(m.get("tool_calls") for m in out28), "[28] unanswered tool_calls survived a tailless compaction"
for i in range(1, len(out28)):
    assert out28[i]["role"] != out28[i-1]["role"], (
        f"[28] alternation broken at {i}: {out28[i-1]['role']}->{out28[i]['role']}"
    )
print(f"[28] preserve_last_n=0 sanitizes the head, roles={[m['role'] for m in out28]} ✓")

# 29. PKCS#8 plain header and truncated-block key material fully redacted.
blob29a = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqsecretA\n-----END PRIVATE KEY-----"
blob29b = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADtruncated"  # cut, at end of input
blob29c = "-----BEGIN RSA PRIVATE KEY-----\nabc123trunc\n-----BEGIN CERTIFICATE-----\nCERTDATA"
for blob, gone, kept in (
    (blob29a, "secretA", None),
    (blob29b, "truncated", None),
    (blob29c, "abc123trunc", "CERTDATA"),  # redact to next BEGIN, don't over-eat
):
    scrubbed29 = mod._scrub_secrets(blob)
    assert gone not in scrubbed29, f"[29] key material leaked: {gone!r}"
    if kept:
        assert kept in scrubbed29, f"[29] over-redaction ate {kept!r}"
assert "REDACTED" in mod._scrub_secrets(blob29a)
print("[29] PKCS#8 + truncated private-key blocks fully redacted ✓")

# 30. Transcript writes are exclusive-create unique: two writes inside one
# second must both survive (they used to overwrite each other).
e30 = mod.CompactEngine(context_length=1_000_000)
e30.transcript_enabled = True
e30.transcript_dir = tempfile.mkdtemp(prefix="compact-collide-")
p1 = e30._write_transcript(msgs_chain)
p2 = e30._write_transcript(msgs_chain)
assert p1 and p2 and p1 != p2, f"[30] filenames collided: {p1} == {p2}"
assert os.path.exists(p1) and os.path.exists(p2), "[30] one archive was overwritten"
assert os.path.basename(p1).startswith("compaction_transcript_"), "[30] glob/prune pattern broken"
print("[30] same-second transcript writes are unique and both survive ✓")

# -- v2.5.2: Astra round 2 — straddling transactions, pinned fallback,
#    rescue output must fit ---------------------------------------------------

# 31. A tool transaction straddling the head|tail seam must not survive as
# two halves with the summary between pending calls and their results
# (joint sanitization kept the pair; assembly split it -> protocol 400).
msgs31 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"},
    {"role": "assistant", "tool_calls": [
        {"id": "tc1", "type": "function", "function": {"name": "run", "arguments": "{}"}},
        {"id": "tc2", "type": "function", "function": {"name": "run", "arguments": "{}"}},
    ], "content": ""},
    {"role": "tool", "tool_call_id": "tc1", "content": "R1"},   # head ends here
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
    {"role": "tool", "tool_call_id": "tc2", "content": "R2-LATE"},  # tail straddler
    {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
]
out31 = _fresh_engine().compress(msgs31, current_tokens=250_000)
flat31 = str(out31)
assert "tc2" not in flat31, "[31] straddling call/result must be dropped from BOTH sides"
answered31 = {m.get("tool_call_id") for m in out31 if m.get("role") == "tool"}
for m in out31:
    if m.get("tool_calls"):
        assert set(tc.get("id") for tc in m["tool_calls"]) <= answered31, (
            f"[31] pending calls left open: {[tc.get('id') for tc in m['tool_calls']]}"
        )
for i in range(1, len(out31)):
    r_prev, r_i = out31[i-1]["role"], out31[i]["role"]
    if r_prev == "tool" and r_i == "tool":
        continue  # parallel tool results after one call are legal
    assert r_i != r_prev, f"[31] alternation broken at {i}: {r_prev}->{r_i}"
print(f"[31] straddling transaction dropped from both sides, roles={[m['role'] for m in out31]} ✓")

# 32. The main-model fallback is PINNED: without explicit model args,
# call_llm resolves task='compression' from auxiliary.compression config
# BEFORE the main runtime — the "fallback" could retry the same summarizer.
_calls32 = []
def _pin_llm32(**kwargs):
    _calls32.append(kwargs)
    if kwargs.get("model") == "stub/sum-model":
        raise RuntimeError("summarizer down")
    return FakeResponse("## Goal\nPinned fallback OK")
_cc["model"] = "stub/sum-model"
e32 = mod.CompactEngine(context_length=1_000_000)
e32.update_model(model="real-main-model", provider="main-prov", context_length=1_000_000)
e32.transcript_enabled = False  # AFTER update_model: _load_config resets it from config
mod.call_llm = _pin_llm32
out32 = e32.compress(msgs_chain, current_tokens=250_000)
assert len(_calls32) == 2, f"[32] expected 2 attempts, got {len(_calls32)}"
assert _calls32[1].get("model") == "real-main-model", (
    f"[32] fallback not pinned to main model: {_calls32[1].get('model')}"
)
assert _calls32[1].get("provider") == "main-prov", "[32] fallback provider not pinned"
assert any("Pinned fallback OK" in str(m.get("content", "")) for m in out32)
del _cc["model"]
mod.call_llm = fake_call_llm
print("[32] main-model fallback passes the main route explicitly ✓")

# 33. The rescue output MUST fit: a huge preserved tail tool result gets
# trimmed (whole transactions) until head + stub + tail fits the window.
def _dead_llm33(**kwargs):
    raise RuntimeError("down")
e33 = mod.CompactEngine(context_length=10_000)
e33.on_session_start("t-trim")
e33.transcript_enabled = True
e33.transcript_dir = tempfile.mkdtemp(prefix="compact-trim-")
mod.call_llm = _dead_llm33
msgs33 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "u3"},
    {"role": "assistant", "tool_calls": [{"id": "big", "type": "function",
        "function": {"name": "run", "arguments": "{}"}}], "content": ""},
    {"role": "tool", "tool_call_id": "big", "content": "T" * 200_000},  # ~50K tokens, in tail
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
]
out33 = e33.compress(msgs33, current_tokens=9_800)  # 98% of the 10K window -> rescue
est33 = mod.estimate_messages_tokens_rough(out33)
assert est33 <= 9_500, f"[33] rescue output does not fit: ~{est33} tokens > 9500"
assert not any("TTTT" in str(m.get("content", "")) for m in out33), "[33] oversized tail content must be trimmed"
assert any("Compaction fallback notice" in str(m.get("content", "")) for m in out33), "[33] rescue stub missing"
assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in out33), "[33] trimmed transaction left debris"
for i in range(1, len(out33)):
    assert out33[i]["role"] != out33[i-1]["role"], (
        f"[33] alternation broken at {i}: {out33[i-1]['role']}->{out33[i]['role']}"
    )
mod.call_llm = fake_call_llm
print(f"[33] rescue trims the tail until output fits (~{est33} tokens), roles={[m['role'] for m in out33]} ✓")

# -- v2.6.0: Astra round 3 ----------------------------------------------------

# 34. JSON-syntax and unquoted secrets are redacted (the closing quote
# between field name and colon used to break the match).
blob34a = '"password": "correct-horse-battery-staple"'
blob34b = '"api_key": "custom-secret-123456789"'
blob34c = "API_KEY=env-secret-value-987654321"
for blob34, gone34 in ((blob34a, "correct-horse-battery-staple"),
                       (blob34b, "custom-secret-123456789"),
                       (blob34c, "env-secret-value-987654321")):
    out34 = mod._scrub_secrets(blob34)
    assert gone34 not in out34, f"[34] secret survived: {gone34!r} in {out34!r}"
print("[34] JSON/unquoted secret assignments fully redacted ✓")

# 35. finish_reason='length' is NOT a usable summary: a truncated response
# falls through to the fallback instead of replacing history mid-thought.
class _ChoiceFR:
    def __init__(self, content, finish):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish
def _resp_fr(content, finish):
    return type("R", (), {"choices": [_ChoiceFR(content, finish)]})()
_calls35 = []
def _trunc_llm35(**kwargs):
    _calls35.append(kwargs)
    if kwargs.get("model") == "stub/sum-model":
        return _resp_fr("## <analysis> partial thou", "length")
    return _resp_fr("## Goal\nFull summary", "stop")
_cc["model"] = "stub/sum-model"
e35 = mod.CompactEngine(context_length=1_000_000)
e35.transcript_enabled = False
mod.call_llm = _trunc_llm35
out35 = e35.compress(msgs_chain, current_tokens=250_000)
assert len(_calls35) == 2, f"[35] truncated summary must fall back, attempts={len(_calls35)}"
assert any("Full summary" in str(m.get("content", "")) for m in out35), "[35] fallback summary missing"
assert e35._consecutive_failures == 0
def _all_trunc_llm(**kwargs):
    return _resp_fr("## <analysis> still partial", "length")
mod.call_llm = _all_trunc_llm
e35b = mod.CompactEngine(context_length=1_000_000)
e35b.transcript_enabled = False
out35b = e35b.compress(msgs_chain, current_tokens=250_000)
assert out35b is msgs_chain and e35b._consecutive_failures == 1, "[35] all-truncated must fail open"
del _cc["model"]
mod.call_llm = fake_call_llm
print("[35] finish_reason=length rejected; falls back or fails open ✓")

# 36. Config reloads are complete: removed keys RESET, and one invalid value
# must not abort the load (a bad target_tokens once left a stale 200K
# threshold after switching to a 10K window).
_cc["model"] = "stub/sum-model"
e36 = mod.CompactEngine(context_length=1_000_000)
del _cc["model"]
e36._load_config()
assert e36._summary_model is None, "[36] removed model key must reset on reload"
_cc["threshold_percent"] = 0.8
e36b = mod.CompactEngine(context_length=1_000_000)
del _cc["threshold_percent"]
e36b._load_config()
assert e36b.threshold_tokens == 200_000 and e36b._explicit_percent is False, (
    f"[36] removed percent must revert to default: {e36b.threshold_tokens}"
)
_cc["target_tokens"] = "garbage"
_cc["threshold_tokens"] = 300000
e36c = mod.CompactEngine(context_length=1_000_000)
assert e36c.target_tokens == 7000, f"[36] invalid target_tokens must default: {e36c.target_tokens}"
e36c.update_model("m", context_length=10_000)
assert e36c.threshold_tokens == 2000, (
    f"[36] threshold not recomputed after window switch: {e36c.threshold_tokens}"
)
del _cc["target_tokens"], _cc["threshold_tokens"]
print("[36] config reloads: removals reset, per-key validation, threshold always recomputed ✓")

# 37. The main fallback pins api_mode too (the resolver otherwise takes the
# auxiliary config's mode over the main runtime's).
_calls37 = []
def _pin_llm37(**kwargs):
    _calls37.append(kwargs)
    if kwargs.get("model") == "stub/sum-model":
        raise RuntimeError("down")
    return FakeResponse("## Goal\nMode pinned")
_cc["model"] = "stub/sum-model"
e37 = mod.CompactEngine(context_length=1_000_000)
e37.update_model(model="real-main", provider="main-prov", base_url="https://main.example",
                 api_key="mk", api_mode="chat_completions", context_length=1_000_000)
e37.transcript_enabled = False  # AFTER update_model: _load_config resets it from config
mod.call_llm = _pin_llm37
e37.compress(msgs_chain, current_tokens=250_000)
assert len(_calls37) == 2 and _calls37[1].get("api_mode") == "chat_completions", (
    f"[37] fallback api_mode not pinned: {_calls37[1].get('api_mode')}"
)
del _cc["model"]
mod.call_llm = fake_call_llm
print("[37] main fallback pins model/provider/base_url/api_key/api_mode ✓")

# 38. No verified transcript -> NO rescue: urgent + dead chain with archives
# disabled keeps messages (deleting content with no recovery path is worse
# than a visible overflow).
def _dead_llm38(**kwargs):
    raise RuntimeError("down")
e38 = mod.CompactEngine(context_length=200_000)
e38.transcript_enabled = False
mod.call_llm = _dead_llm38
out38 = e38.compress(msgs_chain, current_tokens=int(200_000 * 0.96))
assert out38 is msgs_chain, "[38] rescue must be refused without a verified archive"
assert e38._consecutive_failures == 1
assert not any("Compaction fallback notice" in str(m.get("content", "")) for m in out38)
mod.call_llm = fake_call_llm
print("[38] urgent failure with no archive fails open instead of destroying content ✓")

# 39. Pruning keeps the chain root: the OLDEST transcript (only verbatim
# copy of the earliest messages) survives retention; scoping is per session.
e39 = mod.CompactEngine(context_length=1_000_000)
e39.on_session_start("t-prune")
e39.transcript_enabled = True
e39.transcript_dir = tempfile.mkdtemp(prefix="compact-prune-")
paths39 = [e39._write_transcript(msgs_chain) for _ in range(4)]
for _i, _p in enumerate(paths39):  # deterministic order despite same-second writes
    os.utime(_p, (1_700_000_000 + _i, 1_700_000_000 + _i))
for _p in paths39[1:]:
    e39._prune_old_transcripts()
assert os.path.exists(paths39[0]), "[39] chain-root transcript was pruned (verbatim originals lost)"
assert os.path.exists(paths39[3]), "[39] newest transcript missing"
assert "_consolidated_into" in json.loads(Path(paths39[1]).read_text()), "[39] middle archive not replaced with a redirect"
print("[39] prune keeps newest N plus the chain root, scoped per session ✓")

# 40. Zero-tail + completed answer: the previous user request is NOT
# re-appended as a re-ask (it re-triggered completed work).
msgs40 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2-final-q"}, {"role": "assistant", "content": "a2-done"},
]
out40 = _fresh_engine(preserve_last_n=0).compress(msgs40, current_tokens=250_000)
assert not (out40[-1]["role"] == "user" and out40[-1].get("content") == "u2-final-q"), (
    f"[40] completed request re-asked: last={out40[-1]}"
)
for i in range(1, len(out40)):
    assert out40[i]["role"] != out40[i-1]["role"], (
        f"[40] alternation broken at {i}: {out40[i-1]['role']}->{out40[i]['role']}"
    )
print(f"[40] zero-tail completed session: no re-ask, roles={[m['role'] for m in out40]} ✓")

# 41. Coherent continuation instructions: with NO user message after the
# summary the "respond ONLY to the latest user message" prefix would
# contradict the resume note — the continuation variant is used instead.
summ41 = [m for m in out40 if m.get(mod.COMPRESSED_SUMMARY_METADATA_KEY)]
assert summ41, "[41] summary message not found"
assert "Respond ONLY" not in summ41[0]["content"], "[41] contradictory prefix used with no following user"
assert "No new user message follows" in summ41[0]["content"], "[41] continuation prefix missing"
summ27 = [m for m in out27 if m.get(mod.COMPRESSED_SUMMARY_METADATA_KEY)]
assert summ27 and "Respond ONLY" in summ27[0]["content"], (
    "[41] classic prefix must be kept when a user message DOES follow"
)
print("[41] prefix variant matches whether a user message follows the summary ✓")

# 42. Budget enforcement applies to SUCCESSFUL compactions too, measured on
# the COMPLETE assembled output (wrappers and appended messages included).
# Window is 20K: the REQUEST (~12K incl. output reserve) passes the 16K
# guard, but the huge preserved tail busts the 18K budget -> trimmed.
# Trimming tail content requires a VERIFIED transcript archive (the tail
# was never summarized — cutting it without recovery destroys it), so this
# scenario runs with transcripts enabled.
e42 = mod.CompactEngine(context_length=20_000)  # budget = 18000 after headroom
e42.transcript_enabled = True
e42.transcript_dir = tempfile.mkdtemp(prefix="compact-test-42-")
msgs42 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "u3"},
    {"role": "assistant", "tool_calls": [{"id": "big42", "type": "function",
        "function": {"name": "run", "arguments": "{}"}}], "content": ""},
    {"role": "tool", "tool_call_id": "big42", "content": "T" * 200_000},
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
]
out42 = e42.compress(msgs42, current_tokens=15_000)  # NOT urgent — success path
est42 = mod.estimate_messages_tokens_rough(out42)
assert est42 <= 18_000, f"[42] successful compaction over budget: ~{est42} tokens"
assert not any("TTTT" in str(m.get("content", "")) for m in out42), "[42] oversized tail not trimmed"
assert e42._consecutive_failures == 0, "[42] success path must reset failures"
for i in range(1, len(out42)):
    assert out42[i]["role"] != out42[i-1]["role"], (
        f"[42] alternation broken at {i}: {out42[i-1]['role']}->{out42[i]['role']}"
    )
print(f"[42] success path enforces the output budget (~{est42} tokens) ✓")

# 43. The guard measures the FORMATTED request: a body whose raw tool output
# crosses the guard still compacts when the formatter truncates it small.
_calls43 = []
def _tap43(**kwargs):
    _calls43.append(kwargs)
    return FakeResponse("## Goal\nFormatted guard OK")
e43 = mod.CompactEngine(context_length=200_000)  # raw guard alone = 160K
e43.transcript_enabled = False
mod.call_llm = _tap43
msgs43 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "tool_calls": [{"id": "t43", "type": "function",
        "function": {"name": "run", "arguments": "{}"}}], "content": ""},
    {"role": "tool", "tool_call_id": "t43", "content": "X" * 800_000},  # ~200K raw, truncated to ~1K
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
    {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
]
out43 = e43.compress(msgs43, current_tokens=250_000)
assert len(_calls43) >= 1, "[43] formatted-small request was guard-skipped"
assert any("Formatted guard OK" in str(m.get("content", "")) for m in out43), "[43] compaction missing"
mod.call_llm = fake_call_llm
print(f"[43] guard measures the formatted request ({len(_calls43)} attempt made) ✓")

# -- v2.6.1: Astra round 4 — postcondition + oversized request, archive-
#    gated trimming, consolidation, quoted secrets, config ranges --------

# 44. An oversized LATEST USER MESSAGE must not re-enter the output verbatim
# after the trim loop empties the tail: it is truncated to the remaining
# budget with a pointer to the verified archive (full text recoverable).
e44 = mod.CompactEngine(context_length=10_000)  # budget = 8976
e44.target_tokens = 1000  # small output reserve so the guard passes on 10K
e44.transcript_enabled = True
e44.transcript_dir = tempfile.mkdtemp(prefix="compact-test-44-")
msgs44 = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "bq1"}, {"role": "assistant", "content": "ba1"},
    {"role": "user", "content": "bq2"}, {"role": "assistant", "content": "ba2"},
]
for i in range(3):
    msgs44.append({"role": "user", "content": f"tail-q{i} small"})
    msgs44.append({"role": "assistant", "content": f"tail-a{i} small"})
GIANT44 = "Please analyse this dataset: " + ("g" * 48_000)  # ~12K tokens
msgs44.append({"role": "user", "content": GIANT44})
out44 = e44.compress(msgs44, current_tokens=9_000)  # non-urgent success path
est44 = mod.estimate_messages_tokens_rough(out44)
assert est44 <= 8_976, f"[44] oversized latest request re-broke the window: ~{est44} tokens"
last44 = out44[-1]
assert last44["role"] == "user", f"[44] expected appended user last, got {last44['role']}"
assert "truncated by context compaction" in str(last44.get("content", "")), (
    "[44] truncation note missing from the oversized request"
)
_tpath44 = None
for _f in os.listdir(e44.transcript_dir):
    _tpath44 = os.path.join(e44.transcript_dir, _f)
if _tpath44:
    with open(_tpath44) as _fh:
        _archived44 = _fh.read()
    assert "g" * 1000 in _archived44, "[44] full original text not archived for recovery"
    assert _tpath44 in str(last44.get("content", "")), "[44] truncation note lacks archive pointer"
for i in range(1, len(out44)):
    assert out44[i]["role"] != out44[i-1]["role"], (
        f"[44] alternation broken at {i}: {out44[i-1]['role']}->{out44[i]['role']}"
    )
print(f"[44] oversized latest request truncated to fit (~{est44} tokens), archive pointer kept ✓")

# 45. Without a verified transcript archive the trim loop must REFUSE to cut
# the preserved tail: that content was never sent to the summarizer, so
# trimming would destroy it outright. Output stays intact (over budget, and
# loudly reported) instead of silently losing messages.
_records45 = []
class _ListHandler45(_logging.Handler):
    def emit(self, record):
        _records45.append(record.getMessage())
_h45 = _ListHandler45()
mod.logger.addHandler(_h45)
try:
    e45 = mod.CompactEngine(context_length=10_000)
    e45.target_tokens = 1000
    e45.transcript_enabled = False  # no archive -> no permission to trim
    msgs45 = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "b1"}, {"role": "assistant", "content": "b2"},
        {"role": "user", "content": "b3"}, {"role": "assistant", "content": "b4"},
    ]
    UNIQUE45 = "UNIQUE-TAIL-DETAIL-xyzzy"
    FILLER45 = "f" * 24_000  # ~6K tokens/message; tail >> budget
    msgs45.append({"role": "user", "content": UNIQUE45 + FILLER45})  # tail[0]
    for _ in range(2):
        msgs45.append({"role": "assistant", "content": FILLER45})
        msgs45.append({"role": "user", "content": FILLER45})
    msgs45.append({"role": "assistant", "content": FILLER45})
    out45 = e45.compress(msgs45, current_tokens=9_000)
    _all45 = "\n".join(str(m.get("content", "")) for m in out45)
    assert UNIQUE45 in _all45, "[45] unsummarized tail detail destroyed without an archive"
    est45 = mod.estimate_messages_tokens_rough(out45)
    assert est45 > 8_976, "[45] scenario lost its over-budget property"
    assert any("no verified transcript" in r for r in _records45), (
        "[45] refusal warning not emitted"
    )
    assert any("exceeds the context window" in r for r in _records45), (
        "[45] final-size postcondition did not report the overflow"
    )
finally:
    mod.logger.removeHandler(_h45)
print(f"[45] no-archive trim refused, tail intact (~{est45} tokens over budget, reported loudly) ✓")

# 46. Retention consolidation: a pruned INTERMEDIATE archive carries the only
# verbatim copy of the turns summarized out of it — its records must be
# consolidated into the chain root before unlinking.
e46 = mod.CompactEngine(context_length=1_000_000)
e46.transcript_enabled = True
e46.transcript_dir = tempfile.mkdtemp(prefix="compact-test-46-")
e46.transcript_retain = 2
paths46 = []
import time as _time46
for gen in range(1, 5):
    _p = e46._write_transcript([
        {"role": "system", "content": f"gen{gen}"},
        {"role": "user", "content": f"EXCLUSIVE-GEN{gen}-CONTENT"},
    ])
    paths46.append(_p)
    _time46.sleep(0.02)
e46._prune_old_transcripts()
assert "_consolidated_into" in json.loads(Path(paths46[1]).read_text()), "[46] intermediate archive not replaced with a redirect"
with open(paths46[0]) as _fh:  # chain root survives and carries gen-2's records
    _root46 = _fh.read()
assert "EXCLUSIVE-GEN1-CONTENT" in _root46, "[46] chain root lost its own content"
assert "EXCLUSIVE-GEN2-CONTENT" in _root46, "[46] pruned generation not consolidated into root"
assert "_consolidated_from" in _root46, "[46] consolidation marker missing"
_union46 = _root46
for _p in (paths46[2], paths46[3]):
    with open(_p) as _fh:
        _union46 += _fh.read()
for gen in range(1, 5):
    assert f"EXCLUSIVE-GEN{gen}-CONTENT" in _union46, f"[46] gen{gen} verbatim lost from every archive"
print("[46] pruned intermediates consolidated into the chain root (union covers all generations) ✓")

# 47. Quoted secret values may contain spaces — the quoted pattern must match
# through the CLOSING quote (the unquoted token pattern stops at whitespace
# and used to leak every word after the first, or miss short first words).
_s47a = mod._scrub_secrets('password="correct horse battery staple"')
assert "horse" not in _s47a and "staple" not in _s47a, f"[47] multi-word quoted password leaked: {_s47a}"
_s47b = mod._scrub_secrets('password="longfirstword rest of secret"')
assert "rest of secret" not in _s47b, f"[47] quoted password tail leaked: {_s47b}"
assert mod._scrub_secrets("password: 'hunter2 secure pass'") == "[REDACTED]", (
    "[47] single-quoted multi-word value not fully redacted"
)
assert "[REDACTED]" in mod._scrub_secrets('{"api_key": "qwertyuiop12345678"}'), (
    "[47] JSON quoted value regression"
)
assert "[REDACTED]" in mod._scrub_secrets("API_KEY=abcdefgh12345678"), (
    "[47] unquoted assignment regression"
)
print("[47] quoted (space-containing) and unquoted secret values fully redacted ✓")

# 48. Config ranges validate (a negative target_tokens used to flow to
# max_tokens=-150), and the threshold recomputes after a model switch EVEN
# WHEN the config load itself fails (it used to retain 200K on a 10K window).
_calls48 = []
def _tap48(**kwargs):
    _calls48.append(kwargs)
    return FakeResponse("## Goal\nRanges OK")
_cc["target_tokens"] = -100
mod.call_llm = _tap48
e48 = mod.CompactEngine(context_length=100_000)
e48.transcript_enabled = False
assert e48.target_tokens == 7000, f"[48] negative target_tokens accepted: {e48.target_tokens}"
e48.compress(msgs_chain, current_tokens=50_000)
assert _calls48 and _calls48[0]["max_tokens"] == 10_500, (
    f"[48] max_tokens not derived from validated target: {_calls48[0].get('max_tokens') if _calls48 else 'no call'}"
)
del _cc["target_tokens"]
import hermes_cli.config as _hcfg48
_orig_load48 = _hcfg48.load_config
_hcfg48.load_config = lambda: (_ for _ in ()).throw(RuntimeError("config unreadable"))
try:
    e48b = mod.CompactEngine(context_length=1_000_000)  # threshold 200_000
    assert e48b.threshold_tokens == 200_000
    e48b.update_model("m2", context_length=10_000)
    assert e48b.threshold_tokens == 2_000, (
        f"[48] stale threshold after model switch with failed config load: {e48b.threshold_tokens}"
    )
finally:
    _hcfg48.load_config = _orig_load48
    mod.call_llm = fake_call_llm
print("[48] config ranges validated; threshold recomputed despite failed config load ✓")

print("\nExisting 48 checks passed")

# Review regressions use actual config loading and the complete compress() output.

@contextmanager
def _review_case(**config):
    saved_cfg = FAKE_CONFIG["compact-context"].copy()
    saved_llm = mod.call_llm
    with tempfile.TemporaryDirectory(prefix="compact-review-") as directory:
        FAKE_CONFIG["compact-context"].clear()
        FAKE_CONFIG["compact-context"].update({
            "target_tokens": 500, "preserve_first_n": 3, "preserve_last_n": 6,
            "transcript_enabled": True, "transcript_dir": directory,
            "transcript_retain": 1,
        })
        FAKE_CONFIG["compact-context"].update(config)
        mod.call_llm = fake_call_llm
        try:
            yield mod.CompactEngine(context_length=10_000), Path(directory)
        finally:
            mod.call_llm = saved_llm
            FAKE_CONFIG["compact-context"].clear()
            FAKE_CONFIG["compact-context"].update(saved_cfg)

def _review_history():
    rows = [{"role": "system", "content": "System instructions"}]
    for i in range(10):
        rows += [{"role": "user", "content": f"question-{i}"},
                 {"role": "assistant", "content": f"answer-{i}"}]
    return rows

def _assert_review_protocol(rows):
    pending = set()
    for i, row in enumerate(rows):
        role = row["role"]
        if i and role in ("user", "assistant"):
            assert rows[i - 1]["role"] != role, "adjacent conversational roles"
        if role == "tool":
            assert row["tool_call_id"] in pending, "orphaned tool result"
            pending.remove(row["tool_call_id"])
        else:
            assert not pending, "summary interrupted pending tool results"
            pending = {c["id"] for c in row.get("tool_calls", [])}
    assert not pending, "unanswered tool calls"

def _review_session_isolation():
    for sid_a, sid_b in (
        ("alpha", "alpha_beta"),
        ("x" * 24 + "a", "x" * 24 + "b"),
        ("a/b", "a-b"),
        (None, None),  # engines without a host session ID remain isolated
    ):
        with _review_case() as (a, directory):
            a.on_session_start(sid_a)
            a.compress(_review_history(), current_tokens=5000)
            before = set(directory.glob("*.jsonl"))
            b = mod.CompactEngine(context_length=10_000)
            b.on_session_start(sid_b)
            private = _review_history()
            private[7]["content"] = "OTHER-SESSION-PRIVATE-DETAIL"
            b.compress(private, current_tokens=5000)
            b_files = set(directory.glob("*.jsonl")) - before
            b_bytes = {p: p.read_bytes() for p in b_files}
            a.compress(_review_history(), current_tokens=5000)
            assert b_files, "second session did not create its own archive"
            assert all(p.exists() and p.read_bytes() == data for p, data in b_bytes.items()), (
                "retention modified another session's archive"
            )
            assert all("OTHER-SESSION-PRIVATE-DETAIL" not in p.read_text()
                       for p in set(directory.glob("*.jsonl")) - b_files), (
                "retention copied another session's content"
            )

def _review_archive_pointers():
    with _review_case(preserve_last_n=2) as (engine, directory):
        engine.on_session_start("pointer-session")
        rows = _review_history()
        paths = []
        for generation in range(5):
            rows += [{"role": "user", "content": f"EXACT-GENERATION-{generation}"}]
            for i in range(4):
                rows += [{"role": "assistant", "content": f"reply-{generation}-{i}"},
                         {"role": "user", "content": f"next-{generation}-{i}"}]
            before = set(directory.glob("*.jsonl"))
            rows = engine.compress(rows, current_tokens=5000)
            paths.extend(set(directory.glob("*.jsonl")) - before)
        assert len(paths) == 5
        recovered = ""
        redirects = 0
        for path in paths:
            assert path.exists(), "a transcript pointer became dangling"
            first = json.loads(path.read_text().splitlines()[0])
            if "_consolidated_into" in first:
                redirects += 1
                target = Path(first["_consolidated_into"])
                assert target.exists(), "redirect target does not exist"
                recovered += target.read_text()
            else:
                recovered += path.read_text()
        assert redirects >= 2, "intermediate archives were not consolidated"
        for generation in range(5):
            assert f"EXACT-GENERATION-{generation}" in recovered
    # A failed atomic redirect replacement must leave the source readable.
    from unittest.mock import patch
    with _review_case() as (engine, directory):
        engine.on_session_start("failed-redirect")
        engine.compress(_review_history(), current_tokens=5000)
        before = set(directory.glob("*.jsonl"))
        engine.compress(_review_history(), current_tokens=5000)
        middle = (set(directory.glob("*.jsonl")) - before).pop()
        original = middle.read_bytes()
        with patch.object(mod.os, "replace", side_effect=OSError("simulated replacement failure")):
            engine.compress(_review_history(), current_tokens=5000)
        assert middle.read_bytes() == original, "failed replacement destroyed the source"
        assert not list(directory.glob(".compact-redirect-*")), "temporary redirect leaked"

def _review_mixed_quotes():
    fixtures = [
        'password="abc\'defghijkl" PUBLIC',
        'password="abcdefgh\'rest-of-secret" PUBLIC',
        "'password': 'abc\"defghijkl' PUBLIC",
        json.dumps({"password": 'abc"escaped-private-tail'}) + " PUBLIC",
        "password='abc\\'escaped-private-tail' PUBLIC",
    ]
    with _review_case() as (engine, _):
        for source in fixtures:
            mod.call_llm = lambda source=source, **kw: FakeResponse(source)
            out = engine.compress(_review_history(), current_tokens=5000)
            summary = next(r["content"] for r in out if r.get("_compressed_summary"))
            assert "defghijkl" not in summary and "rest-of-secret" not in summary
            assert "private-tail" not in summary, "escaped quote leaked the suffix"
            assert "[REDACTED]" in summary and "PUBLIC" in summary

def _review_final_budget():
    for position, short in ((2, False), (1, True), (-1, False)):
        with _review_case() as (engine, directory):
            rows = _review_history() if not short else _review_history()[:3]
            rows[position]["content"] = "HUGE-ARCHIVED-DETAIL " + "X" * 42000
            if position == -1:
                rows.append({"role": "user", "content": "LATEST-REQUEST " + "Y" * 42000})
            original = copy.deepcopy(rows)
            out = engine.compress(rows, current_tokens=mod.estimate_messages_tokens_rough(rows))
            assert mod.estimate_messages_tokens_rough(out) <= 8976, "final output exceeds budget"
            assert out[0]["content"].startswith("System instructions")
            assert rows == original, "input history was mutated"
            assert engine.compression_count == 1
            _assert_review_protocol(out)
            archived = "".join(p.read_text() for p in directory.glob("*.jsonl"))
            assert "HUGE-ARCHIVED-DETAIL" in archived
            assert any("compaction_transcript_" in str(r.get("content", "")) for r in out)
            if position == -1:
                assert out[-1]["role"] == "user" and "LATEST-REQUEST" in out[-1]["content"]
    with _review_case() as (engine, _):
        rows = _review_history()
        rows[0]["content"] = "SYSTEM MUST NOT BE TRUNCATED " + "Z" * 42000
        original = copy.deepcopy(rows)
        try:
            engine.compress(rows, current_tokens=11000)
        except ValueError as exc:
            assert "system" in str(exc).lower()
        else:
            raise AssertionError("an impossible system floor was reported as a successful compaction")
        assert rows == original and engine.compression_count == 0
    with _review_case(transcript_enabled=False) as (engine, _):
        rows = _review_history()
        rows[2]["content"] = "NO ARCHIVE " + "X" * 42000
        out = engine.compress(rows, current_tokens=11000)
        assert out is rows, "unrecoverable content must fail open"
        assert engine.compression_count == 0

    for role, position in (("system", 5), ("developer", -2)):
        with _review_case() as (engine, _):
            rows = _review_history()
            rows.insert(position, {"role": role, "content": "AUTHORITATIVE POLICY " + "Z" * 42000})
            original = copy.deepcopy(rows)
            try:
                engine.compress(rows, current_tokens=11000)
            except ValueError:
                pass
            else:
                raise AssertionError("late authoritative instruction was discarded")
            assert rows == original and engine.compression_count == 0
        with _review_case() as (engine, _):
            rows = _review_history()
            rows.insert(position, {"role": role, "content": "KEEP THIS POLICY"})
            out = engine.compress(rows, current_tokens=5000)
            assert sum(r.get("content") == "KEEP THIS POLICY" for r in out) == 1
            _assert_review_protocol(out)

    for role, position in (("developer", -3), ("system", 2)):
        with _review_case() as (engine, _):
            rows = _review_history()
            rows[position] = {"role": role, "content": "KEEP THIS POLICY"}
            _assert_review_protocol(rows)
            out = engine.compress(rows, current_tokens=5000)
            assert sum(r.get("content") == "KEEP THIS POLICY" for r in out) == 1
            _assert_review_protocol(out)

    with _review_case(transcript_enabled=False) as (engine, _):
        rows = _review_history()
        rows[2]["content"] = "X" * 37000
        for attempt in range(3):
            assert engine.compress(rows, current_tokens=9400) is rows
            assert engine._consecutive_failures == attempt + 1
        assert not engine.should_compress(prompt_tokens=9400), "oversized output never backs off"

_review_failures = []
for _number, _check in enumerate((
    _review_session_isolation, _review_archive_pointers,
    _review_mixed_quotes, _review_final_budget,
), 49):
    try:
        _check()
        print(f"[{_number}] {_check.__name__} passed")
    except AssertionError as exc:
        _review_failures.append(f"[{_number}] {_check.__name__}: {exc}")
assert not _review_failures, "\n".join(_review_failures)
print("\nALL 52 CHECKS PASSED")
