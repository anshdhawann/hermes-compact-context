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
out24c = e24c.compress(msgs24, current_tokens=250_000)
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

print("\nALL 25 CHECKS PASSED ✅")
