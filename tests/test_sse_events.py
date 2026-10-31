"""SSE structured-event contract tests.

Verifies that `_sse_stream` emits tool activity as structured `delta.agent_event`
(paired `tool_call`/`tool_result` sharing an `id`) and keeps `delta.content` for the
real answer only — never mixing the two in one chunk.
"""
import json

import app as appmod


class _FakeChunk:
    def __init__(self, text):
        self.metadata = {"title": "Doc"}
        self.document_id = "doc123"
        self.text = text


class _FakeSource:
    def __init__(self):
        self.score = 0.5
        self.chunk = _FakeChunk("x")


class _FakeResponse:
    def __init__(self, answer):
        self.answer = answer
        self.sources = []
        self.prompt = "Agent ran 1 iterations\n[search] context"
        self.tool_calls = [{"tool": "search", "args": {}, "result_preview": "x"}]
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _FakeQuerier:
    """Scripts a realistic agent event sequence (2 tools, then a streamed answer)."""

    def ask_stream(self, question, mode=None, history=None):
        yield {"type": "thinking", "content": "Iteration 1…"}
        yield {"type": "tool_call", "tool": "read_summaries", "args": {}}
        yield {"type": "tool_result", "tool": "read_summaries", "preview": "Group L1: …", "length": 4068}
        yield {"type": "tool_call", "tool": "search", "args": {"query": "IA", "top_k": 8}}
        yield {"type": "tool_result", "tool": "search", "preview": "[score=0.55] …", "length": 4689}
        yield {"type": "answer_delta", "content": "Voici "}
        yield {"type": "answer_delta", "content": "la réponse."}
        yield {"type": "answer", "response": _FakeResponse("Voici la réponse.")}


def _parse(monkeypatch, judge=("default",), log_ok=True, logged_rows=None):
    """Run the stream with stubbed querier/judge/logging. `judge`: dict result, None, or
    'raise' to simulate a judge failure. Default is a high-confidence result."""
    monkeypatch.setattr(appmod, "get_querier", lambda model: _FakeQuerier())

    if judge == ("default",):
        judge = {"score": 95, "unsure": False, "reason": "ok"}

    def _fake_judge(question, answer, context, tools):
        if judge == "raise":
            raise RuntimeError("judge down")
        return judge

    monkeypatch.setattr(appmod, "_judge_confidence", _fake_judge)

    def _fake_log(row):
        if logged_rows is not None:
            logged_rows.append(row)
        return log_ok

    monkeypatch.setattr(appmod, "_log_unsure", _fake_log)

    deltas = []
    for raw in appmod._sse_stream("q", [], appmod.DEFAULT_MODEL, session_id="sess-1"):
        assert raw.startswith("data: ")
        body = raw[len("data: "):].strip()
        if body == "[DONE]":
            continue
        deltas.append(json.loads(body)["choices"][0]["delta"])
    return deltas


def test_tool_events_are_structured_and_paired(monkeypatch):
    deltas = _parse(monkeypatch)
    events = [d["agent_event"] for d in deltas if "agent_event" in d]

    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["tool"] for c in calls] == ["read_summaries", "search"]
    assert [r["tool"] for r in results] == ["read_summaries", "search"]

    # Each tool_call id is matched by a tool_result with the same id, in order.
    assert [c["id"] for c in calls] == [r["id"] for r in results]
    assert all(e["id"].startswith("tc_") for e in calls + results)

    # tool_call carries args; tool_result carries status/length.
    assert calls[1]["args"] == {"query": "IA", "top_k": 8}
    assert results[0]["length"] == 4068
    assert all(r["status"] == "ok" for r in results)

    # seq is strictly monotonic across events.
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_tool_chunks_have_no_content_and_answer_has_no_event(monkeypatch):
    deltas = _parse(monkeypatch)
    for d in deltas:
        # A single chunk is never both a tool event and answer content.
        assert not ("agent_event" in d and d.get("content"))

    # The concatenated answer content is pure — no tool markdown leaked in.
    answer = "".join(d.get("content", "") for d in deltas if "agent_event" not in d)
    assert "Voici la réponse." in answer
    assert "🔍" not in answer and "> 📄" not in answer and "---" not in answer


def test_confidence_event_always_emitted(monkeypatch):
    deltas = _parse(monkeypatch)  # default high-confidence judge
    events = [d["agent_event"] for d in deltas if "agent_event" in d]
    conf = [e for e in events if e["type"] == "confidence"]
    assert len(conf) == 1
    assert conf[0]["score"] == 95
    assert conf[0]["unsure"] is False
    assert conf[0]["logged"] is False
    # No low-confidence warning in content when score is high.
    answer = "".join(d.get("content", "") for d in deltas if "agent_event" not in d)
    assert "Confiance faible" not in answer
    # confidence chunk carries no content.
    assert all(not (d.get("agent_event", {}).get("type") == "confidence" and d.get("content"))
               for d in deltas)


def test_low_confidence_logs_and_warns(monkeypatch):
    rows = []
    deltas = _parse(
        monkeypatch,
        judge={"score": 20, "unsure": False, "reason": "insufficient context"},
        log_ok=True,
        logged_rows=rows,
    )
    conf = [d["agent_event"] for d in deltas
            if d.get("agent_event", {}).get("type") == "confidence"][0]
    assert conf["score"] == 20
    assert conf["unsure"] is True          # score < threshold → unsure_final
    assert conf["logged"] is True

    # A user-facing warning is streamed in content.
    answer = "".join(d.get("content", "") for d in deltas if "agent_event" not in d)
    assert "Confiance faible (20/100)" in answer
    assert "enregistrée" in answer

    # One row was logged with the expected fields.
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-1"
    assert rows[0]["score"] == 20
    assert rows[0]["unsure"] is True


def test_low_confidence_log_failure_is_reported(monkeypatch):
    deltas = _parse(
        monkeypatch,
        judge={"score": 10, "unsure": True, "reason": "no answer"},
        log_ok=False,
    )
    conf = [d["agent_event"] for d in deltas
            if d.get("agent_event", {}).get("type") == "confidence"][0]
    assert conf["logged"] is False
    answer = "".join(d.get("content", "") for d in deltas if "agent_event" not in d)
    assert "échec de journalisation" in answer


def test_judge_failure_does_not_crash_and_emits_no_confidence(monkeypatch):
    for bad in (None, "raise"):
        deltas = _parse(monkeypatch, judge=bad)
        events = [d["agent_event"] for d in deltas if "agent_event" in d]
        assert not any(e["type"] == "confidence" for e in events)
        # Answer still fully streamed.
        answer = "".join(d.get("content", "") for d in deltas if "agent_event" not in d)
        assert "Voici la réponse." in answer
