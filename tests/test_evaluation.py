import base64
import io
import json
from dataclasses import replace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from agentrobocritic import runner
from agentrobocritic.data import FEATURES, MUSE_SYSTEM, PROTOCOL, rows, unpack
from agentrobocritic.providers import PROFILES, read_answer, request
from agentrobocritic.scoring import parse, score


def target(outcome="achieved", mode=None, subtype=None):
    return dict(intent_success=outcome, failure_mechanism=mode, m1_subcategory=subtype,
                failure_reason="The object slipped." if outcome == "not_achieved" else None)


def human(outcome="achieved", mode=None, subtype=None):
    value = target(outcome, mode, subtype)
    del value["failure_reason"]
    return value


def image(color):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def packet(sid):
    contents = [{"type": "text", "text": "Before"}, {"type": "image", "text": None},
                {"type": "text", "text": "After"}, {"type": "image", "text": None}]
    return {"id": sid, "human_label": human(), "messages": [
        {"role": "system", "content": [{"type": "text", "text": PROTOCOL["system"]}]},
        {"role": "user", "content": contents}],
        "images": [{"path": None, "bytes": image(c)} for c in ("red", "blue")]}


@pytest.mark.parametrize("profile", ["qwen", "muse"])
def test_run_parquet_score_and_resume(profile, tmp_path, monkeypatch):
    source = tmp_path / "test-00000.parquet"
    data = [packet("action-A"), packet("action-B")]
    def write():
        pq.write_table(pa.Table.from_pylist(data, schema=FEATURES.arrow_schema), source)
    write()
    inference_rows = list(rows(source))
    assert list(inference_rows[0]) == ["id", "messages", "images"]
    seen = []
    model = PROFILES[profile]
    if model.key_env:
        monkeypatch.setenv(model.key_env, "test-key")
    def respond(req):
        assert req.url.path.endswith("/chat/completions")
        body = json.loads(req.content)
        seen.append(body)
        assert set(body["messages"][0]) == {"role", "content"}
        assert body["messages"][0]["content"] == (MUSE_SYSTEM if profile == "muse" else PROTOCOL["system"])
        user = body["messages"][1]["content"]
        assert [b["type"] for b in user] == ["text", "image_url", "text", "image_url"]
        for part, img in zip(user[1::2], data[0]["images"]):
            assert base64.b64decode(part["image_url"]["url"].split(",")[1]) == img["bytes"]
        assert "human_label" not in req.content.decode() and "action-A" not in req.content.decode()
        if profile == "muse":
            assert (body["temperature"], body["seed"], body["max_tokens"]) == (0, 42, 1024)
        else:
            assert body["response_format"]["json_schema"]["schema"] == PROTOCOL["schema"]
            assert body["reasoning_effort"] == "high" and body["max_completion_tokens"] == 8192
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(target())},
                                          "finish_reason": "stop"}], "usage": {"total_tokens": 15}})
    client = httpx.Client
    monkeypatch.setattr(runner.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    def run(limit=None, model=model):
        return runner.run(model, str(source), None, tmp_path / "out", 1, limit)
    assert run(1) == 1 and run() == 2 and run() == 2
    assert len(seen) == 2
    output = tmp_path / "out/predictions.jsonl"
    report = score(rows(source, labels_only=True), runner.records(output))
    assert report["overall"] == {"n": 2, "correct": 2, "accuracy": 1.0}
    with pytest.raises(ValueError, match="more actions"):
        run(1)
    assert len(list(runner.records(output))) == 2
    with pytest.raises(ValueError, match="another model"):
        run(model=replace(model, tokens=100))
    data[0]["messages"][1]["content"][0]["text"] += " revised"
    write()
    with pytest.raises(ValueError, match="Request changed"):
        run()
    assert len(seen) == 2


@pytest.mark.parametrize("profile", ["anthropic", "openai", "gemini", "kimi", "glm", "deepseek", "fireworks", "chat"])
def test_native_envelopes(profile):
    path, body = request(PROFILES[profile], *unpack(packet("hidden-action-id")))
    if profile != "fireworks":
        limits = body.get("generationConfig", body)
        assert [v for k, v in limits.items() if k in {
            "max_tokens", "max_output_tokens", "max_completion_tokens", "maxOutputTokens"}] == [
            65536 if profile == "deepseek" else 8192]
    raw = json.dumps(body)
    assert "hidden-action-id" not in raw and "human_label" not in raw
    assert raw.count(base64.b64encode(image("blue")).decode()) == 1
    if profile == "openai":
        assert path == "/responses" and body["text"]["format"]["strict"] and body["store"] is False
    elif profile == "anthropic":
        assert path == "/messages" and body["output_config"]["effort"] == "high"
    elif profile == "gemini":
        assert path.endswith(":generateContent") and body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
    elif profile == "fireworks":
        assert body["max_tokens"] == 1024
        assert body["context_length_exceeded_behavior"] == "error" and body["messages"][0]["content"] == MUSE_SYSTEM
    else:
        assert path == "/chat/completions"


def test_response_boundaries():
    text = json.dumps(target())
    msg = {"message": {"content": text}, "finish_reason": "length"}
    assert read_answer("qwen", {"choices": [msg]})["prediction"] is None
    assert read_answer("muse", {"choices": [msg]})["prediction"] == target()
    assert read_answer("responses", {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "refusal", "refusal": "declined"}, {"type": "output_text", "text": text}]}]})["error"]
    assert read_answer("gemini", {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "thinking", "thought": True}, {"text": text[:20]}, {"text": text[20:]}]}}]})["prediction"] == target()
    assert parse(json.dumps(target("unverifiable")), binary=True)[1]
    assert parse(json.dumps(target("unverifiable")), binary=False)[0]
    assert parse(json.dumps(target("not_achieved", "M1")), binary=True)[1]
    assert parse('{"intent_success":true,"intent_success":false}', binary=True)[1]
    assert parse("unfinished", binary=False)[1]


def test_denominators_and_alignment():
    refs = [{"id": str(i), "human_label": lbl} for i, lbl in enumerate([
        human(), human("not_achieved", "M1", "S4"), human("not_achieved", "M4"), human()])]
    preds = [{"id": str(i), "prediction": val, "error": None if val else "invalid_json"} for i, val in enumerate([
        target(), target("not_achieved", "M1", "S2"), target("not_achieved", "M5"), None])]
    report = score(refs, reversed(preds))
    assert report["outcome"]["correct"] == 3 and report["outcome"]["kappa"] == pytest.approx(0.6)
    assert report["failure_mode"] == {"n": 2, "correct": 1, "accuracy": .5, "kappa": pytest.approx(1/3)}
    assert report["overall"]["correct"] == 2 and report["invalid_predictions"] == 1
    for invalid in (preds[1:], preds + [preds[0]], preds + [{**preds[0], "id": "extra"}]):
        with pytest.raises(ValueError):
            score(refs, invalid)
    with pytest.raises(ValueError, match="human label"):
        score([{"id": "0", "human_label": None}], preds[:1])


def test_alignment_rejects_extra_and_missing_images():
    p = packet("a")
    p["images"].append(p["images"][0])
    with pytest.raises(ValueError, match="count"):
        unpack(p)
    p["images"] = p["images"][:1]
    with pytest.raises(ValueError, match="embedded PNG"):
        unpack(p)
