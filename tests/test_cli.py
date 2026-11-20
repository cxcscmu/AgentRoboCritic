import json
import sys

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agentrobocritic.cli import main
from agentrobocritic.data import FEATURES
from agentrobocritic.runner import records
from test_evaluation import packet, target


@pytest.mark.parametrize("shell_key,token_args,expected_tokens", [
    (None, [], 8192),
    ("test-shell-key", ["--max-tokens", "4096"], 4096),
])
def test_api_key_from_dotenv_and_shell_override(tmp_path, monkeypatch, shell_key, token_args, expected_tokens):
    subdir = tmp_path / "work"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    (tmp_path / ".env").write_text("MOONSHOT_API_KEY='test-file-key'\n")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    if shell_key:
        monkeypatch.setenv("MOONSHOT_API_KEY", shell_key)
    dataset = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist([packet("one")], schema=FEATURES.arrow_schema), dataset)
    out = tmp_path / "results"
    def send(request):
        assert request.headers["Authorization"] == "Bearer " + (shell_key or "test-file-key")
        assert json.loads(request.content)["max_completion_tokens"] == expected_tokens
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(target())},
                                                     "finish_reason": "stop"}]})
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(send), **kw))
    monkeypatch.setattr(sys, "argv", ["arc", "evaluate", "--profile", "kimi", "--out", str(out),
                                      "--data", str(dataset), *token_args])
    main()
    assert [r["id"] for r in records(out / "predictions.jsonl")] == ["one"]
    assert json.loads((out / "run.json").read_text())["profile"]["tokens"] == expected_tokens


def test_portable_cli_resumes_after_service_error(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "test-tiny.parquet"
    pq.write_table(pa.Table.from_pylist([packet("one"), packet("two")], schema=FEATURES.arrow_schema), dataset)
    out = tmp_path / "results"
    calls = []
    def send(request):
        calls.append(request)
        if len(calls) == 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(target())},
                                                     "finish_reason": "stop"}]})
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(send), **kw))
    args = ["arc", "evaluate", "--data", str(dataset), "--profile", "muse", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(httpx.HTTPStatusError):
        main()
    assert not (out / "predictions.jsonl").exists()
    main()
    assert len(calls) == 3 and [p["id"] for p in records(out / "predictions.jsonl")] == ["one", "two"]
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["arc", "score", "--data", str(dataset),
                                     "--predictions", str(out / "predictions.jsonl")])
    main()
    assert json.loads(capsys.readouterr().out)["overall"]["correct"] == 2
