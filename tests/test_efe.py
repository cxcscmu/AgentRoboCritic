import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import av
from datasets import Features
import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from agentrobocritic.data import FEATURES, rows, unpack
from agentrobocritic.efe import convert_trial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "efe"))
import arc_capture
from instrument import instrument
from prepare_task import prepare


SERVER = """
def do_submit(meta, A):
    calls.append("submit")
    A[0, 0] = -1
    if meta.get("crash"):
        raise RuntimeError("simulator stopped")
    return {"chunk_id":len(calls), "executed_rows":meta.get("executed", 1)}, views(len(calls))

def do_observe(meta):
    calls.append("observe")
    return {"obs_id":len(calls)}, views(len(calls))
"""
CLIENT = '''
class Robot:
    def submit(self, action_type, actions, note=""):
        """Send an action."""
        return send(note)
'''


def array(name, value):
    return name, "array", value.tobytes(), {"shape": list(value.shape), "dtype": str(value.dtype)}


def views(call):
    stack = np.full((3, 4, 4, 3), 20 * call, np.uint8)
    stack[:, :, :, 0] += np.arange(3, dtype=np.uint8)[:, None, None]
    return [array("frames/wrist", stack), array("last/wrist", stack[-1]),
            array("frame_step/wrist", np.arange(3) + 10 * call)]


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(arc_capture, "ROOT", tmp_path / "arc")
    scope = {"calls": [], "views": views}
    exec(instrument(SERVER, "server"), scope)
    return scope


@pytest.mark.parametrize("note", [None, "", " \n ", 42, "n"*501])
def test_refuse_before_motion_on_both_sides(native, note):
    sent = []
    client = {"send": sent.append}
    exec(instrument(CLIENT, "client"), client)
    with pytest.raises(ValueError):
        client["Robot"]().submit("joint_position", [[0]], note=note)
    with pytest.raises(ValueError):
        native["do_submit"]({"note": note}, np.zeros((2, 1)))
    assert sent == native["calls"] == []
    assert [p.name for p in arc_capture.ROOT.iterdir()] == ["rejections.jsonl"]


def test_native_method_bad_request(native):
    class Refused(Exception):
        code = 400
    @contextmanager
    def bad_request():
        try:
            yield
        except ValueError as exc:
            raise Refused from exc
    source = '''
class Service:
    def _submit(self, meta, parts):
        return {"chunk_id":1, "executed_rows":1}, views(1)
    def _observe(self, meta, parts):
        return {"obs_id":2}, views(2)
'''
    env = {"bad_request": bad_request, "views": views}
    exec(instrument(source, "service"), env)
    sim = env["Service"]()
    with pytest.raises(Refused):
        sim._submit({}, {})
    out, _ = sim._submit({"note": "Reach toward the cup."}, {"actions": {"data": [[2, 3]]}})
    assert out["note"] == "Reach toward the cup."
    assert json.loads((arc_capture.ROOT / "00000001/call.json").read_text())["command"]["actions"] == [[2, 3]]


def test_collect_convert_label_and_align(native, tmp_path):
    observe, submit = native["do_observe"], native["do_submit"]
    note = "  Lower the gripper toward the mug handle. "
    observe({})
    commands = np.array([[0.25], [0.5]])
    result, _ = submit({"note": note, "type": "joint_position", "dt": 0.05}, commands)
    observe({})
    submit({"note": "Open the gripper.", "executed": 0}, np.zeros((1, 1)))
    assert result["note"] == note and commands[0, 0] == -1
    out = tmp_path / "test-actions.parquet"
    assert convert_trial(tmp_path, "episode/1", "Pick up the mug", None, out) == 1
    assert Features.from_arrow_schema(pq.read_schema(out)) == FEATURES
    row, = rows(out)
    assert row["id"] == "efe/episode%2F1/2"
    text = row["messages"][1]["content"]
    header = json.loads(text[0]["text"].split("\n", 1)[1])
    assert header["original_note"] == note and header["command"]["actions"] == [[.25], [.5]]
    assert header["execution"]["executed_rows"] == 1
    captions = [json.loads(part["text"])["image_contexts"][0] for part in text[1::2]]
    assert [(c["native_call_id"], c["control_step"]) for c in captions] == [(1,12), (2,20), (2,21), (2,22), (3,32)]
    assert captions[-1]["temporal_role"] == "following observation"
    assert [Image.open(io.BytesIO(im["bytes"])).getpixel((0,0))[0] for im in row["images"]] == [22,40,41,42,62]
    assert len(unpack(row)[1]) == 11
    assert list(rows(out, labels_only=True))[0]["human_label"] is None
    label = {"intent_success": "achieved", "failure_mechanism": None, "m1_subcategory": None}
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": row["id"], "human_label": label}))
    labeled = tmp_path / "test-labeled.parquet"
    convert_trial(tmp_path, "episode/1", "Pick up the mug", labels, labeled)
    assert list(rows(labeled, labels_only=True))[0]["human_label"] == label
    labels.write_text(json.dumps({"id": "other-id", "human_label": label}))
    with pytest.raises(ValueError, match="No reference"):
        convert_trial(tmp_path, "episode/1", "Pick up the mug", labels, tmp_path / "bad.parquet")
    assert not (tmp_path / "bad.parquet").exists()


def test_capture_video_default(native, tmp_path):
    stream = io.BytesIO()
    with av.open(stream, mode="w", format="mp4") as movie:
        track = movie.add_stream("mpeg4", rate=10)
        track.width = track.height = 16
        track.pix_fmt = "yuv420p"
        for v in (20, 90, 150):
            frame = av.VideoFrame.from_ndarray(np.full((16,16,3), v, np.uint8), format="rgb24")
            for pkt in track.encode(frame):
                movie.mux(pkt)
        for pkt in track.encode():
            movie.mux(pkt)
    native["views"] = lambda call: [("video/top", "mp4", stream.getvalue(), {"n":3}),
                                    array("last/top", np.full((16,16,3), 159, np.uint8))]
    native["do_submit"]({"note": "Approach the handle."}, np.ones((1,1)))
    out = tmp_path / "test-video.parquet"
    convert_trial(tmp_path, "trial", "Open the drawer", None, out)
    row, = rows(out)
    values = [Image.open(io.BytesIO(i["bytes"])).getpixel((0,0))[0] for i in row["images"]]
    assert len(values) == 3 and values[0] < values[1] < values[2] == 159


def test_failed_capture_refuses_partial_export(native, tmp_path):
    with pytest.raises(RuntimeError, match="simulator stopped"):
        native["do_submit"]({"note": "Lift the mug.", "crash": True}, np.zeros((1,1)))
    with pytest.raises(ValueError, match="Incomplete"):
        convert_trial(tmp_path, "episode", "Lift the mug", None, tmp_path / "bad.parquet")
    assert not (tmp_path / "bad.parquet").exists()


def test_prepare_preserves_original_task(tmp_path):
    task = tmp_path / "yamlab/demo-standard"
    (task / "environment/sim").mkdir(parents=True)
    for p in [task / "environment", task / "environment/sim"]:
        (p / "Dockerfile").write_text("FROM python:3.12\n")
    (task / "task.toml").write_text("artifacts = []\n")
    (task / "instruction.md").write_text('Pick up the mug.\nr.submit("joint_position", A)\n')
    dest = tmp_path / "new/demo-standard"
    prepare(task, dest)
    assert "note=local_intent" in (dest / "instruction.md").read_text()
    assert "note=" not in (task / "instruction.md").read_text()
    assert "/data/arc/" in (dest / "task.toml").read_text()
    assert "server /srv/server.py" in (dest / "environment/sim/Dockerfile").read_text()
    with pytest.raises(ValueError):
        prepare(task, dest)
