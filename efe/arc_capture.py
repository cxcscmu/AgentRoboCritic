"""Installed in the simulator image. Capture a native call only at its execution boundary."""
import copy
import inspect
import json
import sys
import tempfile
from contextlib import nullcontext
from functools import wraps
from pathlib import Path

ROOT = Path("/data/arc")


def require_note(note):
    if not isinstance(note, str) or not note.strip() or len(note) > 500:
        raise ValueError("submit requires a nonblank prospective note of at most 500 characters")
    return note


def rejected():
    ROOT.mkdir(exist_ok=True, parents=True)
    with (ROOT / "rejections.jsonl").open("a") as f:
        f.write('{"event":"submit_rejected","reason":"note_required"}\n')


def capture(kind, *, method=False, bad_request=nullcontext):
    def decorate(fn):
        signature = inspect.signature(fn)
        @wraps(fn)
        def wrapped(*args, **kwargs):
            import numpy as np
            args_by_name = signature.bind(*args, **kwargs).arguments
            meta = args_by_name["meta"]
            command = None
            if kind == "submit":
                with bad_request():
                    try:
                        note = require_note(meta.get("note"))
                    except ValueError:
                        rejected()
                        raise
                if method:
                    parts = args_by_name["parts"]
                    actions = parts["actions"]["data"] if "actions" in parts else meta.get("actions")
                else:
                    actions = args_by_name["A"]
                # Copy before the simulator can clip, cast or otherwise alter the request.
                command = copy.deepcopy(actions)
            ROOT.mkdir(exist_ok=True, parents=True)
            pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=ROOT))
            try:
                out, blobs = fn(*args, **kwargs)
            except Exception as exc:
                # EFE maps these validation exceptions to a refused request. Preserve an
                # unfinished marker if the simulator failed at an unknown execution point.
                if ((not method and isinstance(exc, ValueError)) or
                    (method and getattr(exc, "code", None) in (400, 409))):
                    pending.rmdir()
                raise
            # A recorder error after physics must never cause the caller to repeat motion.
            # An unfinished directory makes the offline converter reject the trace.
            try:
                call = out["chunk_id" if kind == "submit" else "obs_id"]
                if type(call) is not int or call < 0:
                    raise ValueError("Unexpected native call ID")
                state = {k: out[k] for k in ("executed_rows", "control_steps_chunk",
                         "control_steps_watch", "sim_time_s", "control_steps", "resets", "episode") if k in out}
                if type(out.get("stopped")) is bool:
                    state["stopped"] = out["stopped"]
                record = {"kind": kind, "call": call, "execution": state, "parts": []}
                if kind == "submit":
                    out["note"] = note
                    record["note"] = note
                    record["command"] = {k: meta[k] for k in ("type", "dt", "interpolate") if k in meta}
                    if "type" not in meta and "type" in out:
                        record["command"]["type"] = out["type"]
                    record["command"]["actions"] = np.asarray(command).tolist()
                raw_stacks = {name[7:] for name, encoding, _, _ in blobs
                              if name.startswith("frames/") and encoding == "array"}
                for i, (name, encoding, raw, attrs) in enumerate(blobs):
                    stem = name.split("/", 1)[0]
                    if encoding == "mp4" and name.startswith("video/") and name[6:] not in raw_stacks:
                        filename = f"part-{i}.mp4"
                        (pending / filename).write_bytes(raw)
                    elif encoding == "array" and stem in ("last", "frames", "frame_step", "frame_time_s"):
                        array = np.frombuffer(raw, dtype=attrs["dtype"]).reshape(attrs["shape"])
                        filename = f"part-{i}.npy"
                        np.save(pending / filename, array, allow_pickle=False)
                    else:
                        continue
                    record["parts"].append({"name": name, "file": filename})
                (pending / "call.json").write_text(json.dumps(record, ensure_ascii=False, allow_nan=False))
                pending.rename(ROOT / f"{call:08d}")
            except Exception as exc:
                print(f"AgentRoboCritic capture incomplete at {pending.name}: {exc}", file=sys.stderr, flush=True)
            return out, blobs
        return wrapped
    return decorate
