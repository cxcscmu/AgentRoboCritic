"""Convert completed native EFE call records into benchmark-format multimodal actions."""
import io
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .data import FEATURES, PROTOCOL
from .scoring import validate_label


def recordings(trial):
    roots = [trial] if trial.name == "arc" else [p for p in trial.rglob("arc") if p.is_dir()]
    if len(roots) != 1:
        raise ValueError("Supply the arc artifact directory from one completed trial")
    root = roots[0]
    result = []
    for folder in sorted(root.iterdir()):
        if folder.name == "rejections.jsonl":
            continue
        if not folder.is_dir() or not re.fullmatch(r"\d{8,}", folder.name):
            raise ValueError(f"Incomplete or unexpected EFE capture: {folder.name}")
        record = json.loads((folder / "call.json").read_text())
        if type(record["call"]) is not int or record["call"] != int(folder.name) or record["kind"] not in ("observe", "submit"):
            raise ValueError(f"Invalid native call identity: {folder.name}")
        result.append((folder, record))
    return sorted(result, key=lambda entry: entry[1]["call"])


def video_samples(path, end_only):
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install agentrobocritic[efe] to read native MP4 observations") from exc
    # Some EFE encoders omit the stream frame count. Count those clips first to
    # select the same indices for video and raw arrays without buffering a movie.
    with av.open(str(path)) as clip:
        n = clip.streams.video[0].frames or sum(1 for _ in clip.decode(video=0))
    selected = {n - 1} if end_only else {0, (n - 1) // 2, n - 1}
    with av.open(str(path)) as clip:
        frames = {i: f.to_ndarray(format="rgb24") for i, f in enumerate(clip.decode(video=0)) if i in selected}
    if len(frames) != len(selected):
        raise ValueError(f"Incomplete camera video at {path.name}")
    return n, frames


def visuals(entry, role):
    folder, record = entry
    parts = {}
    for part in record["parts"]:
        name, filename = part["name"], part["file"]
        if name in parts:
            raise ValueError("Repeated native image part")
        if name.startswith("video/") and re.fullmatch(r"part-\d+\.mp4", filename):
            parts[name] = folder / filename
        elif re.fullmatch(r"part-\d+\.npy", filename):
            parts[name] = np.load(folder / filename, mmap_mode="r", allow_pickle=False)
        else:
            raise ValueError("Invalid native array or video binding")
    cameras = dict.fromkeys(name.split("/", 1)[1] for name in parts if name.startswith(("last/", "frames/", "video/")))
    for camera in cameras:
        stack, last = parts.get("frames/" + camera), parts.get("last/" + camera)
        video = parts.get("video/" + camera)
        end_only = role != "action"
        if stack is not None and len(stack):
            n = len(stack)
            indices = [n - 1] if end_only else sorted({0, (n - 1) // 2, n - 1})
            frames = {i: stack[i] for i in indices}
        elif video is not None and (last is None or not end_only):
            n, frames = video_samples(video, end_only)
        else:
            n, frames = 1, {}
        if last is not None:
            frames[n - 1] = last
        for i, frame in sorted(frames.items()):
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                raise ValueError("Expected uint8 RGB or RGBA observation")
            end = i == n - 1
            timing = {}
            for stem, key in (("frame_step", "control_step"), ("frame_time_s", "sim_time_s")):
                values = parts.get(stem + "/" + camera, parts.get(stem))
                if values is not None and values.size and (end or values.size == n):
                    timing[key] = values.reshape(-1)[-1 if end else i].item()
                elif end:
                    field = "control_steps" if key == "control_step" else key
                    if field in record["execution"]:
                        timing[key] = record["execution"][field]
            temporal = ("action end" if end else "during action") if role == "action" else role
            caption = {"camera": camera, "native_call_id": record["call"], **timing, "temporal_role": temporal}
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG")
            yield caption, {"path": None, "bytes": buf.getvalue()}


def adjacent(left, right):
    return (left["call"] + 1 == right["call"] and
            all(left["execution"].get(k) == right["execution"].get(k) for k in ("episode", "resets")))


def convert_trial(trial, episode, task, label_file, output):
    if not episode.strip() or not task.strip() or output.suffix != ".parquet" or output.exists():
        raise ValueError("Supply an episode ID, task, and new .parquet destination")
    labels = {}
    if label_file:
        for line in label_file.read_text().splitlines():
            obj = json.loads(line)
            validate_label(obj["human_label"])
            if obj["id"] in labels:
                raise ValueError("Duplicate reference ID")
            labels[obj["id"]] = obj["human_label"]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        partial = Path(temp) / "actions.parquet"
        with pq.ParquetWriter(partial, FEATURES.arrow_schema, compression="zstd") as writer:
            entries = recordings(trial)
            for index, entry in enumerate(entries):
                _, rec = entry
                if rec["kind"] != "submit":
                    continue
                executed = rec["execution"].get("executed_rows")
                if type(executed) is not int or executed < 0:
                    raise ValueError(f"Missing execution count for call {rec['call']}")
                if executed == 0:
                    continue
                if not isinstance(rec["note"], str) or not rec["note"].strip():
                    raise ValueError("Executed action has no original intent")
                sid = f"efe/{quote(episode, safe='')}/{rec['call']}"
                header = {"task": task, "original_note": rec["note"], "intent_basis": "verbatim_original_note",
                          "command": rec["command"], "execution": rec["execution"]}
                user = [{"type": "text", "text": "Assess the designated action using this input and the ordered images. "
                         "Return only the four-field JSON target.\n" + json.dumps(header, ensure_ascii=False)}]
                current = list(visuals(entry, "action"))
                if not current:
                    raise ValueError(f"No RGB evidence for executed action {sid}")
                evidence = []
                if index and adjacent(entries[index - 1][1], rec):
                    evidence.extend(visuals(entries[index - 1], "preceding call end"))
                evidence.extend(current)
                if index + 1 < len(entries):
                    follow = entries[index + 1]
                    if follow[1]["kind"] == "observe" and adjacent(rec, follow[1]):
                        evidence.extend(visuals(follow, "following observation"))
                images = []
                for caption, image in evidence:
                    user.extend([{"type": "text", "text": json.dumps({"image_contexts": [caption]})},
                                 {"type": "image", "text": None}])
                    images.append(image)
                if label_file and sid not in labels:
                    raise ValueError(f"No reference label for {sid}")
                label = labels.pop(sid) if label_file else None
                row = {"id": sid, "messages": [
                    {"role": "system", "content": [{"type": "text", "text": PROTOCOL["system"]}]},
                    {"role": "user", "content": user}], "images": images, "human_label": label}
                writer.write_table(pa.Table.from_pylist([row], schema=FEATURES.arrow_schema))
                count += 1
        if not count or labels:
            raise ValueError("No executed actions or unmatched label IDs")
        os.link(partial, output)
    return count
