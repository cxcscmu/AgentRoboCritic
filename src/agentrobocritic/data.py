import base64
import json
from pathlib import Path

import datasets
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DATASET = "cx-cmu/AgentRoboCritic-Bench"
REVISION = "78e545f29488525560bbc35676aac71adef538c3"
PROMPTS = Path(__file__).with_name("prompts")
PROTOCOL = json.loads((PROMPTS / "benchmark.json").read_text())
MUSE_SYSTEM = (PROMPTS / "muse.txt").read_text()
PNG = b"\x89PNG\r\n\x1a\n"
LABEL_KEYS = ("intent_success", "failure_mechanism", "m1_subcategory")
FEATURES = datasets.Features({
    "id": datasets.Value("string"),
    "messages": datasets.List({"role": datasets.Value("string"),
        "content": datasets.List({"type": datasets.Value("string"), "text": datasets.Value("string")})}),
    "images": datasets.List(datasets.Image()),
    "human_label": {key: datasets.Value("string") for key in LABEL_KEYS},
})


def rows(source, revision=None, *, labels_only=False):
    root = Path(source)
    if not root.exists():
        root = Path(snapshot_download(source, repo_type="dataset", revision=revision,
                                      allow_patterns=["data/test-*.parquet"]))
    if root.is_file():
        paths = [root]
    else:
        directory = root / "data" if (root / "data").is_dir() else root
        paths = sorted(directory.glob("test-*.parquet"))
    if not paths:
        raise ValueError(f"No test Parquet shards under {root}")
    seen = set()
    for path in paths:
        with pq.ParquetFile(path) as shard:
            cols = ["id", "human_label"] if labels_only else ["id", "messages", "images"]
            for batch in shard.iter_batches(batch_size=4, columns=cols):
                for row in batch.to_pylist():
                    sid = row["id"]
                    if not isinstance(sid, str) or not sid or sid in seen:
                        raise ValueError(f"Empty or repeated action ID in {path}")
                    seen.add(sid)
                    yield row


def unpack(row):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("Expected system and user turns")
    sys_parts = messages[0]["content"]
    if len(sys_parts) != 1 or sys_parts[0]["type"] != "text" or not isinstance(sys_parts[0]["text"], str):
        raise ValueError("Expected a text system prompt")
    images = iter(row["images"])
    parts = []
    for block in messages[1]["content"]:
        if block["type"] == "text" and isinstance(block["text"], str):
            parts.append(("text", block["text"]))
        elif block["type"] == "image":
            image = next(images, None)
            if image is None or not (image.get("bytes") or b"").startswith(PNG):
                raise ValueError("Image marker requires an embedded PNG")
            parts.append(("image", base64.b64encode(image["bytes"]).decode("ascii")))
        else:
            raise ValueError("Unknown multimodal content block")
    if next(images, None) is not None or not row["images"]:
        raise ValueError("Image count differs from prompt")
    return sys_parts[0]["text"], parts
