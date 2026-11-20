"""Offline check that the benchmark inputs match the evaluated requests of the fine-tuned critic."""
from hashlib import sha256

from .data import rows, unpack
from .providers import PROFILES, request
from .runner import wire

EXPECTED = {
    "muse": "4e4799b239492e7ee6945136b327cb1be9e644de4252f453f92d11fc57d37947",
}


def verify(source, revision):
    fingerprints = {key: [] for key in EXPECTED}
    image_count = 0
    for row in rows(source, revision):
        inputs = unpack(row)
        image_count += len(row["images"])
        for key, values in fingerprints.items():
            _, body = request(PROFILES[key], *inputs)
            values.append([row["id"], sha256(wire(body["messages"])).hexdigest()])
    for key, values in fingerprints.items():
        if len(values) != 400 or sha256(wire(values)).hexdigest() != EXPECTED[key]:
            raise ValueError(f"{key}: benchmark inputs differ from the evaluated requests")
    if image_count != 6936:
        raise ValueError("Unexpected image count")
    return {"actions": 400, "ordered_images": image_count, "verified": list(EXPECTED)}
