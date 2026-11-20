"""Create an intent-grounded copy of one upstream EFE Standard task."""
import argparse
from pathlib import Path
import re
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
SERVICES = {
    "robotwin-2": "server", "yamlab": "server", "robolab": "server", "robodojo": "server",
    "robopaint": "server", "behavior-1k": "server", "maniguard": "server",
    "libero-max": "service", "bigym2": "service", "kinder": "service",
    "pandadyn": "service", "vlabench": "service",
}
ACTOR = """
## Local intent for every action

Before submitting each action chunk, pass a nonblank `note` (at most 500 characters).
State the intended action, its target, and the immediate observable objective of
this chunk. An intermediate approach, grasp, or lift can have its own objective.
Write the note prospectively. A submit without a note is refused before motion.
Use the simulator's original action types, dimensions and units.
In the examples, set `local_intent` in your own words for each new chunk.
Request `video="frames"` when you want intermediate camera observations.
"""


def prepare(task, dest):
    task = task.resolve()
    suite = task.parent.name
    if not task.name.endswith("-standard") or suite not in SERVICES:
        raise ValueError(f"Expected an EFE Standard task in a supported suite: {', '.join(sorted(SERVICES))}")
    if dest.exists() or dest.resolve().is_relative_to(task):
        raise ValueError("Choose a new output directory outside the source task")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        copy = Path(temp) / "task"
        shutil.copytree(task, copy)
        mode = SERVICES[suite]
        for folder, target, hook in (("environment", "/opt/tools/rcb_client.py", "client"),
                                      ("environment/sim", "/srv/" + ("rcb_service.py" if mode == "service" else "server.py"), mode)):
            root = copy / folder
            docker = root / "Dockerfile"
            original = docker.read_text()
            if "arc_capture.py" in original:
                raise ValueError("This task already contains the intent hook")
            for name in ("arc_capture.py", "instrument.py"):
                shutil.copyfile(HERE / name, root / name)
            module_dir = target.rsplit("/", 1)[0]
            with docker.open("a") as f:
                f.write(f"\nCOPY arc_capture.py {module_dir}/arc_capture.py\nCOPY instrument.py /tmp/arc_instrument.py\n"
                        f"RUN python /tmp/arc_instrument.py {hook} {target}\n")
        config = copy / "task.toml"
        text = config.read_text()
        if text.count("artifacts = [") != 1:
            raise ValueError("Unsupported Harbor artifact declaration")
        config.write_text(text.replace("artifacts = [", 'artifacts = [{ source = "/data/arc/", service = "sim" }, '))
        instructions = copy / "instruction.md"
        def add_note(match):
            args = match.group(1)
            return match.group(0) if "note=" in args else f"r.submit({args}, note=local_intent)"
        examples = re.sub(r"r\.submit\(([^()\n]+)\)", add_note, instructions.read_text())
        instructions.write_text(ACTOR.strip() + "\n\n" + examples)
        copy.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    prepare(a.task, a.destination)
