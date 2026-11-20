# Intent-grounded collection with Embodied First Exam

One example is one executed `submit` call. Before it moves the robot, the actor writes a short note stating the intended motion, target and observable result of that chunk. For instance, “Lower the open gripper until the mug handle lies between the fingers” describes an approach that can succeed before a later grasp. EFE’s original task, simulator and episode grader supply the rollout.

## Prepare and run a task

The overlay targets the following [pinned EFE release](https://github.com/mll-lab-nu/embodied-first-exam/tree/487ba34ef5586a68762aeeae6142da0a67268e5e). Use its Docker/Harbor setup and the hardware required by the selected task. Run these commands from the AgentRoboCritic checkout on the collection machine.

```bash
python -m pip install -e '.[efe]'
git clone https://github.com/mll-lab-nu/embodied-first-exam.git ../embodied-first-exam
EFE_ROOT=$(cd ../embodied-first-exam && pwd)
git -C "$EFE_ROOT" checkout 487ba34ef5586a68762aeeae6142da0a67268e5e
git -C "$EFE_ROOT" submodule update --init --recursive
uv tool install --force -e "$EFE_ROOT" --with-executables-from harbor

TASK=robotwin_2_0_adjust_bottle-standard
WORKDIR=$(mktemp -d)
# Create a task copy that requires the actor to state its intent in note= before each action chunk runs.
python efe/prepare_task.py "$EFE_ROOT/tasks/robotwin-2/$TASK" "$WORKDIR/intent/$TASK"
STAGED=$(python "$EFE_ROOT/scripts/stamp_mtimes.py" stage "$WORKDIR/intent/$TASK" "$WORKDIR/staged")
```

`prepare_task.py` adds the following text to the top of the copied task’s `instruction.md`. The actor sees it before choosing any actions.

```markdown
## Local intent for every action

Before submitting each action chunk, pass a nonblank `note` (at most 500 characters).
State the intended action, its target, and the immediate observable objective of
this chunk. An intermediate approach, grasp, or lift can have its own objective.
Write the note prospectively. A submit without a note is refused before motion.
Use the simulator's original action types, dimensions and units.
In the examples, set `local_intent` in your own words for each new chunk.
Request `video="frames"` when you want intermediate camera observations.
```

Set up the actor’s credentials as described in EFE. For example, run a single Codex trial under the original Standard constraints. Use a dedicated job directory and a fresh episode ID for each rollout.

```bash
export ACTOR_MODEL=openai/gpt-6-astra
harbor run -p "$STAGED" --force-build -a codex -m "$ACTOR_MODEL" \
    --ak "config=$(python "$EFE_ROOT/scripts/codex_tools_off.py")" \
    -n 1 -o "$PWD/runs/efe" --job-name bottle-intent-01
```

The preparation step edits a copy of the task. It adds the actor instruction, patches the client and simulator modules used in the built images, and adds `/data/arc/` to the collected trial artifacts. Both the client and the service validate the note. Direct service rejections are recorded separately. The original numeric action and the simulator-reported execution count share the returned native chunk ID. This preserves early termination when only part of a chunk ran.

Supported Standard task layouts at this revision are `behavior-1k`, `bigym2`, `kinder`, `libero-max`, `maniguard`, `pandadyn`, `robodojo`, `robolab`, `robopaint`, `robotwin-2`, `vlabench` and `yamlab`. The instrumentation checks the native method signatures at build time. For image prerequisites and task-specific GPU requirements, use the [EFE running guide](https://github.com/mll-lab-nu/embodied-first-exam/blob/487ba34ef5586a68762aeeae6142da0a67268e5e/docs/protocol/running.md).

## Convert a completed trial

Pass the Harbor trial directory or its collected `arc` directory. Use the original task text and a stable, unique episode ID. For the one-trial job above, the converter can locate the single capture in the job directory.

```bash
arc convert-efe \
    --trial runs/efe/bottle-intent-01 --episode-id roboTwin-adjust-bottle-01 \
    --task "Pick up the bottle on the table headup with the correct arm." \
    --out runs/bottle/data/test-00000.parquet

arc evaluate --data runs/bottle --profile muse --out runs/bottle-predictions
```

The converter writes one row per submit with a positive executed-row count. It selects the preceding call’s final views, the first, middle and last images available from the current call, and the final views of an immediately following observe call when present. It uses native call order and camera metadata; follow-up evidence ends before a new action. Frame arrays and native MP4 clips are supported. When the service returns only a final view, that is the available evidence. Each converted action includes the temporal role of every image and native timing where available. The human-annotated benchmark is already in this format, so conversion is only needed for newly collected trials.

The output has the benchmark’s exact Hugging Face features: `id`, `messages`, `images`, and `human_label`. Fresh rollouts have a null human label and can be evaluated immediately. To score a reviewed rollout, prepare one JSON line per executed action, using the IDs from the Parquet file. For example, if the executed call’s native ID is 2:

```json
{"id":"efe/roboTwin-adjust-bottle-01/2","human_label":{"intent_success":"achieved","failure_mechanism":null,"m1_subcategory":null}}
```

Repeat `arc convert-efe` with the same episode ID and task, add `--labels reviewed.jsonl`, and use a new output filename. Then score the existing predictions against the labeled file:

```bash
arc score --data runs/reviewed/test-00000.parquet \
    --predictions runs/bottle-predictions/predictions.jsonl
```

For a multi-episode dataset, write one `test-*.parquet` shard per episode into the same data directory. Conversion refuses unfinished captures and checks exact coverage when you supply human labels. The actor’s prospective note, the human action judgment and EFE’s official episode result serve different purposes.
