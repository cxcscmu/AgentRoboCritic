import argparse
import json
from dataclasses import replace
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

from .data import DATASET, REVISION, rows
from .providers import PROFILES
from .runner import records, run
from .scoring import score


def main():
    parser = argparse.ArgumentParser(description="AgentRoboCritic evaluation")
    subs = parser.add_subparsers(dest="command", required=True)
    evaluate = subs.add_parser("evaluate", help="Run a commercial API or GPU-served critic")
    scoring = subs.add_parser("score", help="Score predictions against human labels by action ID")
    verification = subs.add_parser("verify", help="Verify the paper benchmark inputs without inference")
    for sub in (evaluate, scoring, verification):
        sub.add_argument("--data", default=DATASET, help="Hub dataset, local shard folder or Parquet file")
        sub.add_argument("--revision", help="Hub revision (the paper benchmark is pinned by default)")
    for sub in (evaluate, scoring):
        sub.add_argument("--limit", type=int, help="Use the first N actions for a smoke run")
    evaluate.add_argument("--profile", choices=PROFILES, required=True)
    evaluate.add_argument("--model", help="Provider model name or deployment ID")
    evaluate.add_argument("--base-url", help="API base URL, including the API version prefix")
    evaluate.add_argument("--key-env", help="Name of the API key environment variable")
    evaluate.add_argument("--max-tokens", type=int, help="Override the profile’s output token limit")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--out", type=Path)
    convert = subs.add_parser("convert-efe", help="Convert intent-grounded EFE calls to benchmark Parquet")
    convert.add_argument("--trial", type=Path, required=True, help="Collected arc directory or a trial containing one")
    convert.add_argument("--episode-id", required=True, help="Unique episode identifier across all runs")
    convert.add_argument("--task", required=True, help="Original task text shown to the actor")
    convert.add_argument("--labels", type=Path, help="Optional human-reviewed JSONL keyed by id")
    convert.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "convert-efe":
        from .efe import convert_trial
        print(f"Wrote {convert_trial(args.trial, args.episode_id, args.task, args.labels, args.out)} actions")
        return
    revision = args.revision or (REVISION if args.data == DATASET else None)
    if args.command == "verify":
        from .verify import verify
        print(json.dumps(verify(args.data, revision), indent=2))
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "evaluate":
        profile = PROFILES[args.profile]
        base = args.base_url or profile.base_url
        url = urlsplit(base)
        if (not url.hostname or url.username or url.password or url.query or url.fragment or
            (url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost","127.0.0.1","::1"}))):
            parser.error("Use an HTTPS endpoint or local HTTP server")
        if url.hostname != urlsplit(profile.base_url).hostname and profile.key_env and not args.key_env:
            parser.error("Specify --key-env for a different provider host")
        profile = replace(profile, model=args.model or profile.model, base_url=base,
                          key_env=args.key_env or profile.key_env,
                          tokens=args.max_tokens if args.max_tokens is not None else profile.tokens)
        if not profile.model or profile.tokens <= 0:
            parser.error("Provide a model and positive output token budget")
        count = run(profile, args.data, revision, args.out, args.workers, args.limit)
        print(f"Saved {count} predictions to {args.out / 'predictions.jsonl'}")
    else:
        report = score(islice(rows(args.data, revision, labels_only=True), args.limit), records(args.predictions))
        result = json.dumps(report, indent=2)
        if args.out:
            args.out.write_text(result + "\n")
        print(result)
