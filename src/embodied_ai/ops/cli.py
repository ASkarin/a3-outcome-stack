"""Command-line entry point for stable experiment operations."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from embodied_ai.robot.cli import add_robot_parser, run_robot_command

from .canonical import load_json
from .checkpoints import verify_checkpoint
from .doctor import doctor_project
from .errors import OpsError, ValidationError
from .experiments import rebuild_registry, register_experiment
from .freeze import create_freeze, verify_freeze_file
from .manifests import (
    build_asset_manifest,
    build_dataset_manifest,
    verify_asset_manifest,
    verify_dataset_manifest,
    write_manifest_immutable,
)
from .results import finalize_result, rebuild_results_index, verify_results_index


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodied-ai")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--root", default=".")

    experiment = commands.add_parser("experiment")
    experiment_commands = experiment.add_subparsers(dest="action", required=True)
    register = experiment_commands.add_parser("register")
    register.add_argument("--spec", required=True)
    register.add_argument("--specs-dir", default="experiments/specs")
    register.add_argument("--registry", default="experiments/registry.csv")
    register.add_argument("--summaries-dir", default="results/summaries")

    dataset = commands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="action", required=True)
    dataset_manifest = dataset_commands.add_parser("manifest")
    dataset_manifest.add_argument("--root", required=True)
    dataset_manifest.add_argument("--version", required=True)
    dataset_manifest.add_argument("--episodes", required=True)
    dataset_manifest.add_argument("--preregistration-id", default="PR-20260722-01")
    dataset_manifest.add_argument(
        "--preregistration-sha256",
        default="sha256:fa5f70343c520681e10ceb8801869deeaddc4664c7383891dc31eea2d5f9e883",
    )
    dataset_manifest.add_argument("--output", required=True)
    dataset_verify = dataset_commands.add_parser("verify")
    dataset_verify.add_argument("--root", required=True)
    dataset_verify.add_argument("--manifest", required=True)

    asset = commands.add_parser("asset")
    asset_commands = asset.add_subparsers(dest="action", required=True)
    asset_manifest = asset_commands.add_parser("manifest")
    asset_manifest.add_argument("--root", required=True)
    asset_manifest.add_argument("--asset-id", required=True)
    asset_manifest.add_argument("--kind", required=True)
    asset_manifest.add_argument("--logical-uri")
    asset_manifest.add_argument("--output", required=True)
    asset_verify = asset_commands.add_parser("verify")
    asset_verify.add_argument("--root", required=True)
    asset_verify.add_argument("--manifest", required=True)

    checkpoint = commands.add_parser("checkpoint")
    checkpoint_commands = checkpoint.add_subparsers(dest="action", required=True)
    checkpoint_verify = checkpoint_commands.add_parser("verify")
    checkpoint_verify.add_argument("--checkpoint", required=True)
    checkpoint_verify.add_argument("--metadata")

    result = commands.add_parser("result")
    result_commands = result.add_subparsers(dest="action", required=True)
    result_finalize = result_commands.add_parser("finalize")
    result_finalize.add_argument("--input", required=True)
    result_finalize.add_argument("--summaries-dir", default="results/summaries")
    result_reindex = result_commands.add_parser("reindex")
    result_reindex.add_argument("--summaries-dir", default="results/summaries")
    result_reindex.add_argument("--index", default="results/index.jsonl")
    result_reindex.add_argument("--specs-dir", default="experiments/specs")
    result_reindex.add_argument("--registry", default="experiments/registry.csv")
    result_verify = result_commands.add_parser("verify")
    result_verify.add_argument("--summaries-dir", default="results/summaries")
    result_verify.add_argument("--index", default="results/index.jsonl")

    freeze = commands.add_parser("freeze")
    freeze_commands = freeze.add_subparsers(dest="action", required=True)
    freeze_create = freeze_commands.add_parser("create")
    freeze_create.add_argument("--input", required=True)
    freeze_create.add_argument("--output", required=True)
    freeze_verify = freeze_commands.add_parser("verify")
    freeze_verify.add_argument("--manifest", required=True)
    add_robot_parser(commands)
    return parser


def _run(args: argparse.Namespace) -> Any:
    if args.command == "robot":
        return run_robot_command(args)
    if args.command == "doctor":
        return doctor_project(args.root)
    if args.command == "experiment" and args.action == "register":
        return register_experiment(
            load_json(args.spec),
            specs_dir=args.specs_dir,
            registry_path=args.registry,
            summaries_dir=args.summaries_dir,
        )
    if args.command == "dataset" and args.action == "manifest":
        episodes = load_json(args.episodes)
        if not isinstance(episodes, list):
            raise ValidationError("episodes JSON must be a list")
        manifest = build_dataset_manifest(
            args.root,
            data_version=args.version,
            episodes=episodes,
            preregistration_id=args.preregistration_id,
            preregistration_sha256=args.preregistration_sha256,
        )
        write_manifest_immutable(manifest, args.output, kind="dataset")
        return manifest
    if args.command == "dataset" and args.action == "verify":
        manifest = load_json(args.manifest)
        verify_dataset_manifest(args.root, manifest)
        return {
            "status": "ok",
            "manifest": args.manifest,
            "manifest_sha256": manifest["manifest_sha256"],
        }
    if args.command == "asset" and args.action == "manifest":
        manifest = build_asset_manifest(
            args.root,
            asset_id=args.asset_id,
            kind=args.kind,
            logical_uri=args.logical_uri,
        )
        write_manifest_immutable(manifest, args.output, kind="asset")
        return manifest
    if args.command == "asset" and args.action == "verify":
        manifest = load_json(args.manifest)
        verify_asset_manifest(args.root, manifest)
        return {
            "status": "ok",
            "manifest": args.manifest,
            "manifest_sha256": manifest["manifest_sha256"],
        }
    if args.command == "checkpoint" and args.action == "verify":
        return verify_checkpoint(args.checkpoint, args.metadata)
    if args.command == "result" and args.action == "finalize":
        destination = finalize_result(load_json(args.input), args.summaries_dir)
        return {"status": "ok", "summary": str(destination)}
    if args.command == "result" and args.action == "reindex":
        summaries = rebuild_results_index(args.summaries_dir, args.index)
        rebuild_registry(args.specs_dir, args.registry, args.summaries_dir)
        return {"status": "ok", "records": len(summaries), "index": args.index}
    if args.command == "result" and args.action == "verify":
        verify_results_index(args.summaries_dir, args.index)
        return {"status": "ok", "index": args.index}
    if args.command == "freeze" and args.action == "create":
        return create_freeze(load_json(args.input), args.output)
    if args.command == "freeze" and args.action == "verify":
        manifest = verify_freeze_file(args.manifest)
        return {
            "status": "ok",
            "manifest": args.manifest,
            "freeze_sha256": manifest["freeze_sha256"],
        }
    raise ValidationError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _emit(_run(args))
        return 0
    except OpsError as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "message": str(exc)}, sort_keys=True
            ),
            file=sys.stderr,
        )
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
