"""Small project CLI; LeRobot owns data, training, and checkpoint lifecycles."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from a3_outcome_stack.robot.cli import add_robot_parser, run_robot_command

from .doctor import doctor_project
from .errors import OpsError, ValidationError


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a3-outcome-stack")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--root", default=".")
    add_robot_parser(commands)
    return parser


def _run(args: argparse.Namespace) -> Any:
    if args.command == "doctor":
        return doctor_project(args.root)
    if args.command == "robot":
        return run_robot_command(args)
    raise ValidationError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _emit(_run(args))
        return 0
    except OpsError as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
