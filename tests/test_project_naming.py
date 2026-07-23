from __future__ import annotations

import subprocess
import sys

import a3_outcome_stack
import embodied_ai


def test_canonical_and_legacy_namespaces_share_version() -> None:
    assert a3_outcome_stack.__version__ == "0.2.0"
    assert embodied_ai.__version__ == a3_outcome_stack.__version__


def test_legacy_module_entrypoint_resolves_canonical_cli() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "embodied_ai.ops", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "a3-outcome-stack" in completed.stdout
