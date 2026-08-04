from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_robot_doctor_reports_local_safety_and_permission_boundary(tmp_path: Path):
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["A3_LOCAL_DEPLOYMENT_ROOT"] = str(tmp_path / "missing-deployment")
    environment["A3_LOCAL_CONTROL_SOCKET"] = str(tmp_path / "missing.sock")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "a3_outcome_stack.ops",
            "robot",
            "doctor",
            "--root",
            str(root),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["hardware_available"] is False
    assert report["hardware_tests_executed"] is False
    assert report["motor_enable_executed"] is False
    assert report["real_can_traffic_executed"] is False
    assert report["hardware_verified"] is False
    assert report["operator_session"] == {
        "network_listener_configured": False,
        "unix_socket_active": False,
    }
    assert report["deployment"] == {
        "exists": False,
        "writable_by_current_process": False,
    }
    assert report["role_boundary"]["raw_hardware_access_granted_to_humans"] is False
    assert report["role_boundary"]["operator_reset_allowed"] is False
    assert report["execution_role"] in {
        "administrator",
        "controlled_operator",
        "collaborator",
        "unassigned",
    }
