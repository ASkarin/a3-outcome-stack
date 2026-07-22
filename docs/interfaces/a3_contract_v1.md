# EduLite A3 interface contract v1

This document describes the stage-1A, hardware-unverified contract. The machine-readable source is `configs/robot/a3_contract_v1.json`.

## Upstream audit

- Official repository: `https://github.com/RobStride/EDULITE_A3`
- Pinned commit: `ea7231f784ebb37e4c4120f7be8e3670514dc9ee`
- Lock: `configs/upstream/edulite_a3.lock.json`
- Audited SDK calls: `ConnectPort`, `DisconnectPort`, `EnableArm`, `DisableArm`, `EmergencyStop`, `JointCtrlList`, `GetArmJointMsgs`, `GetArmJointVelocities`, `GetArmJointEfforts`, `GetArmEndPoseMsgs`, and `GetArmStatus`.
- Audited vendor states: `DISCONNECTED`, `IDLE`, `ENABLED`, `RUNNING`, `ZERO_TORQUE`, and `ERROR`.

The SDK source uses SI units and returns seven values for L1–L6 plus the L7 gripper. The project wraps those calls; project code must not call the vendor object directly.

## Observation

| Field | Shape | Unit | Meaning |
|---|---:|---|---|
| `joint_position` | 7 | rad | L1–L7 absolute joint positions |
| `joint_velocity` | 7 | rad/s | L1–L7 joint velocities |
| `joint_effort` | 7 | N·m | L1–L7 reported joint effort |
| `tcp_position` | 3 | m | TCP position in `base` |
| `tcp_orientation_xyzw` | 4 | unit quaternion | TCP orientation in `base` |

Status and timestamps are audit/control metadata, not default policy inputs. The official TCP method depends on Pinocchio; until that path is installed and validated, `tcp_pose_valid` remains false even though the schema stays stable. Camera feature names and shapes must be configured before connection.

## Action

The only core action is `joint_position_abs`, a seven-element L1–L7 vector in radians. Sequence ID, creation time, deadline, and clock domain are mandatory. Cartesian, delta, and teleoperation commands must be converted above `SafeRobot` and cannot bypass validation.

## Compatibility boundary

Feature dictionaries are available before connection and intentionally match the adapter surface needed by LeRobot. Stage 1A does not claim that LeRobot is installed or that a LeRobot subclass has been validated. That integration remains a stage-1B task.

## Evidence boundary

`A3SdkBackend.hardware_verified` is false in stage 1A. Passing mock, replay, or fake-vendor tests is code-readiness evidence only and is not real-robot evidence.

The deterministic mock uses no random number generator. Exact replay therefore compares every action, observation, transition, receipt, and next virtual-clock value; a future stochastic mock must add and seal its RNG state before it can claim exact replay.
