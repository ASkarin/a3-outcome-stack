# `lerobot_robot_a3`

This is the official-style, auto-discoverable LeRobot hardware plugin for EduLite A3.
The Python distribution and import package both start with `lerobot_robot_`, so
LeRobot discovers `A3RobotConfig` without a project-side registry.

The default mode is `read_only`. `connect()` never runs interactive calibration and
rejects missing or unfrozen calibration. Motion additionally requires the frozen
safety document, verified physical emergency stop, and all five hardware-gate flags.
The motion gate also binds the exact frozen calibration and safety files by SHA-256.
The unique administrator runs the plugin directly from an immutable release; there
is no socket service, permit, or dedicated runtime account.
