# ROS 2 side

Four packages that run the scan, the solve and the 3D view as ROS 2 Jazzy nodes. The
vision code in `../vision/` stays plain Python with no ROS imports. The nodes call into it.

| Package | What it does |
|---|---|
| `cube_interfaces` | `ScanResult` and `ExecutedMove` messages, `Execute` action |
| `cube_scanner` | the guided webcam scan from `cube_solver.scan` in `vision/`; publishes every scan that solves |
| `cube_executor` | stand-in for the robot: runs an `Execute` goal one move at a time |
| `cube_viewer` | the 3D cube; SPACE sends the solution, then it follows each executed move |
| `cube_bringup` | `cube.launch.py` and `config/cube.yaml`, which start all three nodes |

```
scanner --scan_result (ScanResult, transient local)--> viewer
viewer  --execute (Execute action)--> executor
executor --executed_move (ExecutedMove)--> viewer
```

The robot node will later replace `cube_executor`. It takes the same `Execute` goal and
publishes the same `executed_move` topic, so the scanner and the viewer do not change.

## Setup (once)

The repo root is the colcon workspace. The nodes run on the system Python (the one ROS
Jazzy is built for) inside a venv that also holds the vision code's packages.

```bash
cd cube_solver
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv pip install --python .venv/bin/python -e vision
```

## Build

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python -m colcon build --symlink-install --base-paths ros
source install/setup.bash
```

colcon is run with the venv's Python so that the node scripts it installs start on that
Python, and see the vision packages.

## Run

```bash
ros2 launch cube_bringup cube.launch.py
```

Scan the six faces in the scan window as usual. The viewer draws the cube; press SPACE in
the viewer to solve. Parameters are in `cube_bringup/config/cube.yaml`.

## Test

```bash
.venv/bin/python -m colcon test --base-paths ros
colcon test-result --verbose
```
