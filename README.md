# RoboMaster Parking Controller (Group C)

This repository contains the ROS 2 packages, CoppeliaSim scenes, and vision-based control software used for the RoboMaster EP automated parking task.

## Overview

The project includes:

- ROS 2 interfaces and launch files for simulation and control.
- Vision-based parking point detection.
- A state-machine parking controller for autonomous maneuvering.

## Installation and Setup

This section explains how to install CoppeliaSim and build the ROS 2 workspace with Pixi.

### 1. Install CoppeliaSim

#### macOS

1. Download CoppeliaSim for [Apple Silicon](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_macOS15_arm64.zip) or [Intel](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_macOS13_x86_64.zip).
2. Unzip the archive and move the app to `/Applications/coppeliaSim.app`.
3. Open it once manually: right-click -> Open -> Open.
4. If you get permission errors, allow CoppeliaSim in System Settings -> Privacy and Security.

#### Ubuntu

1. Download CoppeliaSim for [Ubuntu 22.04](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04.tar.xz) or [Ubuntu 24.04](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04.tar.xz).
2. Extract it in a directory of your choice:

```bash
cd <COURSE_FOLDER>
tar xvf CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu<UBUNTU_VERSION>.tar.xz
```

### 2. Install the RoboMaster ROS 2 Environment

1. Clone the repository with submodules:

```bash
git clone git@github.com:idsia-robotics/robotics-lab-usi-robomaster.git --recursive
```

2. Ubuntu only: set `COPPELIASIM_ROOT_DIR` in `pixi.toml` to your local CoppeliaSim path, for example:

```toml
[activation.env]
COPPELIASIM_ROOT_DIR = "<PATH_TO_COPPELIA>/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu<UBUNTU_VERSION>"
```

3. Build and install dependencies:

```bash
cd robotics-lab-usi-robomaster
pixi install
pixi shell
colcon build --symlink-install
```

Note: the build may take a few minutes.

### 3. Verify the Installation (Optional)

Launch CoppeliaSim through Pixi:

```bash
source install/setup.zsh
pixi run coppelia
```

Inside CoppeliaSim, add a robot from:
Model browser -> robots -> mobile -> RoboMasterEP

In another terminal, verify discovery from Python:

```bash
cd src/robomaster_sim/examples
pixi shell
python discover.py
```

macOS 15 note: if you see `scan_robot_ip: exception timed out`, enable local network permission for your terminal app.

## Running the Parking Mission

Use 2 terminals.

### Terminal 1: Run CoppeliaSim

```bash
cd robotics-lab-usi-robomaster
pixi run coppelia
```

In CoppeliaSim:

1. Open scene `battle2`.
2. Enable real-time mode (clock icon).
3. Press Play.

This scene already includes the simulation clock helper and the `robomaster_ep_tof_v2.ttm` robot model.

### Terminal 2: Launch Mission

```bash
cd robotics-lab-usi-robomaster
pixi shell
source install/setup.zsh
ros2 launch robomaster_example mission.launch
```

`mission.launch` already includes `ep_tof.launch` and starts both controller nodes.
Only launch `ep_tof.launch` separately for debugging.
If needed, run the parking controller manually in a third terminal:

```bash
ros2 run robomaster_example controller_park4
```

## Algorithm Logic and Assumptions

### Assumptions

Before starting the parking sequence, the following assumptions must hold:

- Four parking points are provided dynamically by the vision node and represent an approximately rectangular parking spot.
- No moving obstacles interfere with the robot during the approach.
- At least 0.5 m of free space is available in front of the parking entrance for alignment.

## State Machine Logic

The parking controller (`controller_park4.py`) is implemented as a sequential state machine:

### 0. STATE = START

- Check whether there is enough 2D space to park using the dimensions from vision.
- Compute geometric vectors `c` (center), `e`, `n`, `m`, and `p` (optimal approach point).

### 1. STATE = MOVE_TO_P

- Rotate toward point `p`, located 0.5 m in front of the parking entrance.
- Move forward until point `p` is reached.

### 2. STATE = ROTATE_TO_FACE_PARKING

- Rotate in place to face point `c` (the parking center).

### 3. STATE = FORWARD

- Move forward toward `c` to enter the parking spot.
- Disable angle correction during the last 25 cm to avoid oscillations near the goal.

### 4. STATE = PARKED

- Stop all motors.
- Log success when the robot is within 5 cm of the parking center.

