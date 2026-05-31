# RoboMaster Parking Controller (Group C)

This repository contains the ROS 2 packages, CoppeliaSim scenes, and vision-based control algorithms for the RoboMaster EP automated parking task.

---

## 🛠️ INSTALLATION & SETUP

This guide provides step-by-step instructions for installing the RoboMaster Simulation running in CoppeliaSim using Pixi.

### 1. Install CoppeliaSim

**For macOS:**
1. Download CoppeliaSim for [Apple Silicon](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_macOS15_arm64.zip) or [Intel](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_macOS13_x86_64.zip).
2. Unzip and move to `/Applications/coppeliaSim.app`.
3. Right-click on `coppeliaSim.app` -> Open -> Open.
4. *Troubleshooting:* If you encounter any permission errors, please authorize coppeliaSim in your Mac's System Settings under Security & Privacy.

**For Ubuntu:**
1. Download CoppeliaSim for [Ubuntu 22.04](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04.tar.xz) or [Ubuntu 24.04](https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04.tar.xz).
2. Extract CoppeliaSim in a directory of your choice (e.g., your `COURSE_FOLDER`):
   ```bash
   cd <COURSE_FOLDER>
   tar xvf CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu<UBUNTU_VERSION>.tar.xz

```

### 2. Install the RoboMaster ROS 2 Environment

We use a Pixi project with all required dependencies for using the RoboMaster in CoppeliaSim.

1. Clone this repository (ensure you use the `--recursive` flag):
```bash
git clone git@github.com:idsia-robotics/robotics-lab-usi-robomaster.git --recursive

```


2. **Ubuntu Only:** Customize the `COPPELIASIM_ROOT_DIR` in the `pixi.toml` of this repo to point to your CoppeliaSim installation. Change it to something like:
```toml
[activation.env]
COPPELIASIM_ROOT_DIR = "<PATH_TO_COPPELIA>/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu<UBUNTU_VERSION>"

```


3. Enter the repository, compile, and install the packages:
```bash
cd robotics-lab-usi-robomaster
pixi install
pixi shell
colcon build --symlink-install

```


*(Note: The build process can take a couple of minutes).*

### 3. Verify the Installation (Optional but Recommended)

Open a terminal and launch CoppeliaSim through pixi to ensure the ROS packages are injected correctly:

```bash
source install/setup.zsh
pixi run coppelia

```

Inside CoppeliaSim, add a RoboMaster: `Model browser -> robots -> mobile -> RoboMasterEP` and press Play.

In a new terminal, check if the Python scripts can find the robot:

```bash
cd src/robomaster_sim/examples
pixi shell
python discover.py

```

*macOS 15 Troubleshooting:* If communication fails with `scan_robot_ip: exception timed out`, go to System Settings -> Privacy & Security -> Local Network and ensure your terminal app is authorized.

---

## 🚀 RUNNING THE PARKING MISSION

Once the setup is complete, you can run the automated parking sequence. You will need 3 separate terminal windows.

### TERMINAL 1: Open CoppeliaSim

Navigate to your workspace and launch the simulator:

```bash
cd robotics-lab-usi-robomaster
pixi run coppelia

```

**Inside CoppeliaSim:**

1. Add / Open the scene: `robomasterv2-clock.ttt`
> *Note: This scene already contains the clock model (`ros2Interface helper tool + clock.ttm`) and the new robot model (`robomaster_ep_tof_v2.ttm`).*


2. **Activate Real-Time Mode** (the clock icon).
3. Press **PLAY** to start the simulation.

### TERMINAL 2: Connect to the RoboMaster

Open a new terminal, enter the pixi shell, and launch the robot's base drivers:

```bash
cd robotics-lab-usi-robomaster
pixi shell
source install/setup.zsh
ros2 launch robomaster_example ep_tof.launch name:=/rm0

```

### TERMINAL 3: Run the Parking Controllers

Open a third terminal, source the workspace, and launch the mission.

```bash
cd robotics-lab-usi-robomaster
pixi shell
source install/setup.zsh
ros2 launch robomaster_example mission.launch

```

*(Make sure that your `mission.launch` file is updated to execute BOTH the vision node and the `controller_park4` node. Alternatively, you can run the parking node manually in a 4th terminal using `ros2 run robomaster_example controller_park4`).*

---

## 🧠 ALGORITHM LOGIC & ASSUMPTIONS

### Assumptions

Before running the parking sequence, please consider the following environmental assumptions:

* The 4 parking points are dynamically given by the vision node, but they assume a rectangular-ish spot.
* There are no moving objects which the robot can collide with during the approach.
* There is at least 0.5m of free space in front of the parking spot to allow the robot to align itself perfectly.

### State Machine Logic

The parking controller (`controller_park4.py`) operates using a sequential state machine:

* **0. STATE = START**
* Evaluate if there is enough 2D space to park based on the dimensions provided by the camera.
* Compute the necessary geometric vectors: `c` (center), `e`, `n`, `m`, and `p` (the optimal approach point).


* **1. STATE = MOVE_TO_P**
* Rotate towards point `p` (located exactly 0.5 m in front of the parking spot entrance).
* Drive forward until point `p` is reached.


* **2. STATE = ROTATE_TO_FACE_PARKING**
* Rotate in place to face `c` (the absolute center of the parking spot).


* **3. STATE = FORWARD**
* Drive straight into the parking spot towards `c`.
* Disable angle correction in the last 25cm to prevent infinite spinning loops due to proximity math.


* **4. STATE = PARKED**
* Stop the motors completely and log a success message once the robot is within 5cm of the center.

