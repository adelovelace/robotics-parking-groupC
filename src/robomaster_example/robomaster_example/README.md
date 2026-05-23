# ASSUMPTIONS:
- The 4 points are given (hardcoded now)
- There are no objects which it can collide with
- The 4 points form a rectangular-ish shape
- There is at least 0.5m space in front of the parking spot

# LOGIC:
0.  STATE = START: 
	- Evaluate if there is enough 2d space
	- Compute c, e, n, m, p
1. STATE = MOVE_TO_P
	- Rotate towards p (0.5 m in front of parking spot)
	- and go to p
2. STATE = ROTATE_TO_FACE_PARKING
	- Rotate to face c (center of parking spot)
3. STATE = PARKED


# SETUP
## 1. TERMINAL 1: OPEN COPELIA
In robotics-lab-usi-robomaster
pixi run coppelia

IN COPPELIA:
ADD SCENE robomasterv2-clock.ttt
	ALREADY CONTAINS:
		- MODEL OF THE CLOCK: ros2Interface helper tool + clock.ttm
		- NEW ROBOT: robomaster_ep_tof_v2.ttm
ACTIVATE REAL TIME MODE
PLAY: START SIMULATION

## 2. TERMINAL 2: CONNECT TO ROBOMASTER
In robotics-lab-usi-robomaster
pixi shell
source install/setup.zsh
ros2 launch robomaster_example ep_tof.launch name:=/rm0

## 3. TERMINAL 3: RUN CONTROLLER
pixi shell
colcon build --symlink-install
source install/setup.zsh
ros2 launch robomaster_example park_4points.launch name:=/rm0