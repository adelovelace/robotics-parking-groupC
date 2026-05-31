# RoboMaster Object-Boundary Exploration and Parking

This project implements a ROS 2 pipeline for a RoboMaster robot that explores the boundary of a red object, builds a 2D world-frame map of object boundaries and visible empty space, detects a feasible parking slot, validates the slot from multiple viewpoints, and finally parks into the selected slot.

The system was refactored from a monolithic controller into a small set of ROS nodes. Each node owns one clear responsibility and communicates through topics. This keeps the code easier to debug and avoids mixing camera processing, map fusion, parking geometry, mission control, visualization, and final parking control in one file.

## Quick start

Build the package in your ROS 2 workspace:

```bash
colcon build --packages-select robomaster_example
source install/setup.bash
```

Run the complete refactored mission for robot namespace `rm0`:

```bash
ros2 launch robomaster_example refactored_mission.launch name:=rm0
```

Run without OpenCV visualization windows:

```bash
ros2 launch robomaster_example refactored_mission.launch name:=rm0 use_visualization:=false
```

Because the launch file starts all nodes under the namespace given by `name`, topics will appear under `/rm0/...` when `name:=rm0` is used.

---

## High-level objective

The robot must solve two related tasks:

1. **Explore the object boundary.**
   The robot repeatedly stops, observes, updates a global map, and moves around the object while keeping a safe clearance from boundary points.

2. **Find and enter a parking slot.**
   A parking slot is not accepted just because a rectangle can be fitted. The system first checks whether at least one rectangle side is a safe and reachable entrance. It then validates the slot from three nearby viewpoints before handing control to the final parking node.

The current mission follows this behavioral pattern:

```text
SEE -> ACT_ARC -> SEE -> ...

When an actionable parking slot appears:

SEE -> VALIDATE_SLOT -> RETURN_TO_P -> DELEGATE -> PARK4 -> PARKED
```

A key design decision is that the global map is only updated from stable observations. The robot does not fuse moving-camera frames while driving around the object.

---

## Package structure

```text
robomaster_example/
  logic/
    grid_map.py
    parking_estimator.py
    slot_geometry.py
    topic_codec.py

  nodes/
    vision_observer_node.py
    map_node.py
    slot_detector_node.py
    mission_controller_node.py
    controller_park4.py
    visualization_node.py

  vision/
    estimate.py
    __init__.py

  launch/
    refactored_mission.launch
```

### `vision/estimate.py`

Low-level vision and projection code.

It handles:

- red-object segmentation;
- floor/support mask estimation;
- object-floor contact boundary extraction;
- empty-space sampling below the detected boundary;
- projection from image pixels to floor coordinates;
- transformation from robot-frame floor coordinates to world-frame coordinates.

The main class is:

```python
FloorProjectiveTransform
```

The most important method for the runtime system is:

```python
estimate_boundaries_world(image, robot_pose, **kwargs)
```

It returns:

```text
world boundary points
world empty-space points
BoundaryResult with debug masks and image pixels
```

---

## Runtime nodes

### 1. `vision_observer_node`

**Responsibility:** produce one static visual observation on request.

This node subscribes to the camera and odometry, but it does not continuously publish map points. It waits until the mission controller requests an observation. Then it waits for the robot to be stable, waits an additional settling delay, processes one synchronized image/odom pair, and publishes the resulting boundary and empty-space point clouds.

This prevents map artifacts caused by odometry/image mis-synchronization while the robot is moving.

#### Subscribes

```text
camera/image_color
odom
mission/observation_request
camera/camera_info
```

#### Publishes

```text
vision/boundary_observation
vision/observation_done
```

#### Observation logic

```text
observation request received
    wait until odometry velocity is small
    wait CAPTURE_SETTLE_S seconds
    capture synchronized image + odom
    estimate boundary and empty-space points in world frame
    publish vision/boundary_observation
    publish vision/observation_done
```

The observation request is a `std_msgs/String`. The request name is used for state-machine coordination and logging. Examples:

```text
see_center
see_left
see_right
validation_center
validation_right
validation_left
```

---

### 2. `map_node`

**Responsibility:** own and update the global map.

This node is the only owner of the accumulated boundary and empty-space point clouds. It receives static observations from `vision_observer_node`, snaps points to a fixed grid resolution, deduplicates them, and republishes the global map.

#### Subscribes

```text
vision/boundary_observation
map/reset
```

#### Publishes

```text
map/global_points
map/stats
```

#### Internal logic

The map stores two point clouds:

```text
boundary_points: observed object-floor boundary points
empty_points: observed traversable/visible empty floor points
```

Both are stored in world coordinates.

The map update is intentionally simple:

```text
sanitize finite points
snap to resolution grid
append to accumulated map
remove duplicates
publish global map
```

The default grid resolution is `0.02 m`.

---

### 3. `slot_detector_node`

**Responsibility:** estimate parking candidates and decide whether they are actionable.

This node takes the current global map and attempts to estimate a parking rectangle. It then checks every rectangle side as a possible entrance. The slot becomes actionable only if at least one side is likely open and the robot can safely reach the pre-parking point for that side.

#### Subscribes

```text
map/global_points
odom
```

#### Publishes

```text
parking/candidate
parking/actionable
parking/debug
```

#### Parking candidate estimation

The `ParkingEstimator` works approximately as follows:

```text
boundary points -> object convex hull
empty points inside hull -> possible interior empty space
remove empty points too close to boundary
cluster safe empty points
choose largest cluster
fit minimum rotated rectangle
```

The output is a candidate parking rectangle with four corners.

A candidate rectangle means:

```text
The map geometry contains a plausible parking region.
```

It does not automatically mean:

```text
The robot can safely enter it now.
```

#### Safe entry-side gate

For every rectangle side, the node computes:

```text
wall_support
p_clearance
approach_clearance
entry_clearance
empty_support
robot distance to p
```

Where:

- `wall_support` measures how much object-boundary evidence lies along that side. High support means the side is probably blocked by a wall.
- `p_clearance` is the distance from the pre-parking point `p` to the nearest boundary point.
- `approach_clearance` is the minimum clearance along the segment from the robot to `p`.
- `entry_clearance` is the minimum clearance along the segment from `p` to the slot center.
- `empty_support` is a soft preference indicating whether the area near `p` has observed empty-space points.

A side is rejected if:

```text
wall_support is too high
or p is too close to boundary points
or robot -> p is blocked
or p -> center is blocked
```

If no safe side exists, the rectangle remains only a candidate and the robot continues go-around exploration.

If a safe side exists, the rectangle is reordered as:

```text
[top_l, top_r, bottom_l, bottom_r]
```

where:

```text
bottom_l -> bottom_r
```

is the selected free entry side.

---

### 4. `mission_controller_node`

**Responsibility:** own the high-level mission state machine.

This node decides when to observe, when to move around the object, when to validate a parking slot, and when to delegate the final parking maneuver to `controller_park4`.

It does not process images, estimate parking rectangles, or own the global map.

#### Subscribes

```text
odom
map/global_points
parking/candidate
parking/actionable
vision/observation_done
```

#### Publishes

```text
cmd_vel
mission/observation_request
mission/state
parking_target
```

#### States

```text
SEE
ACT_ARC
VALIDATE_SLOT
DELEGATE
DONE
```

#### `SEE`

The robot stops and requests one or more static observations.

Most SEE cycles use a center-only observation. Periodically, or while the map is sparse, the robot performs a wider observation sequence:

```text
align approximately perpendicular to nearest boundary
capture center
rotate slightly left and capture
rotate slightly right and capture
```

This balances two requirements:

- avoid overly frequent stopping;
- collect enough boundary evidence for reliable slot detection.

If an actionable slot is available after SEE, the mission controller starts validation. Otherwise it starts another local arc motion around the object.

#### `ACT_ARC`

The robot performs a short local boundary-following motion using the frozen global boundary map.

The controller finds the nearest boundary point, computes the radial direction away from it, computes the tangent direction around it, and commands a small arc-like movement while correcting clearance.

The intent is:

```text
move around the object
keep approximately fixed distance from boundary
avoid long straight-line jumps
return to SEE after a short step
```

This replaced the earlier global viewpoint planner, which selected too-distant goals and made the robot move in a straight line toward them.

#### `VALIDATE_SLOT`

When a slot is actionable, the robot validates it from three nearby views:

```text
move to pre-parking point p
rotate to face the slot
capture center validation view
move slightly right and capture
move slightly left and capture
```

Each validation snapshot is added to the global map. After each snapshot, the parking rectangle is re-estimated from the global map, not from the validation-only observations.

Validation passes if at least two out of three global-map rectangle estimates are consistent and the safe-entry gate still passes.

This avoids the earlier failure where a rectangle estimated from only three local validation snapshots could rotate significantly because the local empty-space cluster was view-dependent.

#### `RETURN_TO_P`

After validation passes, the mission controller moves the robot back to the pre-parking point `p` and heading used for the selected entry side.

This is important because the three validation snapshots include lateral right/left offsets. The robot must return to the correct pre-parking pose before delegating to the final parking controller.

#### `DELEGATE`

The mission controller publishes the ordered rectangle on `parking_target` several times for robustness.

After successful delegation, it releases `cmd_vel` ownership to `controller_park4`. This avoids a command conflict where the mission controller could keep publishing zero velocity while `controller_park4` tries to drive into the slot.

#### `DONE`

Mission control is complete. If parking has been delegated, `mission_controller_node` no longer publishes stop commands.

---

### 5. `controller_park4`

**Responsibility:** perform the final parking maneuver after validation has succeeded.

This node is intentionally separate from exploration. It receives an already ordered parking rectangle and drives the robot into the parking center.

#### Subscribes

```text
odom
parking_target
```

#### Publishes

```text
cmd_vel
```

#### Input rectangle contract

`parking_target` is an 8-value `Float32MultiArray`:

```text
[top_l.x, top_l.y,
 top_r.x, top_r.y,
 bottom_l.x, bottom_l.y,
 bottom_r.x, bottom_r.y]
```

The entry side is:

```text
bottom_l -> bottom_r
```

The parking node must not reselect the closest edge. The closest edge may be blocked by the object. The safe entry side has already been selected by `slot_detector_node` and validated by `mission_controller_node`.

#### Current simplified behavior

Because validation already moves the robot close to `p`, `controller_park4` starts directly from the orientation/entry phase instead of restarting the full approach from scratch.

The expected flow is:

```text
receive ordered slot
compute center c, entry midpoint m, outward normal n, pre-parking point p
rotate to face parking center
drive to center using simple proportional body-frame control
stop when centered
```

---

### 6. `visualization_node`

**Responsibility:** debug visualization only.

This node should not affect robot behavior.

#### Subscribes

```text
camera/image_color
map/global_points
parking/candidate
parking/actionable
odom
```

#### Displays

```text
RoboMaster Global Map
RoboMaster Camera - Boundary & Parking
```

The global map window shows:

```text
red points: object boundary
blue/pink points: empty-space map
cyan/yellow rectangle: detected/actionable parking slot
robot marker and heading
```

The camera window restores the original camera overlay logic:

```text
green dots: detected object-floor contact boundary
blue/orange dots: detected empty-space samples
```

---

## Topic scheme

The refactor avoids custom ROS message definitions. All complex payloads are packed into `Float32MultiArray` by helper functions in:

```text
robomaster_example/logic/topic_codec.py
```

Do not manually parse these arrays in node code. Use the helper functions instead.

### Main topics

| Topic | Type | Publisher | Subscribers | Meaning |
|---|---|---|---|---|
| `camera/image_color` | `sensor_msgs/Image` | simulator/camera | `vision_observer_node`, `visualization_node` | RGB camera frame |
| `odom` | `nav_msgs/Odometry` | robot/simulator | all motion-aware nodes | robot pose and velocity |
| `mission/observation_request` | `std_msgs/String` | `mission_controller_node` | `vision_observer_node` | request one static observation |
| `vision/boundary_observation` | `Float32MultiArray` | `vision_observer_node` | `map_node` | one boundary/empty-space observation |
| `vision/observation_done` | `std_msgs/String` | `vision_observer_node` | `mission_controller_node` | requested observation completed |
| `map/global_points` | `Float32MultiArray` | `map_node` | `slot_detector_node`, `mission_controller_node`, `visualization_node` | accumulated global map |
| `map/stats` | `std_msgs/String` | `map_node` | optional/debug | number of map points |
| `map/reset` | `std_msgs/String` | user/debug | `map_node` | clear accumulated map |
| `parking/candidate` | `Float32MultiArray` | `slot_detector_node` | `mission_controller_node`, `visualization_node` | estimated rectangle, not necessarily safe |
| `parking/actionable` | `Float32MultiArray` | `slot_detector_node` | `mission_controller_node`, `visualization_node` | rectangle with safe reachable entry side |
| `parking/debug` | `std_msgs/String` | `slot_detector_node` | optional/debug | entry-side diagnostics |
| `parking_target` | `Float32MultiArray` | `mission_controller_node` | `controller_park4` | final ordered parking rectangle |
| `mission/state` | `std_msgs/String` | `mission_controller_node` | optional/debug | high-level mission state |
| `cmd_vel` | `geometry_msgs/Twist` | active controller | robot base | robot velocity command |

### Namespacing

When launched as:

```bash
ros2 launch robomaster_example refactored_mission.launch name:=rm0
```

relative topics resolve under `/rm0`. For example:

```text
/rm0/mission/observation_request
/rm0/vision/boundary_observation
/rm0/map/global_points
/rm0/parking/actionable
/rm0/parking_target
/rm0/cmd_vel
```

---

## Message formats

### `vision/boundary_observation`

Packed by `pack_observation()`:

```text
[
  robot_x, robot_y, robot_theta,
  n_boundary,
  boundary_x1, boundary_y1, ...,
  n_empty,
  empty_x1, empty_y1, ...
]
```

### `map/global_points`

Packed by `pack_map_points()`:

```text
[
  n_boundary,
  boundary_x1, boundary_y1, ...,
  n_empty,
  empty_x1, empty_y1, ...
]
```

### `parking/candidate` and `parking/actionable`

Packed by `pack_slot()`:

```text
[
  valid,
  actionable,
  entry_edge,
  p_x, p_y,
  center_x, center_y,
  corner_1_x, corner_1_y,
  corner_2_x, corner_2_y,
  corner_3_x, corner_3_y,
  corner_4_x, corner_4_y
]
```

`valid = 1` means a rectangle exists.

`actionable = 1` means a safe entry side was found.

### `parking_target`

Legacy format consumed by `controller_park4`:

```text
[
  top_l_x, top_l_y,
  top_r_x, top_r_y,
  bottom_l_x, bottom_l_y,
  bottom_r_x, bottom_r_y
]
```

This rectangle must already be ordered by the safe entry side.

---

## Mission logic in detail

### 1. Exploration starts in `SEE`

The mission controller stops the robot and asks the vision node to capture a static observation. The vision node enforces odometry stability and a capture delay.

This prevents a common failure mode:

```text
moving robot + delayed image/odom pair -> projected map artifacts
```

### 2. The global map is updated

The map node receives the observation and fuses it into the global point map. It publishes updated map points.

### 3. Parking candidates are evaluated

The slot detector tries to estimate a parking rectangle from the global map. A rectangle is published as `parking/candidate` if found.

The safe-entry gate then checks whether the robot can actually enter from one of the sides. If yes, the ordered rectangle is published on `parking/actionable`.

### 4. If there is no actionable slot, move around the object

The mission controller enters `ACT_ARC`.

It uses the current global boundary points as a frozen map. It does not update the map during motion.

The robot moves a short arc-like step around the nearest boundary and then returns to `SEE`.

### 5. If an actionable slot exists, validate it

The mission controller starts `VALIDATE_SLOT`.

It moves to the pre-parking point `p`, faces the slot, captures three validation views, and waits for the global-map candidate after each view.

Validation passes if at least two out of three global-map rectangle estimates are consistent.

### 6. Return to `p`

After validation, the robot returns to the pre-parking pose. This removes errors introduced by the right/left validation offsets.

### 7. Delegate final parking

The mission controller publishes `parking_target` repeatedly and then releases `cmd_vel` ownership.

`controller_park4` receives the target and completes the final maneuver.

---

## Important design decisions

### No map update during motion

The global map is updated only through requested static observations. This is deliberate. It avoids projection artifacts from unsynchronized moving camera frames.

### Candidate is not the same as actionable

A fitted rectangle is only a hypothesis. The robot must also find a safe entry side and clear approach corridor before parking validation starts.

### Validation updates the global map

Validation snapshots are not used to independently fit a new local rectangle. They enrich the global map. The rectangle is always estimated from the global map.

This avoids unstable validation rectangles caused by view-dependent local empty-space clouds.

### `cmd_vel` ownership is explicit

During exploration, `mission_controller_node` publishes `cmd_vel`.

After delegation, `controller_park4` owns `cmd_vel`.

The mission controller must not continue publishing zero velocity after parking has been delegated.

---

## Launch file

The main launch file is:

```text
robomaster_example/launch/refactored_mission.launch
```

It starts:

```text
ep_tof.launch
vision_observer_node
map_node
slot_detector_node
mission_controller_node
controller_park4
visualization_node
```

Example:

```bash
ros2 launch robomaster_example refactored_mission.launch name:=rm0
```

---

## Setup file requirements

`setup.py` must install nested Python packages and register all node entry points.

Required package discovery:

```python
from setuptools import find_packages, setup

packages=find_packages(include=[package_name, package_name + '.*'])
```

Required console scripts:

```python
'vision_observer_node = robomaster_example.nodes.vision_observer_node:main'
'map_node = robomaster_example.nodes.map_node:main'
'slot_detector_node = robomaster_example.nodes.slot_detector_node:main'
'mission_controller_node = robomaster_example.nodes.mission_controller_node:main'
'visualization_node = robomaster_example.nodes.visualization_node:main'
'controller_park4 = robomaster_example.nodes.controller_park4:main'
```

A backward-compatible alias may also be installed:

```python
'controller_node = robomaster_example.nodes.mission_controller_node:main'
```

---

## Debugging commands

List topics:

```bash
ros2 topic list | grep rm0
```

Inspect mission state:

```bash
ros2 topic echo /rm0/mission/state
```

Inspect map statistics:

```bash
ros2 topic echo /rm0/map/stats
```

Inspect parking debug output:

```bash
ros2 topic echo /rm0/parking/debug
```

Inspect actionable slot messages:

```bash
ros2 topic echo /rm0/parking/actionable
```

Check whether `controller_park4` receives the final target:

```bash
ros2 topic echo /rm0/parking_target
```

Check who publishes velocity commands:

```bash
ros2 topic info /rm0/cmd_vel
```

During final parking, there should not be competing publishers that repeatedly send zero velocity.

---

## Typical failure modes and where to look

### Map contains projection artifacts

Likely cause:

```text
observation taken before robot was stable
```

Check logs from:

```text
vision_observer_node
```

Look for:

```text
[VISION] Waiting for stable odom
[VISION] Settling
[VISION] Captured ...
```

### Rectangle appears but robot does not validate

Likely cause:

```text
parking candidate exists but no safe entry side exists
```

Check:

```bash
ros2 topic echo /rm0/parking/debug
```

Look for side rejection reasons:

```text
wall_support_high
p_too_close
approach_blocked
entry_blocked
```

### Validation fails even when the slot looks good

Check whether the global-map rectangles after validation captures are stable.

Mission logs should show:

```text
validation center rectangle
validation right rectangle
validation left rectangle
2-of-3 stability result
```

If validation estimates are missing, the map/slot detector update may not be arriving before the mission controller checks it.

### Robot twitches during final parking

Likely cause:

```text
cmd_vel conflict between mission_controller_node and controller_park4
```

After successful delegation, only `controller_park4` should publish motion commands. The mission controller should release `cmd_vel` ownership.

### Robot tries to enter from the wrong side

Check whether `parking_target` is ordered correctly:

```text
[top_l, top_r, bottom_l, bottom_r]
```

The entry side is:

```text
bottom_l -> bottom_r
```

`controller_park4` should trust this ordering and must not reselect the closest edge.

---

## Development notes

- Keep pure geometry and map logic in `robomaster_example/logic/`.
- Keep ROS subscription/publishing code in `robomaster_example/nodes/`.
- Keep image-processing/projection code in `robomaster_example/vision/`.
- Avoid adding custom message types unless the project grows substantially. For now, `topic_codec.py` centralizes all array packing/unpacking.
- Avoid splitting every small behavior into a node. The current split is by data ownership: vision, map, slot detection, mission decision, final parking, visualization.

---

## Summary

The final system is organized around this data flow:

```text
camera + odom
    -> vision_observer_node
    -> vision/boundary_observation
    -> map_node
    -> map/global_points
    -> slot_detector_node
    -> parking/candidate + parking/actionable
    -> mission_controller_node
    -> parking_target
    -> controller_park4
    -> cmd_vel
```

The core behavioral loop is:

```text
observe while stopped
update global map
detect candidate slot
check safe entry side
if no actionable slot: move around boundary in a short arc
if actionable slot: validate from three views
return to pre-parking point
delegate final parking
```

The primary launch command is:

```bash
ros2 launch robomaster_example refactored_mission.launch name:=rm0
```
