# RoboMaster Example

Este paquete contiene el flujo de exploracion + parqueo para simulacion RoboMaster.

## Arquitectura actual

Nodos principales:

- `controller_node` (exploracion y deteccion de espacio)
	- Publica `cmd_vel` durante exploracion manual.
	- Calcula un rectangulo de parqueo (`parking_corners`).
	- Publica el objetivo en `parking_target` (`std_msgs/Float32MultiArray`, 8 valores: 4 puntos x 2 coordenadas).
	- Cuando publica `parking_target`, deja de comandar `cmd_vel` para evitar interferencia.

- `controller_park4` (parqueo)
	- Inicia en estado `IDLE`.
	- Espera mensaje en `parking_target`.
	- Al recibirlo pasa a `START` y ejecuta la maquina de estados de parqueo (`MOVE_TO_P`, `ROTATE_TO_FACE_PARKING`, `FORWARD`, `PARKED`).

## Topicos relevantes

- `cmd_vel`: comando de velocidad del robot.
- `odom`: odometria usada por ambos nodos.
- `parking_target`: objetivo de parqueo publicado por exploracion y consumido por parqueo.

## Launchers

- `mission.launch`
	- Lanza `controller_node` y `controller_park4` en el mismo namespace.
	- Es el launcher recomendado para el flujo completo.

- `controller.launch`
	- Lanza solo `controller_node`.

- `park_4points.launch`
	- Lanza solo `controller_park4`.

- `ep_tof.launch`
	- Lanza la configuracion base del robot EP con sensores TOF desde `robomaster_ros`.

## Ejecucion recomendada

1. Terminal 1: abrir simulacion

```bash
pixi run coppelia
```

En Coppelia, abrir la escena correspondiente y arrancar simulacion en tiempo real.

2. Terminal 2: levantar stack base del robot

```bash
pixi shell
source install/setup.zsh
ros2 launch robomaster_example ep_tof.launch name:=/rm0
```

3. Terminal 3: lanzar mision completa

```bash
pixi shell
colcon build --symlink-install
source install/setup.zsh
ros2 launch robomaster_example mission.launch name:=rm0
```

## Notas

- Ambos nodos deben correr en el mismo namespace para compartir `odom`, `cmd_vel` y `parking_target`.
- `mission.launch` ya configura `use_sim_time=true` para ambos nodos.