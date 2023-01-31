"""Rerun(watch) a modular robot in Mujoco."""

from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ModularRobot
from .runner_mujoco import LocalRunner
from .environment_steering_controller import EnvironmentActorController
from revolve2.core.physics.running import Batch, Environment, PosedActor
import math
from revolve2.core.physics.running import RecordSettings
import numpy as np
from typing import Optional

class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    async def rerun(self, robot: ModularRobot, control_frequency: float, record_dir: Optional[str], record: bool = False) -> None:
        """
        Rerun a single robot.

        :param robot: The robot the simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        """
        batch = Batch(
            simulation_time=60,
            sampling_frequency=5,
            control_frequency=control_frequency,
        )

        actor, self._controller = robot.make_actor_and_controller()

        env = Environment(EnvironmentActorController(self._controller, [(1.0, -1.0), (0.0, -2.0)], steer=True))
        bounding_box = actor.calc_aabb()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, bounding_box.size.z / 2.0 - bounding_box.offset.z]),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(headless=False)
        rs = None
        if record:
            rs = RecordSettings(record_dir)
        res = await runner.run_batch(batch, rs)
        print(self._calculate_point_navigation(res.environment_results[0], [(1.0, -1.0), (0.0, -2.0)]))

    @staticmethod
    def _calculate_point_navigation(results, targets) -> float:
        trajectory = [(0.0, 0.0)] + targets
        distances = []
        target_range = 0.1
        reached_target_counter = 0

        coordinates = [env_state.actor_states[0].position[:2] for env_state in results.environment_states]
        path_length = [compute_distance(coordinates[i-1], coordinates[i]) for i in range(1,len(coordinates))]
        starting_point = 0
        for idx, state in enumerate(coordinates[1:]):
            if reached_target_counter < len(targets) and check_target(state, targets[reached_target_counter], target_range):
                distances.append(sum(path_length[:idx]) - sum(path_length[:starting_point]))
                reached_target_counter += 1
                starting_point = idx
        
        fitness = reached_target_counter * math.sqrt(2)
        if reached_target_counter > 0:
            fitness /= sum(distances)

        if reached_target_counter == len(targets):
            return fitness
        else:
            new_origin = trajectory[reached_target_counter]
            delta = math.atan2(coordinates[-1][1] - new_origin[1], coordinates[-1][0] - new_origin[0])
            target_direction = math.atan2(targets[reached_target_counter][1] - new_origin[1], targets[reached_target_counter][0] - new_origin[0])
            theta = abs(((delta - target_direction) + math.pi) % (2*math.pi) - math.pi)
            gamma = compute_distance(coordinates[-1], trajectory[reached_target_counter])
            alpha = gamma * math.sin(theta)
            beta = gamma * math.cos(theta)
            # check to prevent that the robot goes further than the target
            max_beta = compute_distance(trajectory[reached_target_counter], trajectory[reached_target_counter+1])
            beta = min(beta, max_beta)
            
            path_len = sum(path_length) - sum(path_length[:starting_point])
            omega = 0.01
            epsilon = 10e-10

            fitness += (abs(beta)/(path_len + epsilon)) * (beta/(math.degrees(theta) + 1.0) - omega * alpha)

            return fitness


def check_target(coord, target, target_range):
    if abs(coord[0]-target[0]) < target_range and abs(coord[1]-target[1]) < target_range:
        return True
    else:
        return False

def compute_distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2
    )

if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )

