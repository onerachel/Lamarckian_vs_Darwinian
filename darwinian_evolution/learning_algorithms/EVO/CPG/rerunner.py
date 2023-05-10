"""Rerun(watch) a modular robot in Mujoco."""

from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ModularRobot
from .runner_mujoco import LocalRunner
from .environment_steering_controller import EnvironmentActorController
from revolve2.core.physics.running import Batch, Environment, PosedActor
from revolve2.core.physics import Terrain
from revolve2.core.physics.running import RecordSettings
import numpy as np
from typing import Optional

class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    async def rerun(self, 
                    robot: ModularRobot, 
                    control_frequency: float,
                    terrain: Terrain,
                    record_dir: Optional[str], 
                    record: bool = False) -> None:
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
        env.static_geometries.extend(terrain.static_geometry)
        batch.environments.append(env)

        runner = LocalRunner(headless=False)
        rs = None
        if record:
            rs = RecordSettings(record_dir)
        await runner.run_batch(batch, rs)



if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )