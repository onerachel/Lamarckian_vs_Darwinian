"""Rerun(watch) a modular robot in Mujoco."""

from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from runner_mujoco import LocalRunner
import torch
from config import ACTION_CONSTRAINT


class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    _controller: ActorController

    async def rerun(self, body, controller, control_frequency: float) -> None:
        """
        Rerun a single robot.

        :param robot: The robot the simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        """
        batch = Batch(
            simulation_time=1000000,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
            control=self._control,
        )

        actor, dof_ids = body.to_actor()
        self._controller = controller

        env = Environment()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, 0.1]),
                Quaternion(),
                [0.0 for _ in range(len(dof_ids))],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(headless=False)
        await runner.run_batch(batch)

    def _control(self, environment_index: int, dt: float, control: ActorControl, observations):
        controller = self._controller
        action = controller.get_dof_targets([torch.tensor(obs) for obs in observations])
        control.set_dof_targets(0, torch.clip(action, -ACTION_CONSTRAINT, ACTION_CONSTRAINT))
        return action.tolist()


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )
