from typing import List

import torch
from brain import PPObrain
from config import (ACTION_CONSTRAINT, CONTROL_FREQUENCY, NUM_OBSERVATIONS,
                    SAMPLING_FREQUENCY, SIMULATION_TIME)
from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.physics.running import (ActorControl, Batch, Environment,
                                           PosedActor)
from runner_mujoco import LocalRunner


class AgentRerunner:
    _controller: ActorController

    _body: Body
    _dof_ids: List[int]
    _actor: ActorController

    async def rerun(self, body: Body, file_path) -> None:
        batch = Batch(
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            control=self._control,
        )

        self._body = body
        self._actor, self._dof_ids = self._body.to_actor()
        brain = PPObrain(from_checkpoint=True)
        self._controller = brain.make_controller(self._body, self._dof_ids, file_path)

        bounding_box = self._actor.calc_aabb()
        env = Environment()
        env.actors.append(
            PosedActor(
                self._actor,
                Vector3([0.0, 0.0, bounding_box.size.z / 2.0 - bounding_box.offset.z,]),
                Quaternion(),
                [0.0 for _ in range(len(self._dof_ids))],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner()
        await runner.run_batch(batch, )

    def _control(self, environment_index: int, dt: float, control: ActorControl, observations) -> None:
        action, _, _ = self._controller.get_dof_targets([torch.tensor(obs) for obs in observations])
        control.set_dof_targets(0, torch.clip(action, -ACTION_CONSTRAINT, ACTION_CONSTRAINT))


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )
