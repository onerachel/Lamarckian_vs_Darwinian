"""
Rerun(watch) a modular robot in isaac gym.
"""

from pyrr import Quaternion, Vector3

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from jlo.drl_evolved_body.rl_runner import LocalRunner
import torch
from jlo.drl_evolved_body.config import NUM_OBSERVATIONS

from jlo.drl_evolved_body.rl_brain import RLbrain
from jlo.drl_evolved_body.rl_agent import develop, make_agent


class AgentRerunner:
    _controller: ActorController

    async def rerun(self, ) -> None:
        batch = Batch(
            simulation_time=10,
            sampling_frequency=5,
            control_frequency=5,
            control=self._control,
        )

        agent = make_agent()
        brain = RLbrain(from_checkpoint=True)
        agent.brain = brain
        actor, self._controller = develop(agent).make_actor_and_controller()

        bounding_box = actor.calc_aabb()
        env = Environment()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, bounding_box.size.z / 2.0 - bounding_box.offset.z, ]),
                Quaternion(),
                [0.0 for _ in range(8)],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(LocalRunner.SimParams())
        await runner.run_batch(batch)

    def _control(self, dt: float, control: ActorControl, observations) -> None:
        agent_obs = [[] for _ in range(NUM_OBSERVATIONS)]
        action, value, logp = self._controller.get_dof_targets(observations)
        control.set_dof_targets(0, 0, torch.clip(action, -0.8, 0.8))


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )
