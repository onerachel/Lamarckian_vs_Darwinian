from random import Random
from typing import List
import csv

import torch
from jlo.drl_evolved_body.config import NUM_OBSERVATIONS
from jlo.drl_evolved_body.rl_brain import RLbrain
from jlo.drl_evolved_body.rl_agent import Agent, develop
from pyrr import Quaternion, Vector3

from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from jlo.drl_evolved_body.rl_runner_train import LocalRunnerTrain


class RLOptimizer:
    _runner: Runner

    _controller: ActorController

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float
    _visualize: bool
    _num_agents: int

    def __init__(
            self,
            rng: Random,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            visualize: bool,
            num_agents: int,
    ) -> None:

        self._visualize = visualize
        self._init_runner()
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_agents = num_agents

    def _init_runner(self) -> None:
        self._runner = LocalRunnerTrain(LocalRunnerTrain.SimParams(), headless=(not self._visualize))

    def _control(self, dt: float, control: ActorControl, observations):
        num_agents = observations[0].shape[0]
        actions = []
        values = []
        logps = []

        # for each agent in the simulation make a step
        for control_i in range(num_agents):
            agent_obs = [[] for _ in range(NUM_OBSERVATIONS)]
            for i, obs in enumerate(observations):
                agent_obs[i] = obs[control_i]
            action, value, logp = self._controller.get_dof_targets(agent_obs)
            control.set_dof_targets(control_i, 0, torch.clip(action, -0.8, 0.8))
            actions.append(action.tolist())
            values.append(value.tolist())
            logps.append(logp.tolist())
        return actions, values, logps

    async def train(self, agent, from_checkpoint: bool = False):
        """
        Create the agents, insert them in the simulation and run it
        args:
            agents: list of agents to simulate
            from_checkpoint: if True resumes training from the last checkpoint
        """

        # prepare file to log statistics
        with open('./statistics.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mean_rew', 'mean_val'])

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        # all parallel agents share the same brain
        brain = RLbrain(from_checkpoint=from_checkpoint)

        # insert agents in the simulation environment
        agent.brain = brain
        actor, controller = develop(agent).make_actor_and_controller()
        self._controller = controller
        bounding_box = actor.calc_aabb()
        for _ in range(self._num_agents):
            env = Environment()
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in range(len(actor.joints))]
                )
            )
            batch.environments.append(env)

        # run the simulation
        await self._runner.run_batch(batch, self._controller, self._num_agents)

        return