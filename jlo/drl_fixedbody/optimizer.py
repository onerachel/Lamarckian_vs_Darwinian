from runner_train_mujoco import LocalRunnerTrain
import torch
from random import Random
from typing import List
import csv

import numpy as np
import numpy.typing as npt
from config import ACTION_CONSTRAINT, NUM_ITERATIONS, NUM_OBSERVATIONS
from brain import PPObrain
from revolve2.core.modular_robot import Body
from pyrr import Quaternion, Vector3

from revolve2.actor_controller import ActorController
from revolve2.core.physics.actor import Actor
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
import sys

import os


class PPOOptimizer():

    _runner: Runner

    _body: Body
    _actor: Actor
    _dof_ids: List[int]
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
        robot_body: Body,
    ) -> None:
        
        self._visualize = visualize
        print("torch" in sys.modules)
        self._init_runner()
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_agents = num_agents
        self._body = robot_body
        self._actor, self._dof_ids = self._body.to_actor()

    def _init_runner(self) -> None:
        self._runner = LocalRunnerTrain(headless=(not self._visualize))

    def _control(self, environment_index: int, dt: float, control: ActorControl, observations):
        action, value, logp = self._controller.get_dof_targets([torch.tensor(obs) for obs in observations])
        control.set_dof_targets(0, torch.clip(action, -ACTION_CONSTRAINT, ACTION_CONSTRAINT))
        # controller.train() TODO
        return action.tolist(), value.item(), logp.item()

    async def train(self, from_checkpoint: bool = False):
        """
        Create the agents, insert them in the simulation and run it
        args:
            agents: list of agents to simulate
            from_checkpoint: if True resumes training from the last checkpoint
        """

        # prepare file to log statistics
        if not os.path.exists('ppo_model_states'):
            os.makedirs('ppo_model_states')
        
        if not from_checkpoint:
            with open('ppo_model_states/statistics.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['mean_rew','mean_val'])

        # all parallel agents share the same brain
        brain = PPObrain(from_checkpoint=from_checkpoint)
        self._controller = brain.make_controller(self._body, self._dof_ids)

        restart_pos = None

        for _ in range(NUM_ITERATIONS):

            batch = Batch(
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                control=self._control,
            )

            # insert agents in the simulation environment
            bounding_box = self._actor.calc_aabb()
            for _ in range(self._num_agents):
                env = Environment()
                env.actors.append(
                    PosedActor(
                        self._actor,
                        Vector3(
                            [
                                0.0,
                                0.0,
                                bounding_box.size.z / 2.0,
                            ]
                        ),
                        Quaternion(),
                        [0.0 for _ in range(len(self._dof_ids))],
                    )
                )
                batch.environments.append(env)
            
            # run the simulation
            restart_pos = await self._runner.run_batch(batch, self._controller, self._num_agents, restart_pos)

        return 