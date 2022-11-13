from __future__ import annotations

import csv
import logging
from asyncore import write
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from network import Actor
from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData
from torch.optim import Adam


class NNcontroller(ActorController):
    _num_input_neurons: int
    _num_output_neurons: int
    _dof_ranges: npt.NDArray[np.float_]
    _file_path: str

    _actor: Actor

    def __init__(
        self,
        file_path: str,
        actor: Actor,
        dof_ranges: npt.NDArray[np.float_],
        from_checkpoint: bool = False,
    ):
        """
        The controller for an agent
        args:
            actor: Neural Network controlling the agents
            dof_ranges: value range for the agent motors
            from_checkpoint: if True, resumes training from the last checkpoint
        """
        self._actor = actor
        self._file_path = file_path
        if from_checkpoint:
            checkpoint = torch.load(self._file_path + "/last_checkpoint")
            self._iteration_num = checkpoint['iteration']
            self._actor.load_state_dict(checkpoint['model_state'])
        self._dof_ranges = dof_ranges

    def get_dof_targets(self, observation) -> List[float]:
        """
        Get the target position for the motors of the body
        """
        action_prob = self._actor(observation)
        action = action_prob.sample()
        action = torch.clip(action, -0.8, 0.8)
        return action
    
    # TODO
    def step(self, dt: float):
        return

    # TODO
    def serialize(self) -> StaticData:
        return {
            "num_input_neurons": self._num_input_neurons,
            "num_output_neurons": self._num_output_neurons,
            "dof_ranges": self._dof_ranges.tolist(),
        }

    # TODO
    @classmethod
    def deserialize(cls, data: StaticData) -> NNcontroller:
        if (
            not type(data) == dict
            or not "actor_state" in data
            or not "encoder_state" in data
            or not "num_input_neurons" in data
            or not "num_output_neurons" in data
            or not "dof_ranges" in data
            or not all(type(r) == float for r in data["dof_ranges"])
        ):
            raise SerializeError()

        in_dim = data["num_input_neurons"]
        out_dim = data["num_output_neurons"]
        actor = Actor(in_dim, out_dim)
        actor.load_state_dict(data["actor_state"])
        actor = actor
        return NNcontroller(
            actor,
            np.array(data["dof_ranges"]),
        )


    def load_parameters(self, params):
        torch.nn.utils.vector_to_parameters(params, [p for p in self._actor.parameters()])