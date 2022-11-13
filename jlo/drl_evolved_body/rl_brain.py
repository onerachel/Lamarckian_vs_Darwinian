"""
Neural Network brain for Reinforcement Learning.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brain
from jlo.drl_evolved_body.rl_controller import RLcontroller
from jlo.drl_evolved_body.actor_critic_network import Actor, ActorCritic
from jlo.drl_evolved_body.config import NUM_OBS_TIMES


class RLbrain(Brain, ABC):
    """
    Brain of an agent
    """
    def __init__(self, from_checkpoint: bool = False) -> None:
        super().__init__()
        self._from_checkpoint = from_checkpoint

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        """
        Initialize neural network used as controller and return the brain
        """
        active_hinges = body.find_active_hinges()
        num_hinges = len(active_hinges)
        actor_critic = ActorCritic((len(dof_ids)*NUM_OBS_TIMES, 4), num_hinges)

        return RLcontroller(
            actor_critic,
            np.array([active_hinge.RANGE for active_hinge in active_hinges]),
            from_checkpoint=self._from_checkpoint,
        )