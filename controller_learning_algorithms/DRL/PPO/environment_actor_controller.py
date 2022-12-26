"""Contains EnvironmentActorController, an environment controller for an environment with a single actor that uses a provided ActorController."""

from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, EnvironmentController
import torch
from config import ACTION_CONSTRAINT


class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controller: ActorController

    def __init__(self, actor_controller: ActorController) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for the single actor in the environment.
        """
        self.actor_controller = actor_controller

    def control(self, dt: float, control: ActorControl, observations):
        action, value, logp = self.actor_controller.get_dof_targets([torch.tensor(obs) for obs in observations])
        control.set_dof_targets(0, torch.tanh(action)*ACTION_CONSTRAINT)
        return action.tolist(), value.item(), logp.item()
