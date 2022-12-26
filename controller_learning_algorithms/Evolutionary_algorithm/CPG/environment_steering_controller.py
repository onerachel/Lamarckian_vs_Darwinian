"""Contains EnvironmentActorController, an environment controller for an environment with a single actor that uses a provided ActorController."""

from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, EnvironmentController
import numpy as np
from typing import List, Tuple


class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controller: ActorController
    target_points: List[Tuple[float]]
    reached_target_counter: int
    target_range: float
    n: int

    def __init__(self, actor_controller: ActorController,
                target_points: List[Tuple[float]] = [(0.0,0.0)],
                steer: bool = False) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for the single actor in the environment.
        :param target_points: Target points the agent have to reach.
        :param steer: if True the agent is controlled using a steering policy.
        """
        self.actor_controller = actor_controller
        self.steer = steer
        if steer:
            self.target_points = target_points
            self.reached_target_counter = 0
            self.target_range = 0.1
            self.n = 1

    def control(self, dt: float, actor_control: ActorControl, joint_coordinates=None, agent_pos=None) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        :param joint_coordinates: current coordinates of each joint
        :param agent_pos: current position of the agent
        """

        self.actor_controller.step(dt)
        targets = self.actor_controller.get_dof_targets()

        if self.steer:

            # check if joints are on the left or right
            joint_coordinates = [c[:2] for c in joint_coordinates]
            core_position = agent_pos[:2]
            is_left = [EnvironmentActorController._is_left_of(core_position, p) for p in joint_coordinates[1:]]

            # check if the robot reached the target
            if (abs(core_position[0]-self.target_points[self.reached_target_counter][0]) < self.target_range and
                abs(core_position[1]-self.target_points[self.reached_target_counter][1]) < self.target_range):
                self.reached_target_counter += 1

            # compute steering angle and parameters
            a0 = np.array(core_position) - np.array([0.0,0.0])
            b0 = np.array(self.target_points[self.reached_target_counter]) - np.array([0.0,0.0])
            theta = np.arctan2(a0[1], a0[0]) - np.arctan2(b0[1], b0[0])
            g = ((np.pi-abs(theta))/np.pi) ** self.n

            # apply steering factor
            for i, left in enumerate(is_left):
                if left:
                    if theta < 0:
                        targets[i] *= g
                else:
                    if theta >= 0:
                        targets[i] *= g
        
        actor_control.set_dof_targets(0, targets)

    @staticmethod
    def _is_left_of(agent_pos, joint_pos):
        """
        Check wether a joint is left or right of the line connecting the agent to the origin
        https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line

        :param agent_pos: position of the agent
        :joint_pos: position of the joint
        """
        return ((agent_pos[0])*(joint_pos[1]) - 
                (agent_pos[1])*(joint_pos[0])) > 0
