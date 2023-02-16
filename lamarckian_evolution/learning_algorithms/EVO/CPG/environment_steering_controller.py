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
            self.n = 7
            self.is_left = []
            self.is_right = []

    def control(self, dt: float, actor_control: ActorControl, joint_positions=None, current_pos=None, save_pos=False) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        :param coordinates: current coordinates of each joint
        :param current_pos: current position of the agent
        """
        
        self.actor_controller.step(dt)
        targets = self.actor_controller.get_dof_targets()

        if self.steer:

            # check if the robot reached the target
            if self.reached_target_counter < len(self.target_points):
                core_position = current_pos[:2]
                if (abs(core_position[0]-self.target_points[self.reached_target_counter][0]) < self.target_range and
                    abs(core_position[1]-self.target_points[self.reached_target_counter][1]) < self.target_range):
                    self.reached_target_counter += 1

            if self.reached_target_counter < len(self.target_points):

                if save_pos:
                    for joint_pos in joint_positions[1:]:
                        self.is_left.append(joint_pos[0] > 0)
                        self.is_right.append(joint_pos[0] < 0)

                # check if joints are on the left or right
                joint_positions = [c[:2] for c in joint_positions]

                # compute steering angle and parameters
                trajectory = [(0.0, 0.0)] + self.target_points
                a0 = np.array(core_position) - trajectory[self.reached_target_counter]
                b0 = np.array(self.target_points[self.reached_target_counter]) - trajectory[self.reached_target_counter]
                theta = np.arctan2(a0[1], a0[0]) - np.arctan2(b0[1], b0[0])
                theta = (theta + np.pi) % (2*np.pi) - np.pi
                g = ((np.pi-abs(theta))/np.pi) ** self.n

                # apply steering factor
                for i, (left, right) in enumerate(zip(self.is_left, self.is_right)):
                    if left:
                        if theta < 0:
                            targets[i] *= g
                    elif right:
                        if theta >= 0:
                            targets[i] *= g
            
        actor_control.set_dof_targets(0, targets)