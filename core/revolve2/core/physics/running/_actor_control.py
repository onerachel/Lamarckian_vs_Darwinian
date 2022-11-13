from typing import List, Tuple

# new version
class ActorControl:
    """Interface for controlling degrees of freedom of actors in a simulation."""

    _dof_targets: List[Tuple[int, List[float]]]  # actor, targets

    def __init__(self) -> None:
        """Initialize this object."""
        self._dof_targets = []

    def set_dof_targets(self, actor: int, targets: List[float]) -> None:
        """
        Set the degrees of freedom of an actor.

        :param actor: The actor to control.
        :param targets: The targets to set.
        """
        self._dof_targets.append((actor, targets))


# old version
# class ActorControl:
#     _dof_targets: List[Tuple[int, int, List[float]]]
#
#     def __init__(self) -> None:
#         self._dof_targets = []
#
#     def set_dof_targets(
#         self, environment: int, actor: int, targets: List[float]
#     ) -> None:
#         self._dof_targets.append((environment, actor, targets))
