
import math
from random import Random
from typing import List
import torch

import numpy as np
import numpy.typing as npt
from pyrr import Quaternion, Vector3
from brain import RevDENNbrain
from revde_optimizer import RevDEOptimizer
from revolve2.actor_controller import ActorController
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.modular_robot import Body
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.physics.actor import Actor
from revolve2.core.physics.running import (ActorControl, ActorState, Batch,
                                           Environment, PosedActor, Runner)
from runner_mujoco import LocalRunner
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession

from network import Actor
from config import ACTION_CONSTRAINT, NUM_OBS_TIMES


class Optimizer(RevDEOptimizer):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    _body: Body
    _actor: Actor
    _dof_ids: List[int]
    _network_structure: CpgNetworkStructure

    _runner: Runner
    _controllers: List[ActorController]

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        population_size: int,
        robot_body: Body,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        scaling: float,
        cross_prob: float,
    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database during initialization.
        :param process_id: Unique identifier in the completely program specifically made for this optimizer.
        :param process_id_gen: Can be used to create more unique identifiers.
        :param rng: Random number generator.
        :param population_size: Population size for the OpenAI ES algorithm.
        :param sigma: Standard deviation for the OpenAI ES algorithm.
        :param learning_rate: Directional vector gain for OpenAI ES algorithm.
        :param robot_body: The body to optimize the brain for.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        """
        self._body = robot_body
        self._init_actor_and_network_structure()

        parameters = self._network_structure.parameters()
        vector = torch.nn.utils.parameters_to_vector(parameters)
        stdv = 1. / math.sqrt(vector.shape[0])
        initial_population = (-stdv - stdv) * torch.rand((population_size, vector.shape[0])) + stdv

        await super().ainit_new(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            rng=rng,
            population_size=population_size,
            initial_population=initial_population,
            scaling=scaling,
            cross_prob=cross_prob,
        )

        self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        robot_body: Body,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
    ) -> bool:
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param process_id: Unique identifier in the completely program specifically made for this optimizer.
        :param process_id_gen: Can be used to create more unique identifiers.
        :param rng: Random number generator.
        :param robot_body: The body to optimize the brain for.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        :returns: True if this complete object could be deserialized from the database.
        """
        if not await super().ainit_from_database(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            rng=rng,
        ):
            return False

        self._body = robot_body
        self._init_actor_and_network_structure()

        self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        return True

    def _init_actor_and_network_structure(self) -> None:
        self._actor, self._dof_ids = self._body.to_actor()
        active_hinges_unsorted = self._body.find_active_hinges()
        active_hinge_map = {
            active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
        }
        active_hinges = [active_hinge_map[id] for id in self._dof_ids]

        actor = Actor((len(active_hinges)*NUM_OBS_TIMES, 4,), len(active_hinges))

        self._network_structure = actor

    def _init_runner(self) -> None:
        self._runner = LocalRunner(headless=True)

    async def _evaluate_population(
        self,
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        self._controllers = []

        for params in population:
            
            brain = RevDENNbrain()
            controller = brain.make_controller(self._body, self._dof_ids, params)
            controller.load_parameters(params)

            bounding_box = self._actor.calc_aabb()
            self._controllers.append(controller)
            env = Environment()
            env.actors.append(
                PosedActor(
                    self._actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in range(len(self._dof_ids))],
                )
            )
            batch.environments.append(env)

        batch_results = await self._runner.run_batch(batch)

        return np.array(
            [
                self._calculate_fitness(
                    environment_result.environment_states[0].actor_states[0],
                    environment_result.environment_states[-1].actor_states[0],
                )
                for environment_result in batch_results.environment_results
            ]
        )

    def _control(self, environment_index: int, dt: float, control: ActorControl, observations):
        controller = self._controllers[environment_index]
        action = controller.get_dof_targets([torch.tensor(obs) for obs in observations])
        control.set_dof_targets(0, torch.clip(action, -ACTION_CONSTRAINT, ACTION_CONSTRAINT))
        return action.tolist()

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        return math.sqrt(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_number != self._num_generations
