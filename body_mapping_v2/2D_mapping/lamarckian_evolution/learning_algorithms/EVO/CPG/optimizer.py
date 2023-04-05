
import math
from random import Random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from pyrr import Quaternion, Vector3
from .revde_optimizer import RevDEOptimizer
from revolve2.actor_controllers.cpg import CpgNetworkStructure, Cpg
from revolve2.core.modular_robot import Body
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
from revolve2.core.optimization import DbId
from revolve2.core.physics.actor import Actor
from .environment_steering_controller import EnvironmentActorController
from revolve2.core.physics.running import (ActorState, Batch,
                                           Environment, PosedActor, Runner)
from .runner_mujoco import LocalRunner
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from array_genotype.array_genotype import ArrayGenotype
from array_genotype.array_genotype_mutation import mutate as brain_mutation
from revolve2.standard_resources import terrains

class Optimizer(RevDEOptimizer):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    _TERRAIN = terrains.flat()

    _body: Body
    _actor: Actor
    _dof_ids: List[int]
    _cpg_network_structure: CpgNetworkStructure

    _runner: Runner

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int
    _target_points: List[Tuple[float]]

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
        population_size: int,
        robot_body: Body,
        inherited_brain: List,
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
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
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
        self._init_actor_and_cpg_network_structure()

        inherited_brain = np.array(inherited_brain)
        initial_population = np.repeat(np.expand_dims(inherited_brain, axis=0), population_size, axis=0)
        gaussian_noise = np.random.normal(scale=0.5, size=[len(initial_population)-1, inherited_brain.shape[0]])
        initial_population[1:] += gaussian_noise
        
        await super().ainit_new(
            rng=rng,
            population_size=population_size,
            initial_population=initial_population,
            scaling=scaling,
            cross_prob=cross_prob,
        )

        self._runner = self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._target_points = [(1., -1.), (0., -2.)]

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
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
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
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
            db_id=db_id,
            rng=rng,
        ):
            return False

        self._body = robot_body
        self._init_actor_and_cpg_network_structure()

        self._runner = self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        return True

    def _init_actor_and_cpg_network_structure(self) -> None:
        self._actor, self._dof_ids = self._body.to_actor()
        active_hinges_unsorted = self._body.find_active_hinges()
        active_hinge_map = {
            active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
        }
        active_hinges = [active_hinge_map[id] for id in self._dof_ids]
        cpg_network_structure = make_cpg_network_structure_neighbour(
            active_hinges
        )
        self._cpg_network_structure = cpg_network_structure


    def _init_runner(self, num_simulators: int = 1) -> None:
        return LocalRunner(headless=True, num_simulators=num_simulators)

    async def _evaluate_population(
        self,
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )

        self._runner = self._init_runner(population.shape[0])

        for params in population:
            initial_state = self._cpg_network_structure.make_uniform_state(
                0.5 * math.pi / 2.0
            )
            weight_matrix = (
                self._cpg_network_structure.make_connection_weights_matrix_from_params(
                    params
                )
            )
            dof_ranges = self._cpg_network_structure.make_uniform_dof_ranges(1.0)
            brain = BrainCpgNetworkStatic(
                initial_state,
                self._cpg_network_structure.num_cpgs,
                weight_matrix,
                dof_ranges,
            )
            controller = brain.make_controller(self._body, self._dof_ids)

            bounding_box = self._actor.calc_aabb()
            env = Environment(EnvironmentActorController(controller, self._target_points, steer=True))
            env.static_geometries.extend(self._TERRAIN.static_geometry)
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
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            batch.environments.append(env)

        batch_results = await self._runner.run_batch(batch)

        return np.array(
            [
                self._calculate_point_navigation(
                    environment_result, self._target_points
                )
                for environment_result in batch_results.environment_results
            ]
        )

    @staticmethod
    def _calculate_point_navigation(results, targets) -> float:
        trajectory = [(0.0, 0.0)] + targets
        distances = [compute_distance(trajectory[i], trajectory[i-1]) for i in range(1, len(trajectory))]
        target_range = 0.1
        reached_target_counter = 0
        coordinates = [env_state.actor_states[0].position[:2] for env_state in results.environment_states]
        lengths = [compute_distance(coordinates[i-1], coordinates[i]) for i in range(1,len(coordinates))]
        starting_idx = 0
        for idx, state in enumerate(coordinates):
            if reached_target_counter < len(targets) and check_target(state, targets[reached_target_counter], target_range):
                reached_target_counter += 1
                starting_idx = idx
        
        fitness = 0
        if reached_target_counter > 0:
            path_len = sum(lengths[:starting_idx])
            fitness = sum(distances[:reached_target_counter]) - 0.1*path_len
        if reached_target_counter == len(targets):
            return fitness
        else:
            if reached_target_counter == 0:
                last_target = (0.0, 0.0)
            else:
                last_target = trajectory[reached_target_counter]
            last_coord = coordinates[-1]
            distance = compute_distance(targets[reached_target_counter], last_target)
            distance -= compute_distance(targets[reached_target_counter], last_coord)
            new_path_len = sum(lengths[:]) - sum(lengths[:starting_idx])
            return fitness + (distance - 0.1*new_path_len)


    @staticmethod
    def _calculate_panoramic_rotation(results, vertical_angle_limit = math.pi/4) -> float:
        total_angle = 0.0

        orientations = [env_state.actor_states[0].orientation for env_state in results.environment_states[1:]]
        directions = [compute_directions(o) for o in orientations][1:]

        vertical_limit = math.sin(vertical_angle_limit)

        z_vals = np.argsort([abs(d.x) for d in directions[0]])
        chosen_orientation = z_vals[0]

        for i in range(1, len(directions)):
            u = directions[i-1][chosen_orientation]
            v = directions[i][chosen_orientation]
            
            if abs(u.x) > vertical_limit:
                return total_angle
            
            dot = u.z*v.z + u.y*v.y
            det = u.z*v.y - u.y*v.z
            delta = math.atan2(det, dot)

            total_angle += delta

        return total_angle

    def _must_do_next_gen(self) -> bool:
        return self.generation_number != self._num_generations

def compute_directions(q: Quaternion):
    vi = Vector3()
    vi.x = 1 - 2*(q.y**2+q.z**2)
    vi.y = 2*(q.x*q.y - q.z*q.w)
    vi.z = 2*(q.x*q.z + q.y*q.w)
    
    vj = Vector3()
    vj.x = 2*(q.x*q.y + q.z*q.w)
    vj.y = 1 - 2*(q.x**2 + q.z**2)
    vj.z = 2*(q.y*q.z - q.x*q.w)

    vk = Vector3()
    vk.x = 2*(q.x*q.z - q.y*q.w)
    vk.y = 2*(q.y*q.z + q.x*q.w)
    vk.z = 1 - 2*(q.x**2 + q.y**2)
    return vi, vj, vk

def check_target(coord, target, target_range):
    if abs(coord[0]-target[0]) < target_range and abs(coord[1]-target[1]) < target_range:
        return True
    else:
        return False

def compute_distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2
    )