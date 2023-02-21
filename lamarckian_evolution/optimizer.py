"""Optimizer for finding a good modular robot body and brain using CPPNWIN genotypes and simulation using mujoco."""

import math
import pickle
from random import Random
from typing import List, Tuple
import numpy as np

import multineat
import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
import sqlalchemy
from genotype import Genotype, GenotypeSerializer, crossover, mutate
from pyrr import Quaternion, Vector3
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer
from revolve2.core.optimization import DbId
from _optimizer import EAOptimizer
#from revolve2.core.optimization.ea.generic_ea import EAOptimizer
from revolve2.core.physics.running import (
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from learning_algorithms.EVO.CPG.optimize import main as learn_controller
from revolve2.runners.mujoco import LocalRunner
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
import asyncio
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import (
    develop_v1 as body_develop,)

from revolve2.core.modular_robot import Body, Brain
from learning_algorithms.EVO.CPG.optimizer import Optimizer as ControllerOptimizer
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
import logging

class Optimizer(EAOptimizer[Genotype, float]):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    _db_id: DbId

    _runner: Runner

    _innov_db_body: multineat.InnovationDatabase
    _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int
    _grid_size: int
    _num_potential_joints: int

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        initial_population: List[Genotype],
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        offspring_size: int,
        grid_size: int
    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param initial_population: List of genotypes forming generation 0.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database for the body genotypes.
        :param innov_db_brain: Innovation database for the brain genotypes.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        :param offspring_size: Number of offspring made by the population each generation.
        """
        await super().ainit_new(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
            offspring_size=offspring_size,
            initial_population=initial_population,
        )

        self._db_id = db_id
        self._init_runner()
        self._innov_db_body = innov_db_body
        self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._grid_size = grid_size
        self._num_potential_joints = ((grid_size**2)-1)

        # create database structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        offspring_size: int,
    ) -> bool:
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param process_id: Unique identifier in the completely program specifically made for this optimizer.
        :param process_id_gen: Can be used to create more unique identifiers.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database for the body genotypes.
        :param innov_db_brain: Innovation database for the brain genotypes.
        :returns: True if this complete object could be deserialized from the database.
        :raises IncompatibleError: In case the database is not compatible with this class.
        """
        if not await super().ainit_from_database(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
            offspring_size=offspring_size
        ):
            return False

        self._db_id = db_id
        self._init_runner()

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.db_id == self._db_id.fullname)
                    .order_by(DbOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        # if this happens something is wrong with the database
        if opt_row is None:
            raise IncompatibleError

        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = opt_row.num_generations

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))

        self._innov_db_body = innov_db_body
        self._innov_db_body.Deserialize(opt_row.innov_db_body)
        self._innov_db_brain = innov_db_brain
        self._innov_db_brain.Deserialize(opt_row.innov_db_brain)

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(headless=True)

    def _select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[float],
        num_parent_groups: int,
    ) -> List[List[int]]:
        return [
            selection.multiple_unique(
                2,
                population,
                fitnesses,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        return selection.topn(num_survivors, old_individuals, old_fitnesses)

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype], first_best: bool) -> Genotype:
        assert len(parents) == 2
        return crossover(parents[0], parents[1], self._rng, first_best)

    def _mutate(self, genotype: Genotype) -> Genotype:
        return mutate(genotype, self._innov_db_body, self._innov_db_brain, self._rng)

    async def _evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        db_id: DbId,
    ) -> Tuple[List[float], List[Genotype]]:

        final_fitnesses = []
        starting_fitnesses = []

        new_genotypes = genotypes.copy()
        body_genotypes = [genotype.body for genotype in new_genotypes]
        brain_genotypes = [genotype.brain for genotype in new_genotypes]

        for body_num, (body_genotype, brain_genotype) in enumerate(zip(body_genotypes, brain_genotypes)):
            body = body_develop(body_genotype)
            _, dof_ids = body.to_actor()
            active_hinges_unsorted = body.find_active_hinges()
            active_hinge_map = {
                active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
            }
            active_hinges = [active_hinge_map[id] for id in dof_ids]
            cpg_network_structure = make_cpg_network_structure_neighbour(
                active_hinges
            )
            brain_params = []
            for hinge in active_hinges:
                pos = body.grid_position(hinge)
                cpg_idx = int(pos[0] + pos[1] * self._grid_size + self._grid_size**2 / 2)
                brain_params.append(brain_genotype.params_array[
                    cpg_idx*14
                ])

            for connection in cpg_network_structure.connections:
                hinge1 = connection.cpg_index_highest.index
                pos1 = body.grid_position(active_hinges[hinge1])
                cpg_idx1 = int(pos1[0] + pos1[1] * self._grid_size + self._grid_size**2 / 2)
                hinge2 = connection.cpg_index_lowest.index
                pos2 = body.grid_position(active_hinges[hinge2])
                cpg_idx2 = int(pos2[0] + pos2[1] * self._grid_size + self._grid_size**2 / 2)
                rel_pos = relative_pos(pos1[:2], pos2[:2])
                idx = max(cpg_idx1, cpg_idx2)
                brain_params.append(brain_genotype.params_array[
                    idx*14 + rel_pos
                ])
                
            logging.info("Starting optimization of the controller for morphology num: " + str(body_num))
            final_fitness = 0.0
            starting_fitness = 0.0
            # check that the morphology has at least one active hinge. Otherwise the maximum fitness is 0
            if len(body.find_active_hinges()) <= 0:
                logging.info("Morphology num " + str(body_num) + " has no active hinges")
            else:
                learned_params, final_fitness, starting_fitness = await learn_controller(body, brain_params, self.generation_index, body_num)
                for hinge, learned_weight in zip(active_hinges, learned_params[:len(active_hinges)]):
                    pos = body.grid_position(hinge)
                    cpg_idx = int(pos[0] + pos[1] * self._grid_size + self._grid_size**2 / 2)
                    brain_genotype.params_array[
                        cpg_idx*14
                    ] = learned_weight

                for connection, connection_weight in zip(cpg_network_structure.connections, learned_params[len(active_hinges):]):
                    hinge1 = connection.cpg_index_highest.index
                    pos1 = body.grid_position(active_hinges[hinge1])
                    cpg_idx1 = int(pos1[0] + pos1[1] * self._grid_size + self._grid_size**2 / 2)
                    hinge2 = connection.cpg_index_lowest.index
                    pos2 = body.grid_position(active_hinges[hinge2])
                    cpg_idx2 = int(pos2[0] + pos2[1] * self._grid_size + self._grid_size**2 / 2)
                    rel_pos = relative_pos(pos1[:2], pos2[:2])
                    idx = max(cpg_idx1, cpg_idx2)
                    brain_genotype.params_array[
                        idx*14 + rel_pos
                    ] = connection_weight
                    
            final_fitnesses.append(final_fitness)
            starting_fitnesses.append(starting_fitness)

        return (starting_fitnesses, final_fitnesses), new_genotypes

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        return float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        )

    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                db_id=self._db_id.fullname,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_body=self._innov_db_body.Serialize(),
                innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=self._num_generations,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    """Optimizer state."""

    __tablename__ = "optimizer"

    db_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    innov_db_body = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    innov_db_brain = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    num_generations = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)

def relative_pos(pos1, pos2):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]

    mapping = {(1,0):1, (1,1):2, (0,1):3, (-1,0):4, (-1,-1):5, (0,-1):6,
                (-1,1):7, (1,-1):8, (2,0):9, (0,2):10, (-2,0):11, (0,-2):12, (0,0):13}
    
    return mapping[(dx,dy)]