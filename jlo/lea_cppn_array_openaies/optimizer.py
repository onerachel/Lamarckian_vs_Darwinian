import logging
import math
import pickle
from random import Random
from typing import List, Tuple

import multineat
import numpy as np
import sqlalchemy
from genotype import Genotype, GenotypeSerializer, crossover, mutate
from pyrr import Quaternion, Vector3
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
from jlo.array_genotype.genotype_schema import DbArrayGenotype, DbArrayGenotypeItem
from jlo.array_genotype.array_genotype import ArrayGenotypeSerializer, ArrayGenotype
from revolve2.actor_controller import ActorController
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.optimization.ea.generic_ea import EAOptimizer
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from revolve2.runners.isaacgym import LocalRunner
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import (
    develop_v1 as body_develop,
)
from revolve2.core.modular_robot import ModularRobot, Body
from revolve2.actor_controllers.cpg import CpgNetworkStructure, CpgIndex
from ESoptimizer import ESOptimizer
import numpy.typing as npt
from revolve2.genotypes.cppnwin import Genotype as CppnwinGenotype


class Optimizer(EAOptimizer[Genotype, float]):
    _process_id: int

    _runner: Runner

    _controllers: List[ActorController]

    _innov_db_body: multineat.InnovationDatabase
    # _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int

    async def ainit_new(
            # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            initial_population: List[Genotype],
            rng: Random,
            innov_db_body: multineat.InnovationDatabase,
            # innov_db_brain: multineat.InnovationDatabase,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            num_generations: int,
            offspring_size: int,
    ) -> None:
        await super().ainit_new(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
            offspring_size=offspring_size,
            initial_population=initial_population,
        )

        self._process_id = process_id
        self._init_runner()
        self._innov_db_body = innov_db_body
        # self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        # create database structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            rng: Random,
            innov_db_body: multineat.InnovationDatabase,
            # innov_db_brain: multineat.InnovationDatabase,
    ) -> bool:
        if not await super().ainit_from_database(
                database=database,
                session=session,
                process_id=process_id,
                process_id_gen=process_id_gen,
                genotype_type=Genotype,
                genotype_serializer=GenotypeSerializer,
                fitness_type=float,
                fitness_serializer=FloatSerializer,
        ):
            return False

        self._process_id = process_id
        self._init_runner()

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.process_id == process_id)
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
        # self._innov_db_brain = innov_db_brain
        # self._innov_db_brain.Deserialize(opt_row.innov_db_brain)

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=True)  # TURN OFF SIMULATOR ->True

    def _select_parents(
            self,
            population: List[Genotype],
            fitnesses: List[float],
            num_parent_groups: int,
    ) -> List[List[int]]:
        return [
            selection.multiple_unique(
                population,
                fitnesses,
                2,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
            self,
            old_individuals: List[Genotype],
            old_fitnesses: List[float],
            new_individuals: List[Genotype],
            new_fitnesses: List[float],
            num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        assert len(old_individuals) == num_survivors

        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        assert len(parents) == 2
        return crossover(parents[0], parents[1], self._rng)

    def _mutate(self, genotype: Genotype) -> Genotype:
        return mutate(genotype, self._innov_db_body, self._rng)

    async def _evaluate_generation(
            self,
            genotypes: List[Genotype],
            database: AsyncEngine,
            process_id: int,
            process_id_gen: ProcessIdGen,
    ) -> List[float]:
        NUM_GENERATIONS = 5
        POPULATION_SIZE = 20
        SIGMA = 0.1
        LEARNING_RATE = 0.05
        grid_size = 22

        body_genotypes = [genotype.body for genotype in genotypes]
        brain_genotypes = [genotype.brain.genotype for genotype in genotypes]
        before = await self._get_robot_fitnesses(
            body_genotypes, brain_genotypes
        )

        learned_brain_genotypes: List[npt.NDArray[np.float_]] = []
        for body_genotype, brain_genotype in zip(body_genotypes, brain_genotypes):
            body = body_develop(body_genotype)
            hinges = body.find_active_hinges()
            params = []
            for hinge in hinges:
                pos = body.grid_position(hinge)
                params.append(brain_genotype[int(pos[0] + pos[1] * grid_size + grid_size ** 2 / 2)])

            cpg_structure = self._make_cpg_structure(body)

            new_proc_id = process_id_gen.gen()
            maybe_optimizer = await ESOptimizer.from_database(
                database=database,
                process_id=new_proc_id,
                process_id_gen=process_id_gen,
                rng=self._rng,
                robot_body=body_develop(body_genotype),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=NUM_GENERATIONS,
                cpg_structure=cpg_structure,
            )
            if maybe_optimizer is not None:
                logging.info(
                    f"Recovered. Last finished generation: {maybe_optimizer.generation_number}."
                )
                optimizer = maybe_optimizer
            else:
                logging.info(f"No recovery data found. Starting at generation 0.")
                optimizer = await ESOptimizer.new(
                    database,
                    new_proc_id,
                    process_id_gen,
                    self._rng,
                    POPULATION_SIZE,
                    SIGMA,
                    LEARNING_RATE,
                    body_develop(body_genotype),
                    initial_mean=params,
                    simulation_time=self._simulation_time,
                    sampling_frequency=self._sampling_frequency,
                    control_frequency=self._control_frequency,
                    num_generations=NUM_GENERATIONS,
                    cpg_structure=cpg_structure,
                )

            learned_weights = await optimizer.run()  # mean

            improved_genotype = np.copy(brain_genotype)

            for hinge, learned_weight in zip(hinges, learned_weights):
                pos = body.grid_position(hinge)
                improved_genotype[int(pos[0] + pos[1] * grid_size + grid_size ** 2 / 2)] = learned_weight

            learned_brain_genotypes.append(improved_genotype)

        after = await self._get_robot_fitnesses(body_genotypes, learned_brain_genotypes)

        async with AsyncSession(database) as session:
            async with session.begin():
                dbbrain_before_ids = await ArrayGenotypeSerializer.to_database(session,
                                                                               [ArrayGenotype(params) for params in
                                                                                brain_genotypes])
                dbbrain_after_ids = await ArrayGenotypeSerializer.to_database(session,
                                                                              [ArrayGenotype(params) for params in
                                                                               learned_brain_genotypes])
                dbindividuals = [DbBigLoopIndividual(
                    process_id=self._process_id,
                    gen_num=self.generation_index,
                    gen_index=i,
                    brain_before=brain_before_id,
                    brain_after=brain_after_id,
                    before_fitness=before_fitness,
                    after_fitness=after_fitness)
                    for i,
                        (brain_before_id,
                         brain_after_id,
                         before_fitness,
                         after_fitness) in enumerate(zip(dbbrain_before_ids,
                                                         dbbrain_after_ids,
                                                         before, after))]
                
                session.add_all(dbindividuals)

        return after

    def _make_cpg_structure(self, body: Body) -> CpgNetworkStructure:
        hinges = body.find_active_hinges()
        cpgs = [CpgIndex(i) for i, _ in enumerate(hinges)]
        cpg_structure = CpgNetworkStructure(cpgs, set())
        return cpg_structure

    async def _get_robot_fitnesses(
            self,
            body_genotypes: List[CppnwinGenotype],
            brain_genotypes: List[npt.NDArray[np.float_]],
    ) -> List[float]:
        grid_size = 22

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        self._controllers = []

        for body_genotype, brain_weights in zip(body_genotypes, brain_genotypes):
            body = body_develop(body_genotype)
            param_grid = brain_weights

            hinges = body.find_active_hinges()
            cpg_structure = self._make_cpg_structure(body)

            params = []
            for hinge in hinges:
                pos = body.grid_position(hinge)
                params.append(param_grid[int(pos[0] + pos[1] * grid_size + grid_size ** 2 / 2)])

            initial_state = cpg_structure.make_uniform_state(
                0.5 * math.pi / 2.0
            )
            weight_matrix = cpg_structure.make_weight_matrix_from_params(
                params
            )
            dof_ranges = cpg_structure.make_uniform_dof_ranges(1.0)
            brain = BrainCpgNetworkStatic(
                initial_state,
                cpg_structure.num_cpgs,
                weight_matrix,
                dof_ranges,
            )

            robot = ModularRobot(body, brain)

            actor, controller = robot.make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            self._controllers.append(controller)
            env = Environment()
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()]
                )
            )
            batch.environments.append(env)

        states = await self._runner.run_batch(batch)

        return [
            self._calculate_fitness(
                states[0].envs[i].actor_states[0],
                states[-1].envs[i].actor_states[0],
            )
            for i in range(len(body_genotypes))
        ]

    def _control(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())

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
                process_id=self._process_id,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_body=self._innov_db_body.Serialize(),
                # innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=self._num_generations,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    __tablename__ = "two_optimizers"

    process_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    innov_db_body = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    # innov_db_brain = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    num_generations = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)


# class DbOpenaiESOptimizer(DbBase):
#     __tablename__ = "openaies_optimizer"
#
#     process_id = sqlalchemy.Column(
#         sqlalchemy.Integer,
#         nullable=False,
#         unique=True,
#         primary_key=True,
#     )
#     population_size = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
#     sigma = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
#     learning_rate = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
#     initial_mean = sqlalchemy.Column(
#         sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
#     )
#     initial_rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
#
#
# class DbOpenaiESOptimizerState(DbBase):
#     __tablename__ = "openaies_optimizer_state"
#
#     process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
#     gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
#     mean = sqlalchemy.Column(
#         sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
#     )
#     rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
#
#
class DbBigLoopIndividual(DbBase):
    __tablename__ = "big_loop_individual"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    brain_before = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbArrayGenotype.id), nullable=False
    )
    brain_after = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbArrayGenotype.id), nullable=False
    )
    # brain_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey(DbArrayGenotype.id), nullable=False)
    # brain_before_geno = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    # brain_after_geno = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    before_fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    after_fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
