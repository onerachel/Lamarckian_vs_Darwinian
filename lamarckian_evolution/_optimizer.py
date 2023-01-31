from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, TypeVar

from revolve2.core.database import IncompatibleError, Serializer
from revolve2.core.optimization import Process, DbId
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

from sqlalchemy import Column, Integer, String, Float
from revolve2.core.optimization.ea.generic_ea._database import (
    DbBase,
    DbEAOptimizer,
    DbEAOptimizerGeneration,
    DbEAOptimizerParent,
    DbEAOptimizerState,
)
from morphological_measures import MorphologicalMeasures
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import (
    develop_v1 as body_develop,)
from render.render import Render

Genotype = TypeVar("Genotype")
Fitness = TypeVar("Fitness")


class EAOptimizer(Process, Generic[Genotype, Fitness]):
    """
    A generic optimizer implementation for evolutionary algorithms.

    Inherit from this class and implement its abstract methods.
    See the `Process` parent class on how to make an instance of your implementation.
    You can run the optimization process using the `run` function.

    Results will be saved every generation in the provided database.
    """

    @abstractmethod
    async def _evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        db_id: DbId
    ) -> Tuple[Tuple[List[Fitness]], List[Genotype]]:
        """
        Evaluate a list of genotypes.

        :param genotypes: The genotypes to evaluate. Must not be altered.
        :param database: Database that can be used to store anything you want to save from the evaluation.
        :param process_id: Unique identifier in the completely program specifically made for this function call.
        :param process_id_gen: Can be used to create more unique identifiers.
        :returns: The fitness result and genotype with trained controller.
        """

    @abstractmethod
    def _select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[Fitness],
        num_parent_groups: int,
    ) -> List[List[int]]:
        """
        Select groups of parents that will create offspring.

        :param population: The generation to select sets of parents from. Must not be altered.
        :param fitnesses: Fitnesses of the population.
        :param num_parent_groups: Number of groups to create.
        :returns: The selected sets of parents, each integer representing a population index.
        """

    @abstractmethod
    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[Fitness],
        new_individuals: List[Genotype],
        new_fitnesses: List[Fitness],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Select survivors from the sets of old and new individuals, which will form the next generation.

        :param old_individuals: Original individuals.
        :param old_fitnesses: Fitnesses of the original individuals.
        :param new_individuals: New individuals.
        :param new_fitnesses: Fitnesses of the new individuals.
        :param num_survivors: How many individuals should be selected.
        :returns: Indices of the old survivors and indices of the new survivors.
        """

    @abstractmethod
    def _crossover(self, parents: List[Genotype]) -> Genotype:
        """
        Combine a set of genotypes into a new genotype.

        :param parents: The set of genotypes to combine. Must not be altered.
        :returns: The new genotype.
        """

    @abstractmethod
    def _mutate(self, genotype: Genotype) -> Genotype:
        """
        Apply mutation to an genotype to create a new genotype.

        :param genotype: The original genotype. Must not be altered.
        :returns: The new genotype.
        """

    @abstractmethod
    def _must_do_next_gen(self) -> bool:
        """
        Decide if the optimizer must do another generation.

        :returns: True if it must.
        """

    @abstractmethod
    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        """
        Save the results of this generation to the database.

        This function is called after a generation is finished and results and state are saved to the database.
        Use it to store state and results of the optimizer.
        The session must not be committed, but it may be flushed.

        :param session: The session to use for writing to the database. Must not be committed, but can be flushed.
        """

    __database: AsyncEngine
    __db_id: DbId

    __ea_optimizer_id: int

    __genotype_type: Type[Genotype]
    __genotype_serializer: Type[Serializer[Genotype]]
    __fitness_type: Type[Fitness]
    __fitness_serializer: Type[Serializer[Fitness]]

    __offspring_size: int

    __next_individual_id: int

    __latest_population: List[_Individual[Genotype]]
    __latest_fitnesses: Optional[List[Fitness]]  # None only for the initial population
    __generation_index: int

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        genotype_type: Type[Genotype],
        genotype_serializer: Type[Serializer[Genotype]],
        fitness_type: Type[Fitness],
        fitness_serializer: Type[Serializer[Fitness]],
        offspring_size: int,
        initial_population: List[Genotype],
    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database during initialization.
        :param process_id: Unique identifier in the completely program specifically made for this optimizer.
        :param process_id_gen: Can be used to create more unique identifiers.
        :param genotype_type: Type of the genotype generic parameter.
        :param genotype_serializer: Serializer for serializing genotypes.
        :param fitness_type: Type of the fitness generic parameter.
        :param fitness_serializer: Serializer for serializing fitnesses.
        :param offspring_size: Number of offspring made by the population each generation.
        :param initial_population: List of genotypes forming generation 0.
        """
        self.__database = database
        self.__genotype_type = genotype_type
        self.__genotype_serializer = genotype_serializer
        self.__fitness_type = fitness_type
        self.__fitness_serializer = fitness_serializer
        self.__offspring_size = offspring_size
        self.__db_id = db_id
        self.__next_individual_id = 0
        self.__latest_fitnesses = None
        self.__generation_index = 0

        self.__latest_population = [
            _Individual(self.__gen_next_individual_id(), g, [])
            for g in initial_population
        ]

        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await self.__genotype_serializer.create_tables(session)
        await self.__fitness_serializer.create_tables(session)

        new_opt = DbEAOptimizer(
            db_id=db_id.fullname,
            offspring_size=self.__offspring_size,
            genotype_table=self.__genotype_serializer.identifying_table(),
            fitness_table=self.__fitness_serializer.identifying_table(),
        )
        session.add(new_opt)
        await session.flush()
        assert new_opt.id is not None  # this is impossible because it's not nullable
        self.__ea_optimizer_id = new_opt.id

    async def ainit_from_database(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        genotype_type: Type[Genotype],
        genotype_serializer: Type[Serializer[Genotype]],
        fitness_type: Type[Fitness],
        fitness_serializer: Type[Serializer[Fitness]],
        offspring_size: int,
    ) -> bool:
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param process_id: Unique identifier in the completely program specifically made for this optimizer.
        :param process_id_gen: Can be used to create more unique identifiers.
        :param genotype_type: Type of the genotype generic parameter.
        :param genotype_serializer: Serializer for serializing genotypes.
        :param fitness_type: Type of the fitness generic parameter.
        :param fitness_serializer: Serializer for serializing fitnesses.
        :returns: True if this complete object could be deserialized from the database.
        :raises IncompatibleError: In case the database is not compatible with this class.
        """
        self.__database = database
        self.__db_id = db_id
        self.__genotype_type = genotype_type
        self.__genotype_serializer = genotype_serializer
        self.__fitness_type = fitness_type
        self.__fitness_serializer = fitness_serializer
        self.__offspring_size = offspring_size

        try:
            eo_row = (
                (
                    await session.execute(
                        select(DbEAOptimizer).filter(
                            DbEAOptimizer.db_id == db_id.fullname
                        )
                    )
                )
                .scalars()
                .one()
            )
        except MultipleResultsFound as err:
            raise IncompatibleError() from err
        except (NoResultFound, OperationalError):
            return False

        self.__ea_optimizer_id = eo_row.id
        self.__offspring_size = eo_row.offspring_size

        state_row = (
            (
                await session.execute(
                    select(DbEAOptimizerState)
                    .filter(
                        DbEAOptimizerState.ea_optimizer_id == self.__ea_optimizer_id
                    )
                    .order_by(DbEAOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        if state_row is None:
            raise IncompatibleError()  # not possible that there is no saved state but DbEAOptimizer row exists

        self.__generation_index = state_row.generation_index

        gen_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerGeneration)
                    .filter(
                        (
                            DbEAOptimizerGeneration.ea_optimizer_id
                            == self.__ea_optimizer_id
                        )
                        & (
                            DbEAOptimizerGeneration.generation_index
                            == self.__generation_index
                        )
                    )
                    .order_by(DbEAOptimizerGeneration.individual_index)
                )
            )
            .scalars()
            .all()
        )

        individual_ids = [row.individual_id for row in gen_rows]

        # the highest individual id in the latest generation is the highest id overall.
        self.__next_individual_id = max(individual_ids) + 1

        individual_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual).filter(
                        (
                            DbEAOptimizerIndividual.ea_optimizer_id
                            == self.__ea_optimizer_id
                        )
                        & (DbEAOptimizerIndividual.individual_id.in_(individual_ids))
                    )
                )
            )
            .scalars()
            .all()
        )
        individual_map = {i.individual_id: i for i in individual_rows}

        if not len(individual_ids) == len(individual_rows):
            raise IncompatibleError()

        genotype_ids = [individual_map[id].genotype_id for id in individual_ids]
        genotypes = await self.__genotype_serializer.from_database(
            session, genotype_ids
        )

        assert len(genotypes) == len(genotype_ids)
        self.__latest_population = [
            _Individual(g_id, g, None) for g_id, g in zip(individual_ids, genotypes)
        ]

        if self.__generation_index == 0:
            self.__latest_fitnesses = None
        else:
            final_fitness_ids = [individual_map[id].final_fitness_id for id in individual_ids]
            final_fitnesses = await self.__fitness_serializer.from_database(
                session, final_fitness_ids
            )
            assert len(final_fitnesses) == len(final_fitness_ids)
            starting_fitness_ids = [individual_map[id].starting_fitness_id for id in individual_ids]
            starting_fitnesses = await self.__fitness_serializer.from_database(
                session, starting_fitness_ids
            )
            assert len(starting_fitnesses) == len(starting_fitness_ids)
            fitnesses = [[] for _ in range(2)]
            fitnesses[0] = starting_fitnesses
            fitnesses[1] = final_fitnesses
            self.__latest_fitnesses = fitnesses

        return True

    async def run(self) -> None:
        """Run the optimizer."""
        # evaluate initial population if required
        if self.__latest_fitnesses is None:
            logging.info("Evaluating initial population of morphologies")
            initial_fitnesses, new_genotypes, = await self.__safe_evaluate_generation(
                [i.genotype for i in self.__latest_population],
                self.__database,
                self.__db_id,
            )
            self.__latest_fitnesses = initial_fitnesses
            for i, ind in enumerate(self.__latest_population):
                ind.genotype = new_genotypes[i]
            initial_population = self.__latest_population
            async with AsyncSession(self.__database) as session:
                async with session.begin():
                    await self.__save_generation_using_session(
                        session, None, None, self.__latest_population, None
                    )
            #initial_fitnesses = self.__latest_fitnesses
            logging.info("Finished evaluating initial population of morphologies")
        else:
            initial_population = None
            initial_fitnesses = None

        while self.__safe_must_do_next_gen():

            self.__generation_index += 1

            logging.info(f"Starting generation of morphologies num: {str(self.__generation_index)}")

            # let user select parents
            parent_selections = self.__safe_select_parents(
                [i.genotype for i in self.__latest_population],
                self.__latest_fitnesses[1],
                self.__offspring_size,
            )

            # let user create offspring
            offspring = [
                self.__safe_mutate(
                    self.__safe_crossover(
                        [self.__latest_population[i].genotype for i in s]
                    )
                )
                for s in parent_selections
            ]

            # let user evaluate offspring
            new_fitnesses, new_genotypes = await self.__safe_evaluate_generation(
                offspring,
                self.__database,
                self.__db_id
            )

            # combine to create list of individuals
            new_individuals = [
                _Individual(
                    -1,  # placeholder until later
                    genotype,
                    [self.__latest_population[i].id for i in parent_indices],
                )
                for parent_indices, genotype in zip(parent_selections, new_genotypes)
            ]

            # let user select survivors between old and new individuals
            old_survivors = self.__safe_select_survivors(
                [i.genotype for i in self.__latest_population],
                self.__latest_fitnesses[1],
                len(self.__latest_population) - self.__offspring_size,
            )

            # set ids for new individuals
            for individual in new_individuals:
                individual.id = self.__gen_next_individual_id()

            # combine old and new and store as the new generation
            self.__latest_population = [
                self.__latest_population[i] for i in old_survivors
            ] + new_individuals

            self.__latest_fitnesses = [[self.__latest_fitnesses[0][i] for i in old_survivors] + new_fitnesses[0],
                                        [self.__latest_fitnesses[1][i] for i in old_survivors] + new_fitnesses[1]]

            # save generation and possibly fitnesses of initial population
            # and let user save their state
            async with AsyncSession(self.__database) as session:
                async with session.begin():
                    await self.__save_generation_using_session(
                        session,
                        initial_population,
                        initial_fitnesses,
                        new_individuals,
                        new_fitnesses,
                    )
                    self._on_generation_checkpoint(session)
            # in any case they should be none after saving once
            initial_population = None
            initial_fitnesses = None

            logging.info(f"Finished morphology generation {self.__generation_index}.")

        assert (
            self.__generation_index > 0
        ), "Must create at least one generation beyond initial population. This behaviour is not supported."  # would break database structure

    @property
    def generation_index(self) -> Optional[int]:
        """
        Get the current generation.

        The initial generation is numbered 0.

        :returns: The current generation.
        """
        return self.__generation_index

    def __gen_next_individual_id(self) -> int:
        next_id = self.__next_individual_id
        self.__next_individual_id += 1
        return next_id

    async def __safe_evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        db_id: DbId
    ) -> Tuple[List[Fitness], List[Genotype], List[Fitness]]:
        fitnesses, new_genotypes = await self._evaluate_generation(
            genotypes=genotypes,
            database=database,
            db_id=db_id
        )
        starting_fitnesses = fitnesses[0]
        final_fitnesses = fitnesses[1]
        assert type(final_fitnesses) == list
        assert len(final_fitnesses) == len(genotypes)
        assert all(type(e) == self.__fitness_type for e in final_fitnesses)
        assert type(new_genotypes) == list
        assert len(new_genotypes) == len(genotypes)
        assert all(type(e) == self.__genotype_type for e in new_genotypes)
        assert type(starting_fitnesses) == list
        assert len(starting_fitnesses) == len(genotypes)
        assert all(type(e) == self.__fitness_type for e in starting_fitnesses)
        return (starting_fitnesses, final_fitnesses), new_genotypes

    def __safe_select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[Fitness],
        num_parent_groups: int,
    ) -> List[List[int]]:
        parent_selections = self._select_parents(
            population, fitnesses, num_parent_groups
        )
        assert type(parent_selections) == list
        assert len(parent_selections) == num_parent_groups
        assert all(type(s) == list for s in parent_selections)
        assert all(
            [
                all(type(p) == int and p >= 0 and p < len(population) for p in s)
                for s in parent_selections
            ]
        )
        return parent_selections

    def __safe_crossover(self, parents: List[Genotype]) -> Genotype:
        genotype = self._crossover(parents)
        assert type(genotype) == self.__genotype_type
        return genotype

    def __safe_mutate(self, genotype: Genotype) -> Genotype:
        genotype = self._mutate(genotype)
        assert type(genotype) == self.__genotype_type
        return genotype

    def __safe_select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[Fitness],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        old_survivors = self._select_survivors(
            old_individuals,
            old_fitnesses,
            num_survivors,
        )
        assert type(old_survivors) == list
        assert all(type(s) == int for s in old_survivors)
        return old_survivors

    def __safe_must_do_next_gen(self) -> bool:
        must_do = self._must_do_next_gen()
        assert type(must_do) == bool
        return must_do

    async def __save_generation_using_session(
        self,
        session: AsyncSession,
        initial_population: Optional[List[_Individual[Genotype]]],
        initial_fitnesses: Optional[Tuple[List[Fitness]]],
        new_individuals: List[_Individual[Genotype]],
        new_fitnesses: Optional[Tuple[List[Fitness]]],
    ) -> None:
        # TODO this function can probably be simplified as well as optimized.
        # but it works so I'll leave it for now.

        # update fitnesses of initial population if provided
        if initial_fitnesses is not None:
            assert initial_population is not None

            starting_fitness_ids = await self.__fitness_serializer.to_database(
                session, initial_fitnesses[0]
            )
            final_fitness_ids = await self.__fitness_serializer.to_database(
                session, initial_fitnesses[1]
            )
            assert len(starting_fitness_ids) == len(initial_fitnesses[0])
            assert len(final_fitness_ids) == len(initial_fitnesses[1])

            rows = (
                (
                    await session.execute(
                        select(DbEAOptimizerIndividual)
                        .filter(
                            (
                                DbEAOptimizerIndividual.ea_optimizer_id
                                == self.__ea_optimizer_id
                            )
                            & (
                                DbEAOptimizerIndividual.individual_id.in_(
                                    [i.id for i in initial_population]
                                )
                            )
                        )
                        .order_by(DbEAOptimizerIndividual.individual_id)
                    )
                )
                .scalars()
                .all()
            )
            if len(rows) != len(initial_population):
                raise IncompatibleError()

            for i, row in enumerate(rows):
                row.starting_fitness_id = starting_fitness_ids[i]
                row.final_fitness_id = final_fitness_ids[i]

        # save current optimizer state
        session.add(
            DbEAOptimizerState(
                ea_optimizer_id=self.__ea_optimizer_id,
                generation_index=self.__generation_index,
            )
        )

        # save new individuals
        genotype_ids = await self.__genotype_serializer.to_database(
            session, [i.genotype for i in new_individuals]
        )
        assert len(genotype_ids) == len(new_individuals)
        new_starting_fitness_ids: List[Optional[int]]
        new_final_fitness_ids: List[Optional[int]]
        if new_fitnesses is not None:
            new_starting_fitness_ids = [
                f
                for f in await self.__fitness_serializer.to_database(
                    session, new_fitnesses[0]
                )
            ]  # this extra comprehension is useless but it stops mypy from complaining
            assert len(new_starting_fitness_ids) == len(new_fitnesses[0])
            new_final_fitness_ids = [
                f
                for f in await self.__fitness_serializer.to_database(
                    session, new_fitnesses[1]
                )
            ]  # this extra comprehension is useless but it stops mypy from complaining
            assert len(new_final_fitness_ids) == len(new_fitnesses[1])
        else:
            new_starting_fitness_ids = [None for _ in range(len(new_individuals))]
            new_final_fitness_ids = [None for _ in range(len(new_individuals))]

        bodies = [body_develop(ind.genotype.body) for ind in new_individuals]

        # save body image
        for ind, body in zip(new_individuals, bodies):
            render = Render()
            id = ind.id
            img_path = f'database/body_images/generation_{self.generation_index}/individual_{id}.png'
            render.render_robot(body.core, img_path)

        # compute morphological measures
        measures = [MorphologicalMeasures(body) for body in bodies]

        session.add_all(
            [
                DbEAOptimizerIndividual(
                    ea_optimizer_id=self.__ea_optimizer_id,
                    individual_id=i.id,
                    genotype_id=g_id,
                    starting_fitness_id=s_id,
                    final_fitness_id=f_id,
                    absolute_size = mm.num_modules,
                    proportion = compute_proportion(mm),
                    num_bricks = mm.num_bricks,
                    rel_num_limbs = mm.limbs,
                    symmetry = mm.symmetry,
                    branching = mm.branching
                )
                for i, g_id, s_id, f_id, mm in zip(new_individuals, genotype_ids, new_starting_fitness_ids, new_final_fitness_ids, measures)
            ]
        )

        # save parents of new individuals
        parents: List[DbEAOptimizerParent] = []
        for individual in new_individuals:
            assert (
                individual.parent_ids is not None
            )  # Cannot be None. They are only None after recovery and then they are already saved.
            for p_id in individual.parent_ids:
                parents.append(
                    DbEAOptimizerParent(
                        ea_optimizer_id=self.__ea_optimizer_id,
                        child_individual_id=individual.id,
                        parent_individual_id=p_id,
                    )
                )
        session.add_all(parents)

        # save current generation
        session.add_all(
            [
                DbEAOptimizerGeneration(
                    ea_optimizer_id=self.__ea_optimizer_id,
                    generation_index=self.__generation_index,
                    individual_index=index,
                    individual_id=individual.id,
                )
                for index, individual in enumerate(self.__latest_population)
            ]
        )


def compute_proportion(measure: MorphologicalMeasures):
    depth = measure.bounding_box_depth
    width = measure.bounding_box_width
    height = measure.bounding_box_height

    min_dim = min(depth, width, height)
    max_dim = max(depth, width, height)

    return min_dim / max_dim

@dataclass
class _Individual(Generic[Genotype]):
    id: int
    genotype: Genotype
    # Empty list of parents means this is from the initial population
    # None means we did not bother loading the parents during recovery because they are not needed.
    parent_ids: Optional[List[int]]


class DbEAOptimizerIndividual(DbBase):
    """
    An individual with a fitness which may or may not be assigned.

    Can be part of multiple generations.
    """

    __tablename__ = "ea_morph_optimizer_individual"

    ea_optimizer_id = Column(Integer, nullable=False, primary_key=True)
    individual_id = Column(Integer, nullable=False, primary_key=True)
    genotype_id = Column(Integer, nullable=False)
    starting_fitness_id = Column(Integer, nullable=True)
    final_fitness_id = Column(Integer, nullable=True)
    absolute_size = Column(Integer, nullable=False)
    proportion = Column(Float, nullable=False)
    num_bricks = Column(Integer, nullable=False)
    rel_num_limbs = Column(Float, nullable=False)
    symmetry = Column(Float, nullable=False)
    branching = Column(Float, nullable=False)
    