import logging
import pickle
from abc import ABC, abstractmethod
from random import Random
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import sqlalchemy
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import DbNdarray1xn, Ndarray1xnSerializer
from revolve2.core.optimization import Process, ProcessIdGen
from dataclasses import dataclass

@dataclass #no need to write function
class Individual:
    id: int
    params: npt.NDArray[np.float_] #vector

class DEOptimizer(ABC, Process):

    @abstractmethod
    async def _evaluate_population(
        self,
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
        population: List[npt.NDArray[np.float_]],
    ) -> List[float]:
        """
        Evaluate all individuals in the population, returning their fitnesses.

        :population: List of individuals(parameters), which are vectors.
        :return: Vector of fitnesses
        """

    @abstractmethod
    def _must_do_next_gen(self) -> bool:
        """
        Decide if the optimizer must do another generation.
        :return: True if it must.
        """

    __database: AsyncEngine
    __process_id: int
    __process_id_gen: ProcessIdGen

    __rng: Random

    __gen_num: int
    __population: List[Individual]
    __mutation_factor: float
    __crossover_rate: float
    __fitnesses: Optional[List[float]]

    __next_individual_id: int

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        initial_pop: npt.NDArray[np.float_], # MxN matrix, with M pop sie and N number of individual params,
        mutation_factor: float,
        crossover_rate: float
    ) -> None:
        self.__next_individual_id = 0

        self.__database = database
        self.__process_id = process_id
        self.__process_id_gen = process_id_gen

        self.__rng = rng

        self.__gen_num = 0
        self.__population = [Individual(self.gen_next_individual_id(), params) for params in initial_pop]
        self.__fitnesses = None
        self.__mutation_factor= mutation_factor
        self.__crossover_rate= crossover_rate

        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await Ndarray1xnSerializer.create_tables(session)

        for index, individual in enumerate(initial_pop):
            ndarray1xn_id = await Ndarray1xnSerializer.to_database(session, [individual.params])
            individual = DbDEOptimizerIndividual(
                process_id=self.__process_id,
                id=individual.id,
                ndarray1xn_id=ndarray1xn_id,
                fitness=None)
            session.add(individual)

            gen_entry = DbDEOptimizerGeneration(
                process_id=self.__process_id,
                gen_num=0,
                index=index,
                individual_id=individual.id,
            )
            session.add(gen_entry)

        dbopt = DbDEOptimizer(
            process_id=self.__process_id,
            initial_rng=pickle.dumps(self.__rng.getstate()),
            mutation_factor= self.__mutation_factor,
            crossover_rate= self.__crossover_rate
        )
        session.add(dbopt)

    async def ainit_from_database(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
    ) -> bool:
        # TODO init from database this code is not from this class actually

        self.__database = database
        self.__process_id = process_id
        self.__process_id_gen = process_id_gen

        try:
            opt_row = (
                (
                    await session.execute(
                        select(DbDEOptimizer).filter(
                            DbDEOptimizer.process_id == self.__process_id
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

        db_state = (
            (
                await session.execute(
                    select(DbDEOptimizerState)
                    .filter(DbDEOptimizerState.process_id == self.__process_id)
                    .order_by(DbDEOptimizerState.gen_num.desc())
                )
            )
            .scalars()
            .first()
        )

        self.__rng = rng  # set state from database below

        if db_state is None:
            self.__gen_num = 0
            self.__rng.setstate(pickle.loads(opt_row.initial_rng))
        else:
            self.__gen_num = db_state.gen_num
            self.__rng.setstate(pickle.loads(db_state.rng))

        return True

    def gen_next_individual_id(self) -> int:
        next_id = self.__next_individual_id
        self.__next_individual_id += 1
        return next_id

    async def run(self) -> None:
        '''
        randomly pick 3: x_1, x_2, x_3, generate new: y, y_2
        𝑦 = 𝑥_1 + 𝐹 (𝑥_2 − 𝑥_3 ) , F is mutation rate 0.5
        y_1 = m ⊙ 𝑦 + (1 − m) ⊙ 𝑥_1, CR is crossover rate 0.5
        evaluate all and update pop
        '''
        if self.__fitnesses is None:
            self.__fitnesses = await self._evaluate_population(
                self.__database,
                self.__process_id_gen.gen(),
                self.__process_id_gen,
                [individual.params for individual in self.__population],
            )

            # TODO update fitnesses for initial population
            rows = (
                (
                    await session.execute(
                        select(DbDEOptimizerIndividual)
                            .filter(
                            (
                                    DbDEOptimizerIndividual.process_id
                                    == self.__process_id
                            )
                            & (
                                DbDEOptimizerIndividual.id.in_(
                                    [i.id for i in initial_population]
                                )
                            )
                        )
                            .order_by(DbDEOptimizerIndividual.id)
                    )
                )
                    .scalars()
                    .all()
            )
            if len(rows) != len(initial_population):
                raise IncompatibleError()

            for i, row in enumerate(rows):
                row.fitness_id = fitness_ids[i]

        while self.__safe_must_do_next_gen():
            offspring: List[Individual] = []

            for x1_index in range(len(self.__population)):
                if self.__rng.random() <= self.__crossover_rate:
                    x2_index = x1_index
                    while x2_index == x1_index:
                        x2_index = self.__rng.randint(0, len(self.__population))
                    x3_index = x2_index
                    while x3_index == x1_index or x3_index == x2_index:
                        r1 = self.__rng.randint(0, len(self.__population))

                    x1 = self.__population[x1_index].params
                    x2 = self.__population[x2_index].params
                    x3 = self.__population[x3_index].params

                    y = x1 + self.__mutation_factor * (x2 - x3)

                    offspring.append(Individual(self.gen_next_individual_id(), y))

            fitnesses = await self._evaluate_population(
                self.__database,
                self.__process_id_gen.gen(),
                self.__process_id_gen,
                [individual.params for individual in offspring],
            )

            # TODO save new individuals
            for index, individual, fitness in enumerate(zip(offspring, fitnesses)):
                index = await Ndarray1xnSerializer.to_database(session, [individual])
                individual = DbDEOptimizerIndividual(
                    process_id=self.__process_id,
                    gen_num=0,
                    ndarray1xn_id=index,
                    value_index=id,
                    fitness=fitness)  # creat header
                session.add(individual)  # add rows

            # TODO selection and save generation
            assert fitnesses.shape == (len(population),)
            fitnesses_gaussian = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
            self.__mean = self.__mean + self.__learning_rate / (
                self.__population_size * self.__sigma
            ) * np.dot(pertubations.T, fitnesses_gaussian)

            self.__gen_num += 1

            async with AsyncSession(self.__database) as session:
                async with session.begin():
                    db_mean_id = (
                        await Ndarray1xnSerializer.to_database(session, [self.__mean])
                    )[0]

                    dbopt = DbDEOptimizerState(
                        process_id=self.__process_id,
                        gen_num=self.__gen_num,
                        mean=db_mean_id,
                        rng=pickle.dumps(self.__rng.getstate()),
                    )

                    session.add(dbopt)

                    db_ndarray1xn_ids = await Ndarray1xnSerializer.to_database(
                        session, [i for i in population]
                    )

                    dbgens = [
                        DbDEOptimizerIndividual(
                            process_id=self.__process_id,
                            gen_num=self.__gen_num,
                            gen_index=index,
                            individual=id,
                            fitness=fitness,
                        )
                        for index, id, fitness in zip(
                            range(len(population)), db_ndarray1xn_ids, fitnesses
                        )
                    ]

                    session.add_all(dbgens)

                    logging.info(f"Finished generation {self.__gen_num}")

    @property
    def generation_number(self) -> Optional[int]:
        """
        Get the current generation.
        The initial generation is numbered 0.
        """

        return self.__gen_num

    def __safe_must_do_next_gen(self) -> bool:
        must_do = self._must_do_next_gen()
        assert type(must_do) == bool
        return must_do


DbBase = declarative_base()


class DbDEOptimizer(DbBase):
    __tablename__ = "de_optimizer"

    process_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        primary_key=True,
    )
    initial_rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    crossover_rate = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    mutation_factor = sqlalchemy.Column(sqlalchemy.Float, nullable=True)

class DbDEOptimizerIndividual(DbBase):
    __tablename__ = "de_optimizer_individual"
    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    ndarray1xn_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)

class DbDEOptimizerState(DbBase):
    __tablename__ = "de_optimizer_state"
    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    ndarray1xn_id = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)

class DbDEOptimizerGeneration(DbBase):
    __tablename__ = "de_optimizer_generation"
    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
