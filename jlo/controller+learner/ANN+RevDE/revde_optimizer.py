
import logging
import pickle
from abc import ABC, abstractmethod
from mimetypes import init
from random import Random
from typing import List, Optional
import torch

import numpy as np
import numpy.typing as npt
import sqlalchemy
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import (DbNdarray1xn, FloatSerializer,
                                                Ndarray1xnSerializer)
from revolve2.core.optimization import Process, ProcessIdGen
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound


class RevDEOptimizer(ABC, Process):
    """
    RevDE Optimizer 
    
    based on:
    https://github.com/jmtomczak/popi4sb/blob/master/algorithms/population_optimization_algorithms.py#L49
    """

    @abstractmethod
    async def _evaluate_population(
        self,
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate all individuals in the population, returning their fitnesses.

        :param database: Database that can be used to store anything you want to save from the evaluation.
        :param process_id: Unique identifier in the completely program specifically made for this function call.
        :param process_id_gen: Can be used to create more unique identifiers.
        :param population: MxN array with M the population size and N the size of an individual.
        :returns: M long vector with M the population size, representing the fitness of each individual in `population`.
        """

    @abstractmethod
    def _must_do_next_gen(self) -> bool:
        """
        Decide if the optimizer must do another generation.

        :returns: True if it must.
        """

    __database: AsyncEngine
    __process_id: int
    __process_id_gen: ProcessIdGen

    __rng: Random

    __population_size: int
    __latest_population: List[CpgNetworkStructure]
    __latest_fitnesses: npt.NDArray[np.float_]

    __gen_num: int

    __scaling: float
    __cross_prob: float
    __R: npt.NDArray[np.float_]

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        population_size: int,
        initial_population: List[np.float_],
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
        :param rng: Random number generator used in the complete optimization process.
        :param population_size: Size of the population. OpenAI ES parameter.
        :param sigma: Standard deviation. OpenAI ES parameter.
        :param learning_rate: Gain factor for the directional vector. OpenAI ES parameter.
        :param initial_mean: Nx1 array. Initial guess. OpenAI ES Parameter.
        """
        self.__database = database
        self.__process_id = process_id
        self.__process_id_gen = process_id_gen

        self.__rng = rng

        self.__population_size = population_size
        self.__latest_population = initial_population

        self.__gen_num = 0

        self.__scaling = scaling
        self.__cross_prob = cross_prob

        R = np.asarray([[1, self.__scaling, -self.__scaling],
                        [-self.__scaling, 1. - self.__scaling ** 2, self.__scaling + self.__scaling ** 2],
                        [self.__scaling + self.__scaling ** 2, -self.__scaling + self.__scaling ** 2 + self.__scaling ** 3, 1. - 2. * self.__scaling ** 2 - self.__scaling ** 3]])


        self.__R = np.expand_dims(R, 0)

        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await Ndarray1xnSerializer.create_tables(session)
        await FloatSerializer.create_tables(session)

        dbopt = DbRevDEOptimizer(
            process_id=self.__process_id,
            population_size=self.__population_size,
            initial_rng=pickle.dumps(self.__rng.getstate()),
            scaling=scaling,
            cross_prob=cross_prob,
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
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param process_id: Unique identifier in the completely program specifically made for this optimizer.
        :param process_id_gen: Can be used to create more unique identifiers.
        :param rng: Random number generator used in the complete optimization process. Its state will be overwritten with the serialized state from the database.
        :returns: True if the complete object could be deserialized from the database.
        :raises IncompatibleError: In case the database is not compatible with this class.
        """
        self.__database = database
        self.__process_id = process_id
        self.__process_id_gen = process_id_gen

        try:
            opt_row = (
                (
                    await session.execute(
                        select(DbRevDEOptimizer).filter(
                            DbRevDEOptimizer.process_id == self.__process_id
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

        self.__population_size = opt_row.population_size
        self.__scaling = opt_row.scaling
        self.__cross_prob = opt_row.cross_prob

        R = np.asarray([[1, self.__scaling, -self.__scaling],
                        [-self.__scaling, 1. - self.__scaling ** 2, self.__scaling + self.__scaling ** 2],
                        [self.__scaling + self.__scaling ** 2, -self.__scaling + self.__scaling ** 2 + self.__scaling ** 3, 1. - 2. * self.__scaling ** 2 - self.__scaling ** 3]])

        self.__R = np.expand_dims(R, 0)

        db_state = (
            (
                await session.execute(
                    select(DbRevDEOptimizerState)
                    .filter(DbRevDEOptimizerState.process_id == self.__process_id)
                    .order_by(DbRevDEOptimizerState.gen_num.desc())
                )
            )
            .scalars()
            .first()
        )

        self.__rng = rng  # set state from database below

        self.__gen_num = db_state.gen_num
        self.__rng.setstate(pickle.loads(db_state.rng))

        gen_rows = (
            (
                await session.execute(
                    select(DbRevDEOptimizerGeneration)
                    .filter(
                        (
                            DbRevDEOptimizerGeneration.process_id
                            == self.__process_id
                        )
                        & (
                            DbRevDEOptimizerGeneration.gen_num
                            == self.__gen_num
                        )
                    )
                    .order_by(DbRevDEOptimizerGeneration.gen_index)
                )
            )
            .scalars()
            .all()
        )

        generation_ids = [row.individual_id for row in gen_rows]

        individual_rows = (
            (
                await session.execute(
                    select(DbRevDEOptimizerBestIndividual).filter(
                        (
                            DbRevDEOptimizerBestIndividual.process_id
                            == self.__process_id
                        )
                        & (DbRevDEOptimizerBestIndividual.individual.in_(generation_ids))
                    )
                )
            )
            .scalars()
            .all()
        )
        individual_ids = [row.individual for row in individual_rows]

        individuals = [(await Ndarray1xnSerializer.from_database(session, [id]))[0] for id in individual_ids]
        self.__latest_population = np.array(individuals)
        fitnesses = [i.fitness for i in individual_rows]
        self.__latest_fitnesses = np.array(fitnesses)

        if not len(individual_ids) == len(individual_rows):
            raise IncompatibleError()

        self.__gen_num += 1

        return True

    async def run(self) -> None:
        """Run the optimizer."""
        if self.generation_number == 0:
            fitnesses = await self._evaluate_population(
                self.__database,
                self.__process_id_gen.gen(),
                self.__process_id_gen,
                self.__latest_population,
            )
            assert fitnesses.shape == (len(self.__latest_population),)
            self.__latest_fitnesses = fitnesses
            
            async with AsyncSession(self.__database) as session:
                async with session.begin():

                    # save current optimizer state
                    session.add(
                        DbRevDEOptimizerState(
                            process_id=self.__process_id,
                            gen_num=self.__gen_num,
                            rng=pickle.dumps(self.__rng.getstate()),
                        )
                    )

                    # save new individuals
                    db_individual_ids = []
                    for ind in self.__latest_population:
                        id = await Ndarray1xnSerializer.to_database(
                            session, [ind]
                        )
                        db_individual_ids += id
                    assert len(db_individual_ids) == len(self.__latest_population)

                    session.add_all(
                        [
                            DbRevDEOptimizerIndividual(
                                process_id=self.__process_id,
                                gen_num=self.__gen_num,
                                gen_index=index,
                                individual=id,
                                fitness=fitness,
                            )
                        for index, id, fitness in zip(
                            range(len(self.__latest_population)), db_individual_ids, self.__latest_fitnesses
                        )
                        ]
                    )

                    # save current generation for revovery
                    db_generation_ids = []
                    for ind in self.__latest_population:
                        id = await Ndarray1xnSerializer.to_database(
                            session, [ind]
                        )
                        db_generation_ids += id
                    assert len(db_generation_ids) == len(self.__latest_population)

                    session.add_all(
                        [
                            DbRevDEOptimizerBestIndividual(
                                process_id=self.__process_id,
                                gen_num=self.__gen_num,
                                gen_index=index,
                                individual=id,
                                fitness=fitness,
                            )
                        for index, id, fitness in zip(
                            range(len(self.__latest_population)), db_generation_ids, self.__latest_fitnesses
                        )
                        ]
                    )

                    session.add_all(
                        [
                            DbRevDEOptimizerGeneration(
                                process_id=self.__process_id,
                                gen_num=self.__gen_num,
                                gen_index=index,
                                individual_id=individual_id,
                            )
                            for index, individual_id in enumerate(db_generation_ids)
                        ]
                    )
                    logging.info(f"Finished generation {self.__gen_num}")
                    self.__gen_num += 1

        while self.__safe_must_do_next_gen():
            rng = np.random.Generator(
                np.random.PCG64(self.__rng.randint(0, 2**63))
            )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.

            candidates = self.proposal(self.__latest_population)
            candidates = torch.from_numpy(candidates).float()

            fitnesses = await self._evaluate_population(
                self.__database,
                self.__process_id_gen.gen(),
                self.__process_id_gen,
                candidates,
            )

            full_candidates = np.concatenate((self.__latest_population, candidates), 0)
            full_fitnesses = np.concatenate((self.__latest_fitnesses, fitnesses), 0)
            indexes = np.argsort(full_fitnesses.squeeze())[-self.__latest_population.shape[0]:]
            self.__latest_population = full_candidates[indexes,:]
            self.__latest_fitnesses = full_fitnesses[indexes]

            async with AsyncSession(self.__database) as session:
                async with session.begin():

                    # save current optimizer state
                    session.add(
                        DbRevDEOptimizerState(
                            process_id=self.__process_id,
                            gen_num=self.__gen_num,
                            rng=pickle.dumps(self.__rng.getstate()),
                        )
                    )

                    # save new individuals
                    db_individual_ids = []
                    for ind in candidates:
                        id = await Ndarray1xnSerializer.to_database(
                            session, [ind]
                        )
                        db_individual_ids += id
                    assert len(db_individual_ids) == len(candidates)

                    session.add_all(
                        [
                            DbRevDEOptimizerIndividual(
                                process_id=self.__process_id,
                                gen_num=self.__gen_num,
                                gen_index=index,
                                individual=id,
                                fitness=fitness,
                            )
                        for index, id, fitness in zip(
                            range(len(candidates)), db_individual_ids, full_fitnesses
                        )
                        ]
                    )

                    # save current generation for revovery
                    db_generation_ids = []
                    for ind in self.__latest_population:
                        id = await Ndarray1xnSerializer.to_database(
                            session, [ind]
                        )
                        db_generation_ids += id
                    assert len(db_generation_ids) == len(self.__latest_population)

                    session.add_all(
                        [
                            DbRevDEOptimizerBestIndividual(
                                process_id=self.__process_id,
                                gen_num=self.__gen_num,
                                gen_index=index,
                                individual=id,
                                fitness=fitness,
                            )
                        for index, id, fitness in zip(
                            range(len(candidates)), db_generation_ids, self.__latest_fitnesses
                        )
                        ]
                    )

                    session.add_all(
                        [
                            DbRevDEOptimizerGeneration(
                                process_id=self.__process_id,
                                gen_num=self.__gen_num,
                                gen_index=index,
                                individual_id=individual_id,
                            )
                            for index, individual_id in enumerate(db_generation_ids)
                        ]
                    )
                    logging.info(f"Finished generation {self.__gen_num}")
                    self.__gen_num += 1

    def proposal(self, theta):
        theta_0 = np.expand_dims(theta, 1) # B x 1 x D

        indices_1 = np.random.permutation(theta.shape[0])
        indices_2 = np.random.permutation(theta.shape[0])
        theta_1 = np.expand_dims(theta[indices_1], 1)
        theta_2 = np.expand_dims(theta[indices_2], 1)

        tht = np.concatenate((theta_0, theta_1, theta_2), 1) # B x 3 x D

        y = np.matmul(self.__R, tht)

        theta_new = np.concatenate((y[:,0], y[:,1], y[:,2]), 0)

        p_1 = np.random.binomial(1, self.__cross_prob, theta_new.shape)
        return p_1 * theta_new + (1. - p_1) * np.concatenate((tht[:,0], tht[:,1], tht[:,2]), 0)

    @property
    def generation_number(self) -> Optional[int]:
        """
        Get the current generation.

        The initial generation is numbered 0.

        :returns: The current generation.
        """
        return self.__gen_num

    def __safe_must_do_next_gen(self) -> bool:
        must_do = self._must_do_next_gen()
        assert type(must_do) == bool
        return must_do



DbBase = declarative_base()

class DbRevDEOptimizer(DbBase):
    """Model for the optimizer itself, containing static parameters."""

    __tablename__ = "revde_optimizer"

    process_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        primary_key=True,
    )
    population_size = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    initial_rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    scaling = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    cross_prob = sqlalchemy.Column(sqlalchemy.Float, nullable=False)

class DbRevDEOptimizerState(DbBase):
    """State of the optimizer."""

    __tablename__ = "revde_optimizer_state"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)

class DbRevDEOptimizerIndividual(DbBase):
    """An individual with a fitness which may or may not be assigned."""

    __tablename__ = "revde_optimizer_individual"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)

class DbRevDEOptimizerBestIndividual(DbBase):
    """An individual with a fitness which may or may not be assigned."""

    __tablename__ = "revde_optimizer_best_individual"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)

class DbRevDEOptimizerGeneration(DbBase):
    """A single generation."""

    __tablename__ = "ea_optimizer_generation"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)