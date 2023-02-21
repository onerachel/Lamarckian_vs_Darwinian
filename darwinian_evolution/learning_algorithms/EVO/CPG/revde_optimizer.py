
import logging
import pickle
from abc import ABC, abstractmethod
from mimetypes import init
from random import Random
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import sqlalchemy
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import (DbNdarray1xn, FloatSerializer,
                                                Ndarray1xnSerializer)
from revolve2.core.optimization import DbId, Process
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
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate all individuals in the population, returning their fitnesses.

        :param database: Database that can be used to store anything you want to save from the evaluation.
        :param db_id: Unique identifier in the completely program specifically made for this function call.
        :param population: MxN array with M the population size and N the size of an individual.
        :returns: M long vector with M the population size, representing the fitness of each individual in `population`.
        """

    @abstractmethod
    def _must_do_next_gen(self) -> bool:
        """
        Decide if the optimizer must do another generation.

        :returns: True if it must.
        """

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
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param rng: Random number generator used in the complete optimization process.
        :param population_size: Size of the population. OpenAI ES parameter.
        :param sigma: Standard deviation. OpenAI ES parameter.
        :param learning_rate: Gain factor for the directional vector. OpenAI ES parameter.
        :param initial_mean: Nx1 array. Initial guess. OpenAI ES Parameter.
        """
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

    async def ainit_from_database(
        self,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
    ) -> bool:
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param rng: Random number generator used in the complete optimization process. Its state will be overwritten with the serialized state from the database.
        :returns: True if the complete object could be deserialized from the database.
        :raises IncompatibleError: In case the database is not compatible with this class.
        """
        return False

    async def run(self) -> Tuple[npt.NDArray[np.float_], float]:
        """Run the optimizer."""
        if self.generation_number == 0:
            fitnesses = await self._evaluate_population(
                self.__latest_population,
            )
            assert fitnesses.shape == (len(self.__latest_population),)
            self.__latest_fitnesses = fitnesses
            self.__gen_num += 1

            max_idx = np.argmax(self.__latest_fitnesses)
            best_controller = self.__latest_population[max_idx]
            max_fitness = self.__latest_fitnesses[max_idx]
            initial_fitness = self.__latest_fitnesses[0]

        while self.__safe_must_do_next_gen():
            rng = np.random.Generator(
                np.random.PCG64(self.__rng.randint(0, 2**63))
            )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.

            candidates = self.proposal(self.__latest_population)

            fitnesses = await self._evaluate_population(
                candidates,
            )

            full_candidates = np.concatenate((self.__latest_population, candidates), 0)
            full_fitnesses = np.concatenate((self.__latest_fitnesses, fitnesses), 0)
            indexes = np.argsort(full_fitnesses.squeeze())[-self.__latest_population.shape[0]:]
            self.__latest_population = full_candidates[indexes,:]
            self.__latest_fitnesses = full_fitnesses[indexes]

            max_idx = np.argmax(self.__latest_fitnesses)
            best_controller = self.__latest_population[max_idx]
            max_fitness = self.__latest_fitnesses[max_idx]

            self.__gen_num += 1

        return best_controller, max_fitness.item(), initial_fitness.item()

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

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        primary_key=True,
    )
    db_id = sqlalchemy.Column(sqlalchemy.String, nullable=False, unique=True)
    population_size = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    initial_rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    scaling = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    cross_prob = sqlalchemy.Column(sqlalchemy.Float, nullable=False)

class DbRevDEOptimizerState(DbBase):
    """State of the optimizer."""

    __tablename__ = "revde_optimizer_state"

    ea_optimizer_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)

class DbRevDEOptimizerIndividual(DbBase):
    """An individual with a fitness which may or may not be assigned."""

    __tablename__ = "revde_optimizer_individual"

    ea_optimizer_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)

class DbRevDEOptimizerBestIndividual(DbBase):
    """An individual with a fitness which may or may not be assigned."""

    __tablename__ = "revde_optimizer_best_individual"

    ea_optimizer_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)

class DbRevDEOptimizerGeneration(DbBase):
    """A single generation."""

    __tablename__ = "revde_optimizer_generation"

    ea_optimizer_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)