from __future__ import annotations

import sys
from random import Random
from typing import List

import multineat

from dataclasses import dataclass

from revolve2.core.modular_robot import ModularRobot

from revolve2.genotypes.cppnwin import Genotype as CppnwinGenotype
from revolve2.core.database import IncompatibleError, Serializer
from revolve2.genotypes.cppnwin import GenotypeSerializer as CppnwinGenotypeSerializer
from revolve2.genotypes.cppnwin import crossover_v1 as cppnwin_crossover, mutate_v1 as cppnwin_mutate
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
    random_v1 as brain_random,
)
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
    develop_v1 as brain_develop,
)

from jlo.direct_tree.direct_tree_genotype import DirectTreeGenotype
from jlo.direct_tree.direct_tree_config import DirectTreeGenotypeConfig
from jlo.direct_tree.direct_tree_genotype import GenotypeSerializer as DirectTreeGenotypeSerializer
from jlo.direct_tree.direct_tree_genotype import develop as body_develop

from revolve2.core.modular_robot import Body

import sqlalchemy
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

from jlo.direct_tree.direct_tree_utils import duplicate_subtree
from jlo.direct_tree import robot_zoo


def _make_direct_tree_config() -> DirectTreeGenotypeConfig:

    morph_single_mutation_prob = 0.2
    morph_no_single_mutation_prob = 1 - morph_single_mutation_prob  # 0.8
    morph_no_all_mutation_prob = morph_no_single_mutation_prob ** 4  # 0.4096
    morph_at_least_one_mutation_prob = 1 - morph_no_all_mutation_prob  # 0.5904

    brain_single_mutation_prob = 0.5

    tree_genotype_conf: DirectTreeGenotypeConfig = DirectTreeGenotypeConfig(
        max_parts=50,
        min_parts=10,
        max_oscillation=5,
        init_n_parts_mu=10,
        init_n_parts_sigma=4,
        init_prob_no_child=0.1,
        init_prob_child_block=0.4,
        init_prob_child_active_joint=0.5,
        mutation_p_duplicate_subtree=morph_single_mutation_prob,
        mutation_p_delete_subtree=morph_single_mutation_prob,
        mutation_p_generate_subtree=morph_single_mutation_prob,
        mutation_p_swap_subtree=morph_single_mutation_prob,
        mutation_p_mutate_oscillators=brain_single_mutation_prob,
        mutation_p_mutate_oscillator=0.5,
        mutate_oscillator_amplitude_sigma=0.3,
        mutate_oscillator_period_sigma=0.3,
        mutate_oscillator_phase_sigma=0.3,
    )

    return tree_genotype_conf

def _make_multineat_params() -> multineat.Parameters:
    multineat_params = multineat.Parameters()

    multineat_params.MutateRemLinkProb = 0.02
    multineat_params.RecurrentProb = 0.0
    multineat_params.OverallMutationRate = 0.15
    multineat_params.MutateAddLinkProb = 0.08
    multineat_params.MutateAddNeuronProb = 0.01
    multineat_params.MutateWeightsProb = 0.90
    multineat_params.MaxWeight = 8.0
    multineat_params.WeightMutationMaxPower = 0.2
    multineat_params.WeightReplacementMaxPower = 1.0
    multineat_params.MutateActivationAProb = 0.0
    multineat_params.ActivationAMutationMaxPower = 0.5
    multineat_params.MinActivationA = 0.05
    multineat_params.MaxActivationA = 6.0

    multineat_params.MutateNeuronActivationTypeProb = 0.03

    multineat_params.MutateOutputActivationFunction = False

    multineat_params.ActivationFunction_SignedSigmoid_Prob = 0.0
    multineat_params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    multineat_params.ActivationFunction_Tanh_Prob = 1.0
    multineat_params.ActivationFunction_TanhCubic_Prob = 0.0
    multineat_params.ActivationFunction_SignedStep_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedStep_Prob = 0.0
    multineat_params.ActivationFunction_SignedGauss_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedGauss_Prob = 0.0
    multineat_params.ActivationFunction_Abs_Prob = 0.0
    multineat_params.ActivationFunction_SignedSine_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedSine_Prob = 0.0
    multineat_params.ActivationFunction_Linear_Prob = 1.0

    multineat_params.MutateNeuronTraitsProb = 0.0
    multineat_params.MutateLinkTraitsProb = 0.0

    multineat_params.AllowLoops = False

    return multineat_params


@dataclass
class Genotype:
    body: DirectTreeGenotype
    brain: CppnwinGenotype

class GenotypeSerializer(Serializer[Genotype]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await CppnwinGenotypeSerializer.create_tables(session)
        await DirectTreeGenotypeSerializer.create_tables(session)

    @classmethod
    def identifying_table(cls) -> str:
        return DbGenotype.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[Genotype]
    ) -> List[int]:
        brain_ids = await CppnwinGenotypeSerializer.to_database(
            session, [o.brain for o in objects]
        )
        body_ids = await DirectTreeGenotypeSerializer.to_database(
            session, [o.body for o in objects]
        )

        dbgenotypes = [
            DbGenotype(body_id=body_id, brain_id=brain_id)
            for body_id, brain_id in zip(body_ids, brain_ids)
        ]

        session.add_all(dbgenotypes)
        await session.flush()
        ids = [
            dbfitness.id for dbfitness in dbgenotypes if dbfitness.id is not None
        ]  # cannot be none because not nullable. check if only there to silence mypy.
        assert len(ids) == len(objects)  # but check just to be sure
        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[Genotype]:
        rows = (
            (await session.execute(select(DbGenotype).filter(DbGenotype.id.in_(ids))))
            .scalars()
            .all()
        )

        if len(rows) != len(ids):
            raise IncompatibleError()

        id_map = {t.id: t for t in rows}
        body_ids = [id_map[id].body_id for id in ids]
        brain_ids = [id_map[id].brain_id for id in ids]

        body_genotypes = await DirectTreeGenotypeSerializer.from_database(
            session, body_ids
        )
        brain_genotypes = await CppnwinGenotypeSerializer.from_database(
            session, brain_ids
        )

        genotypes = [
            Genotype(body, brain)
            for body, brain in zip(body_genotypes, brain_genotypes)
        ]

        return genotypes


_MULTINEAT_PARAMS = _make_multineat_params()
_DIRECT_TREE_CONFIG = _make_direct_tree_config()


def random(
    innov_db_brain: multineat.InnovationDatabase,
    rng: Random,
    num_initial_mutations: int,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    body = robot_zoo.make_ant_body()

    brain = brain_random(
        innov_db_brain,
        multineat_rng,
        _MULTINEAT_PARAMS,
        multineat.ActivationFunction.SIGNED_SINE,
        num_initial_mutations,
    )

    return Genotype(body, brain)

def mutate(
    genotype: Genotype,
    innov_db_brain: multineat.InnovationDatabase,
    rng: Random,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    new_body = Body()
    new_body.core = duplicate_subtree(genotype.body.genotype.core)

    return Genotype(
        DirectTreeGenotype(new_body),
        cppnwin_mutate(genotype.brain, _MULTINEAT_PARAMS, innov_db_brain, multineat_rng),

    )

def crossover(
    parent1: Genotype,
    parent2: Genotype,
    rng: Random,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    new_body = Body()
    new_body.core = duplicate_subtree(parent1.body.genotype.core)

    return Genotype(
            DirectTreeGenotype(new_body),
        cppnwin_crossover(
            parent1.brain,
            parent2.brain,
            _MULTINEAT_PARAMS,
            multineat_rng,
            False,
            False,
        ),
    )

def _multineat_rng_from_random(rng: Random) -> multineat.RNG:
    multineat_rng = multineat.RNG()
    multineat_rng.Seed(rng.randint(0, sys.maxsize))
    return multineat_rng

def develop(genotype: Genotype) -> ModularRobot:
    body = body_develop(genotype.body)
    brain = brain_develop(genotype.brain, body)
    return ModularRobot(body, brain)

DbBase = declarative_base()


class DbGenotype(DbBase):
    __tablename__ = "genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    body_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    brain_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)