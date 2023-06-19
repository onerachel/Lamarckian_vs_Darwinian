"""Visualize and simulate the best robot from the optimization process."""

import math
import numpy as np

from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from _optimizer import DbEAOptimizerIndividual
from genotype import DbGenotype, GenotypeSerializer
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1 as body_develop
from typing import Optional
import argparse
from morphological_measures import MorphologicalMeasures
import pandas as pd
from revolve2.core.optimization.ea.generic_ea._database import (
    DbBase,
    DbEAOptimizer,
    DbEAOptimizerGeneration
)
from revolve2.core.database import open_database_sqlite


async def main() -> None:


    width = np.zeros(1500*10*2)
    height = np.zeros(1500*10*2)
    depth = np.zeros(1500*10*2)
    num_hinges = np.zeros(1500*10*2)

    idx = 0

    for ran_num in range(1,11):

        db = open_database_sqlite('asexual_reproduction/rotation/run' + str(ran_num))
        df = pd.read_sql(
            select(
                DbEAOptimizerGeneration,
                DbEAOptimizerIndividual,
                DbGenotype
            ).filter(
                (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
                & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
                & (
                    DbEAOptimizerGeneration.individual_id
                    == DbEAOptimizerIndividual.individual_id
                )
                & (DbGenotype.id == DbEAOptimizerIndividual.genotype_id)
            ),
            db,
        )

        for i, ind in df.iterrows():

            db = open_async_database_sqlite('asexual_reproduction/rotation/run' + str(ran_num))
            async with AsyncSession(db) as session:
                genotype = (await GenotypeSerializer.from_database(session, [ind.id]))[0]

            body = body_develop(genotype.body)
            measures = MorphologicalMeasures(body)
            width[idx] = measures.bounding_box_width
            height[idx] = measures.bounding_box_height
            depth[idx] = measures.bounding_box_depth
            num_hinges[idx] = measures.num_active_hinges
            idx += 1

    for ran_num in range(1,11):

        db = open_database_sqlite('sexual_reproduction/rotation/run' + str(ran_num))
        df = pd.read_sql(
            select(
                DbEAOptimizerGeneration,
                DbEAOptimizerIndividual,
                DbGenotype
            ).filter(
                (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
                & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
                & (
                    DbEAOptimizerGeneration.individual_id
                    == DbEAOptimizerIndividual.individual_id
                )
                & (DbGenotype.id == DbEAOptimizerIndividual.genotype_id)
            ),
            db,
        )

        for i, ind in df.iterrows():

            db = open_async_database_sqlite('sexual_reproduction/rotation/run' + str(ran_num))
            async with AsyncSession(db) as session:
                genotype = (await GenotypeSerializer.from_database(session, [ind.id]))[0]

            body = body_develop(genotype.body)
            measures = MorphologicalMeasures(body)
            width[idx] = measures.bounding_box_width
            height[idx] = measures.bounding_box_height
            depth[idx] = measures.bounding_box_depth
            num_hinges[idx] = measures.num_active_hinges
            idx += 1

    df = pd.read_csv('../alifepaper/summary_dar_rotation.csv')
    df['width'] = width
    df['height'] = height
    df['depth'] = depth
    df['num_hinges'] = num_hinges
    df.to_csv('../alifepaper/new_summary.csv')


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
