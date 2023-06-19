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
    DbEAOptimizerGeneration,
    DbEAOptimizerParent
)
from revolve2.core.database import open_database_sqlite
from revolve2.core.modular_robot import Body, Core, Brick, ActiveHinge, Module
import apted
from apted import APTED
from revolve2.core.database.serializers import FloatSerializer, DbFloat


def body_to_tree(body: Body) -> str:
    tree = body_to_tree_recur(body.core)

    return tree

def body_to_tree_recur(node: Module) -> str:
    tree = None
    if isinstance(node, Core):
        tree = '{c'
    if isinstance(node, Brick):
        tree = '{b'
    if isinstance(node, ActiveHinge):
        tree = '{a'

    if node is None:
        tree = '{e}'
        return tree
    else:
        for child in node.children:
            tree = tree + body_to_tree_recur(child)
        tree = tree + '}'
        return tree

async def main() -> None:

    folder = '/net/ripper3/home/jieluo/jlo_workspace/experiment_Lamarck_darwin_v1_random/rotation/'
    num_runs = 20

    for exp in ('darwinian', 'lamarckian'):
        print(exp)

        parent1_dist = np.zeros(1500*num_runs)
        idx = 0

        for ran_num in range(1,num_runs+1):
            print(ran_num)

            db = open_database_sqlite(folder + exp + '/run' + str(ran_num))
            df = pd.read_sql(
                select(
                    DbEAOptimizer,
                    DbEAOptimizerGeneration,
                    DbEAOptimizerIndividual,
                    DbGenotype,
                    DbFloat
                ).filter(
                    (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
                    & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
                    & (
                        DbEAOptimizerGeneration.individual_id
                        == DbEAOptimizerIndividual.individual_id
                    )
                    & (DbGenotype.id == DbEAOptimizerIndividual.genotype_id)
                    & (DbEAOptimizerIndividual.final_fitness_id == DbFloat.id)
                ),
                db,
            )

            df_parents = pd.read_sql(
                select(
                    DbEAOptimizer,
                    DbEAOptimizerParent
                ).filter(
                    (DbEAOptimizer.id == DbEAOptimizerParent.ea_optimizer_id)
                ),
                db,
            )
            for i, ind in df.iterrows():

                parents = df_parents[df_parents['child_individual_id'] == ind.individual_id]
                if parents.shape[0] > 0:
                    parent1 = df[df['individual_id'] == parents.iloc[0]['parent_individual_id']].iloc[0]
                    parent2 = df[df['individual_id'] == parents.iloc[1]['parent_individual_id']].iloc[0]
                    best_parent = parent1 if parent1.value >= parent2.value else parent2

                    db = open_async_database_sqlite(folder + exp + '/run' + str(ran_num))
                    async with AsyncSession(db) as session:
                        child_genotype = (await GenotypeSerializer.from_database(session, [ind.genotype_id]))[0]
                        parent1_genotype = (await GenotypeSerializer.from_database(session, [int(best_parent.genotype_id)]))[0]

                    child_body = body_develop(child_genotype.body)
                    child_tree = body_to_tree(child_body)

                    parent1_body = body_develop(parent1_genotype.body)
                    parent1_tree = body_to_tree(parent1_body)

                    child_tree = apted.helpers.Tree.from_text(child_tree)
                    parent1_tree = apted.helpers.Tree.from_text(parent1_tree)

                    parent1_apted = APTED(child_tree, parent1_tree)

                    parent1_dist[idx] = parent1_apted.compute_edit_distance()
                idx += 1

        df = pd.read_csv(folder + exp + '/summary.csv')
        df['parent_distance'] = parent1_dist
        df.to_csv(folder + exp + '/new_new_summary_carlo.csv')


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
        