"""Visualize and simulate the best robot from the optimization process."""

import argparse
import math
import os

import numpy as np
import pandas

from revolve2.core.database import open_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea._database import (
    DbBase,
    DbEAOptimizer,
    DbEAOptimizerGeneration
)
from _optimizer import DbEAOptimizerIndividual
from revolve2.core.database.serializers import FloatSerializer, DbFloat
from matplotlib import pyplot as plt
from revolve2.core.optimization import DbId



def read_after(database: str, db_id: DbId) -> None:

    db = open_database_sqlite(database)
    df = pandas.read_sql(
        select(
            DbEAOptimizer,
            DbEAOptimizerGeneration,
            DbEAOptimizerIndividual,
            DbFloat,
        ).filter(
            (DbEAOptimizer.db_id == db_id.fullname)
            & (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.final_fitness_id == DbFloat.id)
            # & (DbEAOptimizerIndividual.starting_fitness_id == DbFloat.id)
            & (
                DbEAOptimizerGeneration.individual_id
                == DbEAOptimizerIndividual.individual_id
            )
        ),
        db,
    )
    return df


def read_before(database: str, db_id: DbId) -> None:

    db = open_database_sqlite(database)
    df = pandas.read_sql(
        select(
            DbEAOptimizer,
            DbEAOptimizerGeneration,
            DbEAOptimizerIndividual,
            DbFloat,
        ).filter(
            (DbEAOptimizer.db_id == db_id.fullname)
            & (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
            # & (DbEAOptimizerIndividual.final_fitness_id == DbFloat.id)
            & (DbEAOptimizerIndividual.starting_fitness_id == DbFloat.id)
            & (
                DbEAOptimizerGeneration.individual_id
                == DbEAOptimizerIndividual.individual_id
            )
        ),
        db,
    )
    return df

def main() -> None:
    path = '/net/ripper3/home/jieluo/jlo_workspace/experiment_Lamarck_darwin/point_navigation/'
    nr_runs = 10

    ##  Lamarckian
    path_lamarckian = path+'lamarckian'
        # '/Users/lj/revolve2/databases/darwinian_rotation'
    os.chdir(f'{path_lamarckian}/')
    table = []
    table_before = []
    for run in range(nr_runs):
        database = f"run{run+1}"
        data = read_after(database, DbId("morphevo"))
        data['run'] = f'{run + 1}'
        table.append(data)
        final_after = pandas.concat(table, ignore_index=True)

        data_before = read_before(database, DbId("morphevo"))
        data_before['run'] = f'{run + 1}'
        table_before.append(data_before)
        final_before = pandas.concat(table_before, ignore_index=True)


    final_after['experiment'] = 'Lamarckian+Learning'
    final_before = final_before.rename(columns={'value': 'before'})
    final1 = pandas.concat([final_after, final_before['before']], axis=1)
    final1['learning_delta'] = final1['value'] - final1['before']

    ##  Darwinian
    path_darwinian = path+'darwinian'
    os.chdir(f'{path_darwinian}/')
    table2 = []
    table_before2 = []
    for run in range(nr_runs):
        database = f"run{run+1}"
        data = read_after(database, DbId("morphevo"))
        data['run'] = f'{run + 1}'
        table2.append(data)
        final_after2 = pandas.concat(table2, ignore_index=True)

        data_before2 = read_before(database, DbId("morphevo"))
        data_before2['run'] = f'{run + 1}'
        table_before2.append(data_before2)
        final_before2 = pandas.concat(table_before2, ignore_index=True)

    final_after2['experiment'] = 'Darwinian+Learning'
    final_before2 = final_before2.rename(columns={'value': 'before'})
    final2 = pandas.concat([final_after2, final_before2['before']], axis=1)
    final2['learning_delta'] = final2['value'] - final2['before']

    final = pandas.concat([final1, final2], ignore_index=True)
    final.to_csv(os.path.join(path, r"summary.csv"), index=False)

if __name__ == "__main__":
    main()