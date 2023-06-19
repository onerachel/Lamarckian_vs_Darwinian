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
    DbEAOptimizerGeneration,
    DbEAOptimizerParent,
)
from _optimizer import DbEAOptimizerIndividual
from revolve2.core.database.serializers import FloatSerializer, DbFloat
from matplotlib import pyplot as plt
from revolve2.core.optimization import DbId



def read_after(database: str, db_id: DbId) -> None:

    db = open_database_sqlite(database)
    df = pandas.read_sql(
        select(
            DbEAOptimizerParent,
        ).filter(
            (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizerParent.ea_optimizer_id)
            & (DbEAOptimizerIndividual.individual_id == DbEAOptimizerParent.child_individual_id)
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
    for run in range(nr_runs):
        database = f"run{run+1}"
        data = read_after(database, DbId("morphevo"))
        data['run'] = f'{run + 1}'
        table.append(data)
        final1 = pandas.concat(table, ignore_index=True)

    final1['experiment'] = 'Lamarckian+Learning'

    ##  Darwinian
    path_darwinian = path+'darwinian'
    os.chdir(f'{path_darwinian}/')
    table2 = []
    for run in range(nr_runs):
        database = f"run{run+1}"
        data = read_after(database, DbId("morphevo"))
        data['run'] = f'{run + 1}'
        table2.append(data)
        final2 = pandas.concat(table2, ignore_index=True)
    final2['experiment'] = 'Darwinian+Learning'

    final = pandas.concat([final1, final2], ignore_index=True)
    final.to_csv(os.path.join(path, r"summary_parent.csv"), index=False)

if __name__ == "__main__":
    main()