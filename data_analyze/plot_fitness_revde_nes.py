#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2 11:28:32 2022

@author: LJ
"""

import matplotlib.pyplot as plt
import pandas
from revolve2.core.database import open_database_sqlite
from jlo.ea_fixedbody_cppn_revde.revde_optimizer import DbRevDEOptimizerIndividual
from sqlalchemy.future import select
import os


def read(database: str, process_id: int) -> None:
    """
    Do the actual plotting.

    :param database: The database with the results.
    :param process_id: The process id in the database of the optimizer to plot. If you don't know what you are doing, '0' is probably correct.
    """
    # open the database
    db = open_database_sqlite(database)
    # read the optimizer data into a pandas dataframe
    df = pandas.read_sql(
        select(DbRevDEOptimizerIndividual).filter(
            DbRevDEOptimizerIndividual.process_id == process_id
        ),
        db,
    )
    return df


def plot(df):
    # print(df)
    describe = df.groupby(["individual"]).describe()["fitness"]
    # print(describe)
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()

    # # plot max min mean, std
    describe[["max", "mean", "min"]].plot()
    plt.fill_between(range(1, len(mean) + 1), mean - std, mean + std)
    plt.show()
    # plt.savefig("nn_revde.png")


def main() -> None:
    # print("Current working directory: {0}".format(os.getcwd()))
    path = '/Users/lj/revolve2/databases_eval580/ANN+RevDE'
    df = []
    robot_zoo = ['babya', 'babyb', 'blokky', 'garrix', 'gecko', 'insect', 'linkin', 'longleg', 'penguin', 'pentapod',
                 'queen', 'salamander', 'squarish',
                 'snake', 'spider', 'stingray', 'tinlicker', 'turtle', 'ww', 'zappa']
    for robot in robot_zoo:
        os.chdir(f'{path}/{robot}')
        table = []
        for run in range(10):
            database = f"database{run + 1}"
            data = read(database, 0)
            data['run'] = f'{run + 1}'
            table.append(data)
            final = pandas.concat(table, ignore_index=True)
        final['robot'] = f'{robot}'
        df.append(final)

    final_runs = pandas.concat(df, ignore_index=True)
    plot(final_runs)
    final_runs.to_csv(os.path.join(path, r"summary.csv"), index=False)


if __name__ == "__main__":
    main()
