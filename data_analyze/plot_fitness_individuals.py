"""
Plot average, min, and max fitness over generations, using the results of the evolutionary optimizer.

Assumes fitness is a float and database is files.
See program help for what inputs to provide.
"""

import argparse
import matplotlib

import matplotlib.pyplot as plt
# print (plt.get_backend())
# matplotlib.use('TkAgg')
# note: matplotlib newest version doesn't compatible with pycharm, uninstall matplotlib and  pip install matplotlib==3.5.3

import pandas
from revolve2.core.database import open_database_sqlite
from jlo.ea_fixedbody_cppn_revde.revde import DbRevDEOptimizerIndividual
from sqlalchemy.future import select
import os


def plot(database: str, process_id: int) -> None:
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
    # print(df)
    df['robot'] = f"{database}".split("_", 1)[1]
    df.to_csv(f'{database}.tsv', index=False)

    # calculate max min avg
    describe = df[["gen_num", "fitness"]].groupby(by="gen_num").describe()["fitness"]
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()

    # plot max min mean, std
    describe[["max", "mean", "min"]].plot()
    plt.fill_between(range(1, len(mean) + 1), mean - std, mean + std)
    # plt.show()
    plt.savefig(f'{database}.png')

def main() -> None:
    # print("Current working directory: {0}".format(os.getcwd()))
    for run in range(10):
        os.chdir(f'/Users/lj/revolve2/databases/revDE/run{run+1}')

        databases = ['database_babya', 'database_babyb', 'database_blokky', 'database_garrix', 'database_gecko', 'database_insect', 'database_linkin', 'database_longleg', 'database_penguin', 'database_pentapod',
                     'database_queen', 'database_salamander', 'database_squarish',
                     'database_snake', 'database_spider', 'database_stingray', 'database_tinlicker', 'database_turtle', 'database_ww', 'database_zappa']

        for db in databases:
            database = db
            plot(database, 0)

if __name__ == "__main__":
    main()
