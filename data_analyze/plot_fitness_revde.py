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
    df['robot'] = f"{database}".split("_", 1)[1]
    return df

def plot(df):
    print(df)
    describe = df.groupby(["individual"]).describe()["fitness"]
    print(describe)
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()

    # # plot max min mean, std
    describe[["max", "mean", "min"]].plot()
    plt.fill_between(range(1, len(mean) + 1), mean - std, mean + std)
    plt.show()
    # plt.savefig(revde.png)


def main() -> None:
    # print("Current working directory: {0}".format(os.getcwd()))
    path = '/Users/lj/revolve2/databases/revDE'
    df = []
    for run in range(10):
        os.chdir(f'{path}/run{run + 1}')

        table = []
        databases = ['database_babya', 'database_babyb', 'database_blokky', 'database_garrix', 'database_gecko',
                     'database_insect', 'database_linkin', 'database_longleg', 'database_penguin', 'database_pentapod',
                     'database_queen', 'database_salamander', 'database_squarish',
                     'database_snake', 'database_spider', 'database_stingray', 'database_tinlicker', 'database_turtle',
                     'database_ww', 'database_zappa']
        for db in databases:
            # iterate each database in a run and append data in an empty list, 20 dataframes
            database = db
            data = read(database, 0)
            table.append(data)
        # concatenate 20 dataframes for each run
        final = pandas.concat(table, ignore_index=True)
        # add a column
        final['run'] = f'{run + 1}'
        df.append(final)
    # concatenate runs
    final_runs = pandas.concat(df, ignore_index=True)
    plot(final_runs)
    final_runs.to_csv(os.path.join(path, r"summary.csv"), index=False)


if __name__ == "__main__":
    main()
