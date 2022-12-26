import argparse

import matplotlib.pyplot as plt
import pandas
from CPG.revde_optimizer import (DbRevDEOptimizerBestIndividual,
                             DbRevDEOptimizerIndividual)
from revolve2.core.database import open_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization import DbId



def plot(database: str, db_id: DbId) -> None:
    """
    Do the actual plotting.

    :param database: The database with the results.
    :param process_id: The process id in the database of the optimizer to plot. If you don't know what you are doing, '0' is probably correct.
    """
    # open the database
    db = open_database_sqlite(database)
    # read the optimizer data into a pandas dataframe
    df = pandas.read_sql(
        select(DbRevDEOptimizerIndividual),
        db,
    )
    # calculate max min avg
    describe = df.sort_values(by='individual')[["fitness"]]

    describe.plot()
    plt.show()


def main() -> None:
    """Run the program."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to plot.",
    )
    parser.add_argument("db_id", type=str, help="The id of the ea optimizer to plot.")
    args = parser.parse_args()

    plot(args.database, args.db_id)


if __name__ == "__main__":
    main()
