"""Visualize and simulate the best robot from the optimization process."""

import argparse
import math
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



def plot(database: str, db_id: DbId) -> None:

    """Run the script."""
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
            & (
                DbEAOptimizerGeneration.individual_id
                == DbEAOptimizerIndividual.individual_id
            )
        ),
        db,
    )
    describe = (
        df[["generation_index", "value"]]
        .groupby(by="generation_index")
        .describe()["value"]
    )
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()

    # plot max min mean, std
    describe[["max", "mean", "min"]].plot()
    plt.fill_between(range(len(mean)), mean - std, mean + std)
    plt.show()



def main() -> None:
    """Run this file as a command line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to plot.",
    )
    parser.add_argument("db_id", type=str, help="The id of the ea optimizer to plot.")
    args = parser.parse_args()

    plot(args.database, DbId(args.db_id))

if __name__ == "__main__":
    main()