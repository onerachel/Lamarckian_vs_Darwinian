"""Setup and running of the optimize modular program."""

import logging
from random import Random

import multineat
from genotype import random as random_genotype
from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import DbId
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from genotype import DbGenotype, GenotypeSerializer, Genotype
from revolve2.core.database.serializers import FloatSerializer


async def main(old_database) -> None:
    """Run the optimization process."""
    # number of initial mutations for body and brain CPPNWIN networks
    NUM_INITIAL_MUTATIONS = 10

    SIMULATION_TIME = 40
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5

    POPULATION_SIZE = 50
    OFFSPRING_SIZE = 25
    NUM_GENERATIONS = 29

    GRID_SIZE = 22

    # database
    database = open_async_database_sqlite("./darw_asex_database", create=True)

    fileh = logging.FileHandler("darw_asex_database/exp.log")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s")
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.setLevel(logging.INFO)
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)

    # random number generator
    rng = Random()

    # unique database identifier for optimizer
    db_id = DbId.root("morphevo")

    # multineat innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    initial_population = await database_population(old_database, POPULATION_SIZE)

    maybe_optimizer = await Optimizer.from_database(
        database=database,
        db_id=db_id,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_generations=NUM_GENERATIONS,
        offspring_size=OFFSPRING_SIZE,
        grid_size=GRID_SIZE
    )
    if maybe_optimizer is not None:
        optimizer = maybe_optimizer
    else:
        optimizer = await Optimizer.new(
            database=database,
            db_id=db_id,
            initial_population=initial_population,
            rng=rng,
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            offspring_size=OFFSPRING_SIZE,
            grid_size = GRID_SIZE
        )

    logging.info("Starting moprhology optimization")

    await optimizer.run()

    logging.info("Finished optimizing morphology.")

async def database_population(database: str, population_size):
    db = open_async_database_sqlite(database)

    genotypes = []
    async with AsyncSession(db) as session:

        genotype_ids = list(range(1, population_size + 1))
        for id in genotype_ids:
            genotype_db = (
                (
                    await session.execute(
                        select(DbGenotype).filter(
                            DbGenotype.id == id
                        )
                    )
                )
                .all()[0]
            )[0]
            genotype = (await GenotypeSerializer.from_database(session, [genotype_db.id]))[0]
            genotypes.append(genotype)

    return genotypes

if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to use for the rerun.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.database))
