"""Setup and running of the optimize modular program."""

import logging
from random import Random

import multineat
from genotype import random as random_genotype
from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import DbId


async def main() -> None:
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
    database = open_async_database_sqlite("./lamarc_asex_database", create=True)

    fileh = logging.FileHandler("lamarc_asex_database/exp.log")
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

    initial_population = [
        random_genotype(innov_db_body, rng, NUM_INITIAL_MUTATIONS, robot_grid_size=GRID_SIZE)
        for _ in range(POPULATION_SIZE)
    ]

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


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
