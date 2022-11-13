"""Setup and running of the openai es optimization program."""

import logging
from random import Random

from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from revolve2.standard_resources.modular_robots import *


async def main() -> None:
    """Run the optimization process."""
    POPULATION_SIZE = 10 # initial sample
    NUM_GENERATIONS = 10
    SCALING = 0.5
    CROSS_PROB = 0.9

    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5 #60

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # random number generator
    rng = Random()
    rng.seed(0)

    # robot_zoo = ['babya']

    robot_zoo = ['babya', 'babyb', 'blokky', 'garrix', 'gecko', 'insect', 'linkin', 'longleg', 'penguin', 'pentapod', 'queen', 'salamander', 'squarish',
    'snake', 'spider', 'stingray', 'tinlicker', 'turtle', 'ww', 'zappa']

    for robot in robot_zoo:
        # database
        database = open_async_database_sqlite(f"./database_{robot}")

        # process id generator
        process_id_gen = ProcessIdGen()
        process_id = process_id_gen.gen()

        # body = babya()
        body = get(robot)
        maybe_optimizer = await Optimizer.from_database(
            database=database,
            process_id=process_id,
            process_id_gen=process_id_gen,
            rng=rng,
            robot_body=body,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
        )
        if maybe_optimizer is not None:
            logging.info(
                f"Recovered. Last finished generation: {maybe_optimizer.generation_number}."
            )
            optimizer = maybe_optimizer
        else:
            logging.info("No recovery data found. Starting at generation 0.")
            optimizer = await Optimizer.new(
                database,
                process_id,
                process_id_gen,
                rng,
                POPULATION_SIZE,
                body,
                simulation_time=SIMULATION_TIME,
                sampling_frequency=SAMPLING_FREQUENCY,
                control_frequency=CONTROL_FREQUENCY,
                num_generations=NUM_GENERATIONS,
                scaling=SCALING,
                cross_prob=CROSS_PROB,
            )

        logging.info("Starting optimization process..")

        await optimizer.run()

        logging.info("Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
