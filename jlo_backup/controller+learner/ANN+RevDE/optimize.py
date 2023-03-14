"""Setup and running of the openai es optimization program."""

import argparse
import logging
from random import Random

from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from revolve2.standard_resources import modular_robots


async def main() -> None:
    """Run the optimization process."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "body",
        type=str,
        help="The body of the robot.",
    )
    parser.add_argument(
        "num",
        type=str,
        help="The number of the experiment",
    )
    args = parser.parse_args()
    num = args.num
    body = args.body

    POPULATION_SIZE = 10
    NUM_GENERATIONS = 35 #20
    SCALING = 0.5
    CROSS_PROB = 0.9

    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5

    # random number generator
    rng = Random()
    rng.seed(0)

    file_path = "./data/RevDENN/"+body+"/database"+num

    # database
    database = open_async_database_sqlite(file_path)

    fileh = logging.FileHandler(file_path+"/exp.log")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s")
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.setLevel(logging.INFO)
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh) 

    # process id generator
    process_id_gen = ProcessIdGen()
    process_id = process_id_gen.gen()

    body = modular_robots.get(body)

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
            f"Recovered. Last finished generation: {maybe_optimizer.generation_number - 1}."
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
