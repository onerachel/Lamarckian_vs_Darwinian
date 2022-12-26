"""Setup and running of the openai es optimization program."""

import argparse
import logging
from random import Random

from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import DbId
from revolve2.standard_resources import modular_robots
from revde_optimizer import DbRevDEOptimizerBestIndividual
from revolve2.core.database.serializers import Ndarray1xnSerializer
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
import math

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select


async def main(body, num) -> None:
    """Run the optimization process."""

    POPULATION_SIZE = 10
    NUM_GENERATIONS = 10
    SCALING = 0.5
    CROSS_PROB = 0.9

    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5

    # database
    database = open_async_database_sqlite('test/RevDECPG/' + str(body) + '/database_' + str(num), create=True)

    fileh = logging.FileHandler('test/RevDECPG/' + str(body) + '/database_' + str(num) + '/exp.log')
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s")
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.setLevel(logging.INFO)
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)

    # random number generator
    rng = Random()
    rng.seed(42)

    # unique database identifier for optimizer
    db_id = DbId.root('controlopt')

    body = modular_robots.get(body)

    maybe_optimizer = await Optimizer.from_database(
        database=database,
        db_id=db_id,
        rng=rng,
        robot_body=body,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_generations=NUM_GENERATIONS,
    )
    if maybe_optimizer is not None:
        optimizer = maybe_optimizer
    else:
        optimizer = await Optimizer.new(
            database=database,
            db_id=db_id,
            rng=rng,
            population_size=POPULATION_SIZE,
            robot_body=body,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            scaling=SCALING,
            cross_prob=CROSS_PROB,
        )

    logging.info("Starting controller optimization process..")

    await optimizer.run()

    logging.info("Finished optimizing controller.")

    async with AsyncSession(database) as session:
        best_individual = (
            (
                await session.execute(
                    select(DbRevDEOptimizerBestIndividual).order_by(
                        DbRevDEOptimizerBestIndividual.fitness.desc()
                    )
                )
            )
            .scalars()
            .all()[0]
        )

        best_fitness = best_individual.fitness
        best_individual_id = best_individual.individual
        best_params = (await Ndarray1xnSerializer.from_database(session, [best_individual_id]))[0]

        return best_params, best_fitness


if __name__ == "__main__":
    import asyncio

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
    body = args.body
    num = args.num

    asyncio.run(main(body, num))
