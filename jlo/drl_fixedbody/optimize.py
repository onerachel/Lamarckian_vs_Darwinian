import logging
import argparse

from optimizer import PPOOptimizer 
from random import Random
from config import NUM_PARALLEL_AGENT, SAMPLING_FREQUENCY, CONTROL_FREQUENCY, SIMULATION_TIME
from revolve2.standard_resources.modular_robots import *

async def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_checkpoint",
        action="store_true",
        help="Resumes training from past checkpoint if True.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="visualize the simulation if True.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting learning")

    # random number generator
    rng = Random()
    rng.seed(42)

    body = gecko()

    optimizer = PPOOptimizer(
        rng=rng,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        simulation_time=SIMULATION_TIME,
        visualize=args.visualize,
        num_agents=NUM_PARALLEL_AGENT,
        robot_body=body,
    )

    logging.info("Starting learning process..")

    await optimizer.train(from_checkpoint=args.from_checkpoint)

    logging.info(f"Finished learning.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
