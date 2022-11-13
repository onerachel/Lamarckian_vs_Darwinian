import argparse
import logging
from random import Random

from config import (CONTROL_FREQUENCY, NUM_PARALLEL_AGENT, SAMPLING_FREQUENCY,
                    SIMULATION_TIME)
from optimizer import PPOOptimizer
from revolve2.standard_resources import modular_robots
import os


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

    file_path = "./data/PPO/"+body+"/database"+num
    os.makedirs(file_path, exist_ok=True)

    fileh = logging.FileHandler(file_path+"/exp.log", mode='w')
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s")
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.setLevel(logging.INFO)
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)

    logging.info(f"Starting learning")

    # random number generator
    rng = Random()
    rng.seed(42)

    body = modular_robots.get(body)

    optimizer = PPOOptimizer(
        rng=rng,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        simulation_time=SIMULATION_TIME,
        visualize=args.visualize,
        num_agents=NUM_PARALLEL_AGENT,
        robot_body=body,
        file_path = file_path
    )

    logging.info("Starting learning process..")

    await optimizer.train(from_checkpoint=args.from_checkpoint)

    logging.info(f"Finished learning.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
