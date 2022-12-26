"""Visualize and simulate the best robot from the optimization process."""

import math
from brain import RevDENNbrain
from network import Actor
from config import NUM_OBS_TIMES
from revde_optimizer import DbRevDEOptimizerIndividual
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import Ndarray1xnSerializer
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
from rerunner import ModularRobotRerunner
from revolve2.standard_resources import modular_robots
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
import argparse
import torch


async def main() -> None:

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

    file_path = "./data/RevDENN/"+body+"/database"+num


    """Run the script."""
    db = open_async_database_sqlite(file_path)
    async with AsyncSession(db) as session:
        best_individual = (
            (
                await session.execute(
                    select(DbRevDEOptimizerIndividual).order_by(
                        DbRevDEOptimizerIndividual.fitness.desc()
                    )
                )
            )
            .scalars()
            .all()[0]
        )

        params = [
            p
            for p in (
                await Ndarray1xnSerializer.from_database(
                    session, [best_individual.individual]
                )
            )[0]
        ]

        print(f"fitness: {best_individual.fitness}")
        print(f"params: {params}")

        body = modular_robots.get(body)

        actor, dof_ids = body.to_actor()
        active_hinges_unsorted = body.find_active_hinges()
        active_hinge_map = {
            active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
        }
        active_hinges = [active_hinge_map[id] for id in dof_ids]

        brain = RevDENNbrain()
        controller = brain.make_controller(body, dof_ids)
        controller.load_parameters(torch.tensor(params, dtype=torch.float32))

        bot = ModularRobot(body, brain)

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(body, controller, 5)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
