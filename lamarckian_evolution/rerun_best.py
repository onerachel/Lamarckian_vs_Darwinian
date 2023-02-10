"""Visualize and simulate the best robot from the optimization process."""

import math
import numpy as np

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
from learning_algorithms.EVO.CPG.rerunner import ModularRobotRerunner
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea._database import (
    DbBase,
)
from _optimizer import DbEAOptimizerIndividual
from genotype import DbGenotype, GenotypeSerializer, Genotype
from revolve2.core.database.serializers import FloatSerializer
from array_genotype.array_genotype import ArrayGenotypeSerializer as BrainSerializer, develop as brain_develop
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1 as body_develop
from revolve2.genotypes.cppnwin._genotype import GenotypeSerializer as BodySerializer
from revolve2.actor_controllers.cpg import CpgNetworkStructure, Cpg
from typing import Optional
import argparse

async def main(record_dir: Optional[str], record: bool = False) -> None:

    """Run the script."""
    db = open_async_database_sqlite('lamarckian_database/')
    async with AsyncSession(db) as session:
        individuals = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual.genotype_id, DbEAOptimizerIndividual.final_fitness_id,
                    DbEAOptimizerIndividual.absolute_size, DbEAOptimizerIndividual.proportion, DbEAOptimizerIndividual.num_bricks,
                    DbEAOptimizerIndividual.rel_num_limbs, DbEAOptimizerIndividual.symmetry, DbEAOptimizerIndividual.branching
                    )
                )
            )
            .all()
        )

        fitnesses_ids = [ind.final_fitness_id for ind in individuals]
        fitnesses = np.array([(await FloatSerializer.from_database(session, [id]))[0] for id in fitnesses_ids])
        max_id = np.argsort(fitnesses)[-1]
        print(f"Fitness: {fitnesses[max_id]}")
        ind = individuals[max_id]
        print(f'abs_size: {ind.absolute_size}, proportion: {ind.proportion}, num_bricks: {ind.num_bricks}')
        print(f'rel_num_limbs: {ind.rel_num_limbs}, symmetry: {ind.symmetry}, branching: {ind.branching}')

        genotype_id = individuals[max_id][0]
        genotype_db = (
            (
                await session.execute(
                    select(DbGenotype).filter(
                        DbGenotype.id == genotype_id
                    )
                )
            )
            .all()[0]
        )[0]
        genotype = (await GenotypeSerializer.from_database(session, [genotype_db.id]))[0]

        body = body_develop(genotype.body)

        actor, dof_ids = body.to_actor()
        active_hinges_unsorted = body.find_active_hinges()
        active_hinge_map = {
            active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
        }
        active_hinges = [active_hinge_map[id] for id in dof_ids]

        cpgs = [Cpg(i) for i, _ in enumerate(active_hinges)]
        cpg_network_structure = CpgNetworkStructure(cpgs, set())

        brain_params = []
        for hinge in active_hinges:
            pos = body.grid_position(hinge)
            brain_params.append(genotype.brain.internal_params[int(pos[0] + pos[1] * 22 + pos[2] * 22**2 + 22**3 / 2)])

        initial_state = cpg_network_structure.make_uniform_state(0.5 * math.pi / 2.0)
        weight_matrix = (
            cpg_network_structure.make_connection_weights_matrix_from_params(brain_params)
        )
        dof_ranges = cpg_network_structure.make_uniform_dof_ranges(1.0)
        brain = BrainCpgNetworkStatic(
            initial_state,
            cpg_network_structure.num_cpgs,
            weight_matrix,
            dof_ranges,
        )

        bot = ModularRobot(body, brain)

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(bot, 5, record_dir, record)


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        "--record",
        action='store_true',
        help="Record the robot simulation.",
    )
    parser.add_argument(
        "record_dir", 
        type=str,
        nargs='?',
        const='none',
        help="The directory where the recording is goint to be saved."
    )
    args = parser.parse_args()

    asyncio.run(main(args.record_dir, args.record))
