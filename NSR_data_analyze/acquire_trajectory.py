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
from genotype import DbGenotype, GenotypeSerializer
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1 as body_develop
from matplotlib import pyplot as plt
import asyncio
import pandas as pd


async def main() -> None:
    file = '../alifepaper/lamarckian/summary_reproduction_point_nav.csv'
    df = pd.read_csv(file)
    asexual_df = df[df['experiment'] == 'Asexual_Lamarckian']
    asexual_df = asexual_df[asexual_df['generation'] == 29].sort_values('after')

    fig, ax = plt.subplots()

    print('Rerunning robots to acquire trajectory data...')

    num_traj_toplot = 10
    individuals = asexual_df[-num_traj_toplot:].reset_index()
    trajectories = np.zeros((len(individuals[-num_traj_toplot:]), 2, 301))

    for i, ind in individuals.iterrows():
        traj_x, traj_y = await run_robot(ind.genotype_id, ind.experiment, ind.run)
        trajectories[i, 0] = np.array(traj_x)
        trajectories[i, 1] = np.array(traj_y)

    np.save('lamarck_pointnav_asexual_traj', trajectories)


async def run_robot(genotype_id, experiment, run):
    ex = 'sexual_reproduction' if experiment == 'Sexual_Lamarckian' else 'asexual_reproduction'
    db = open_async_database_sqlite('../alifepaper/lamarckian/' + ex + '/point_navigation/run' + str(run))
    async with AsyncSession(db) as session:
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

    cpg_network_structure = make_cpg_network_structure_neighbour(
        active_hinges
    )

    brain_genotype = genotype.brain
    grid_size = 22
    num_potential_joints = ((grid_size ** 2) - 1)
    brain_params = []
    for hinge in active_hinges:
        pos = body.grid_position(hinge)
        cpg_idx = int(pos[0] + pos[1] * grid_size + grid_size ** 2 / 2)
        brain_params.append(brain_genotype.params_array[
                                cpg_idx * 14
                                ])

    for connection in cpg_network_structure.connections:
        hinge1 = connection.cpg_index_highest.index
        pos1 = body.grid_position(active_hinges[hinge1])
        cpg_idx1 = int(pos1[0] + pos1[1] * grid_size + grid_size ** 2 / 2)
        hinge2 = connection.cpg_index_lowest.index
        pos2 = body.grid_position(active_hinges[hinge2])
        cpg_idx2 = int(pos2[0] + pos2[1] * grid_size + grid_size ** 2 / 2)
        rel_pos = relative_pos(pos1[:2], pos2[:2])
        idx = max(cpg_idx1, cpg_idx2)
        brain_params.append(brain_genotype.params_array[
                                idx * 14 + rel_pos
                                ])

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
    traj_x, traj_y = await rerunner.rerun(bot, 5, headless=True)
    return traj_x, traj_y


def relative_pos(pos1, pos2):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]

    mapping = {(1, 0): 1, (1, 1): 2, (0, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6,
               (-1, 1): 7, (1, -1): 8, (2, 0): 9, (0, 2): 10, (-2, 0): 11, (0, -2): 12, (0, 0): 13}

    return mapping[(dx, dy)]


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())