from __future__ import annotations

from dataclasses import dataclass
from distutils import core
import queue
from typing import Any, List, Optional, Set, Tuple, cast

from revolve2.core.modular_robot import Brain
from revolve2.core.modular_robot import ActiveHinge, Brick, Module, Core
from revolve2.serialization import Serializable, SerializeError, StaticData
# from direct_tree.direct_tree_utils import bfs_iterate_modules, duplicate_subtree

from revolve2.core.database import IncompatibleError, Serializer
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from .genotype_schema import DbBase, DbArrayGenotype
import numpy.typing as npt
import numpy as np
import itertools
from revolve2.core.database.serializers import DbNdarray1xn, Ndarray1xnSerializer
from random import Random

@dataclass
class ArrayGenotype:
    internal_params: npt.NDArray[np.float_]
    external_params: npt.NDArray[np.float_]

def random_v1(
        length: int,
        rng: Random,
) -> ArrayGenotype:
    nprng = np.random.Generator(
        np.random.PCG64(rng.randint(0, 2 ** 63))
    )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
    internal_params = nprng.standard_normal(length)
    external_params = nprng.standard_normal(1)
    return ArrayGenotype(internal_params, external_params)


def develop(genotype: ArrayGenotype) -> Brain:
    #genotype.genotype.finalize()
    return np.concatenate(genotype.internal_params, genotype.external_params)

class ArrayGenotypeSerializer(Serializer[ArrayGenotype]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await Ndarray1xnSerializer.create_tables(session)

    @classmethod
    def identifying_table(cls) -> str:
        return DbArrayGenotype.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[ArrayGenotype]
    ) -> List[int]:

        int_ids = []
        for obj in objects:
            id = await Ndarray1xnSerializer.to_database(
                session, [obj.internal_params]
            )
            int_ids += id
        assert len(int_ids) == len(objects)

        ext_ids = []
        for obj in objects:
            id = await Ndarray1xnSerializer.to_database(
                session, [obj.external_params]
            )
            ext_ids += id
        assert len(ext_ids) == len(objects)


        dbgenotypes = [DbArrayGenotype() for _ in objects]
        for i, (int_id, ext_id) in enumerate(zip(int_ids, ext_ids)):
            dbgenotypes[i].internal_weights = int_id
            dbgenotypes[i].external_weights = ext_id

        session.add_all(dbgenotypes)
        await session.flush()
        ids = [
            dbgenotype.id for dbgenotype in dbgenotypes
        ]

        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[ArrayGenotype]:

        arrays = (
            (
                await session.execute(
                    select(DbArrayGenotype)
                    .filter(DbArrayGenotype.id.in_(ids))
                )
            )
            .scalars()
            .all()
        )

        int_param_ids = [a.internal_weights for a in arrays]
        ext_param_ids = [a.external_weights for a in arrays]
        internal_params = [(await Ndarray1xnSerializer.from_database(session, [id]))[0] for id in int_param_ids]
        external_params = [(await Ndarray1xnSerializer.from_database(session, [id]))[0] for id in ext_param_ids]

        genotypes: List[ArrayGenotype] = [
            ArrayGenotype(np.array(int_param), np.array(ext_param))
            for int_param, ext_param in zip(internal_params, external_params)
        ]

        return genotypes
