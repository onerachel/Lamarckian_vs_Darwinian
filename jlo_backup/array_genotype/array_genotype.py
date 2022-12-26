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
from .genotype_schema import DbBase, DbArrayGenotype, DbArrayGenotypeItem
import numpy.typing as npt
import numpy as np
import itertools
from revolve2.core.database.serializers import DbNdarray1xn, Ndarray1xnSerializer
from random import Random

@dataclass
class ArrayGenotype:
    genotype: npt.NDArray[np.float_] # vector

def random_v1(
        length: int,
        rng: Random,
) -> ArrayGenotype:
    nprng = np.random.Generator(
        np.random.PCG64(rng.randint(0, 2 ** 63))
    )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
    params = nprng.standard_normal(length)
    return ArrayGenotype(params)


def develop(genotype: ArrayGenotype) -> Brain:
    genotype.genotype.finalize()
    return genotype.genotype

class ArrayGenotypeSerializer(Serializer[ArrayGenotype]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

    @classmethod
    def identifying_table(cls) -> str:
        return DbArrayGenotype.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[ArrayGenotype]
    ) -> List[int]:
        dbgenotypes = [DbArrayGenotype() for _ in objects]
        session.add_all(dbgenotypes)
        await session.flush()
        ids = [
            dbgenotype.id for dbgenotype in dbgenotypes
        ]

        items = [
            DbArrayGenotypeItem(array_genotype_id=id, array_index=i, value=v)
            for id, object in zip(ids, objects)
            for i, v in enumerate(object.genotype)
        ]

        session.add_all(items)

        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[ArrayGenotype]:
        items = (
            (
                await session.execute(
                    select(DbArrayGenotypeItem)
                    .filter(DbArrayGenotypeItem.array_genotype_id.in_(ids))
                    .order_by(DbArrayGenotypeItem.array_index)
                )
            )
            .scalars()
            .all()
        )

        genotypes: List[ArrayGenotype] = [
            ArrayGenotype(np.array([item.value for item in group]))
            for _, group in itertools.groupby(
                items, key=lambda item: cast(int, item.array_genotype_id)
            )
        ]

        return genotypes
