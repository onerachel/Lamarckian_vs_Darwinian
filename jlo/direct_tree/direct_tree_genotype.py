from __future__ import annotations

from dataclasses import dataclass
import queue
from typing import List, Tuple

from revolve2.core.modular_robot import ActiveHinge
from revolve2.core.modular_robot import Body
from revolve2.core.modular_robot import Brick, Module
from revolve2.serialization import Serializable
from jlo.direct_tree.direct_tree_utils import bfs_iterate_modules, duplicate_subtree

from revolve2.core.database import IncompatibleError, Serializer
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from jlo.direct_tree.genotype_schema import DbBase, DbGenotype

@dataclass
class DirectTreeGenotype(Serializable):
    genotype: Body

    @dataclass
    class _Module:
        position: Tuple[int, int, int]
        forward: Tuple[int, int, int]
        up: Tuple[int, int, int]
        chain_length: int
        module_reference: Module

    def deserialize(self, data: str) -> None:

        slot_queue = queue.Queue()  # infinite FIFO queue

        def append_new_empty_slots(module: Module):
            for slot, _ in enumerate(module.children):
                slot_queue.put((module, slot))

        genotype = self.genotype
        append_new_empty_slots(genotype.core)

        lines = data.splitlines()

        for line in lines:
            fields = line.split()
            module_type = fields[0]
            rotation = float(fields[1])
            parent, slot = slot_queue.get_nowait()
            if module_type != 'none':
                new_module = None
                if module_type == 'brick':
                    new_module = Brick(rotation)
                elif module_type == 'active_hinge':
                    new_module = ActiveHinge(rotation)
                parent.children[slot] = new_module
                append_new_empty_slots(new_module)

            
    def serialize(self) -> str:
        elements = ""
        for parent, elem in bfs_iterate_modules(self.genotype.core, include_none_child=True):
            if parent is not None:
                type = "none"
                rotation = 0.0
                if elem is not None:
                    if isinstance(elem, ActiveHinge):
                        type = 'active_hinge'
                    elif isinstance(elem, Brick):
                        type = 'brick'
                    else:
                        type = 'core'
                    rotation = elem.rotation

                _elem = f"{type} {rotation}\n"
                elements += _elem

        return elements

    def clone(self):
        new_genotype = Body()
        new_genotype.core = duplicate_subtree(self.genotype.core)
        return DirectTreeGenotype(new_genotype)

def develop(genotype: DirectTreeGenotype) -> Body:
    genotype.genotype.finalize()
    return genotype.genotype

class GenotypeSerializer(Serializer[DirectTreeGenotype]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

    @classmethod
    def identifying_table(cls) -> str:
        return DbGenotype.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[DirectTreeGenotype]
    ) -> List[int]:
        dbfitnesses = [
            DbGenotype(serialized_directtree_genome=o.serialize())
            for o in objects
        ]
        session.add_all(dbfitnesses)
        await session.flush()
        ids = [
            dbfitness.id for dbfitness in dbfitnesses if dbfitness.id is not None
        ]  # cannot be none because not nullable. used to silence mypy
        assert len(ids) == len(objects)  # but check just to be sure
        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[DirectTreeGenotype]:
        rows = (
            (await session.execute(select(DbGenotype).filter(DbGenotype.id.in_(ids))))
            .scalars()
            .all()
        )

        if len(rows) != len(ids):
            raise IncompatibleError()

        id_map = {t.id: t for t in rows}
        genotypes = [DirectTreeGenotype(Body()) for _ in ids]
        for id, genotype in zip(ids, genotypes):
            genotype = genotype.deserialize(id_map[id].serialized_directtree_genome)
        return genotypes
