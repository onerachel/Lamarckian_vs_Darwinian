import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base

DbBase = declarative_base()

class DbArrayGenotype(DbBase):
    __tablename__ = "array_genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True, autoincrement=True
    )


class DbArrayGenotypeItem(DbBase):
    __tablename__ = "array_genotype_item"

    array_genotype_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        sqlalchemy.ForeignKey(DbArrayGenotype.id),
        nullable=False,
        primary_key=True,
    )
    array_index = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        primary_key=True,
    )
    value = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
