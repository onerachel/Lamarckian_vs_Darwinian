import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base

DbBase = declarative_base()

class DbGenotype(DbBase):
    __tablename__ = "directtree_genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    serialized_directtree_genome = sqlalchemy.Column(sqlalchemy.String, nullable=False)