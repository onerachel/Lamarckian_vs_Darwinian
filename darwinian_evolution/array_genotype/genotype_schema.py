import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from revolve2.core.database.serializers import DbNdarray1xn

DbBase = declarative_base()

class DbArrayGenotype(DbBase):
    __tablename__ = "array_genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True, autoincrement=True
    )

    internal_weights = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )
