from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
from jlo.drl_evolved_body.rl_rerunner import AgentRerunner

async def main() -> None:

    rerunner = AgentRerunner()
    await rerunner.rerun()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
