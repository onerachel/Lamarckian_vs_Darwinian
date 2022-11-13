from rerunner import AgentRerunner
from revolve2.standard_resources.modular_robots import *


async def main() -> None:

    rerunner = AgentRerunner()
    body = zappa()
    await rerunner.rerun(body)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
