import asyncio, traceback
from tests.conftest import register_fleet

async def test_probe(client):
    await register_fleet(["dgx-01", "dgx-02", "dgx-03"])
    async def one():
        try:
            return await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
        except BaseException as e:
            for line in traceback.format_tb(e.__traceback__):
                if "/backend/app/" in line:
                    print("  APP>", line.strip().split("\n")[0])
            print("  ERR>", type(e).__name__, str(e)[:120])
            return "EXC"
    await asyncio.gather(*[one() for _ in range(3)])
    assert True
