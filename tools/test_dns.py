import asyncio
import httpx
import aiohttp

async def test():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://accounts.google.com/.well-known/openid-configuration") as resp:
                print("aiohttp:", resp.status)
    except Exception as e:
        print("aiohttp error:", e)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://accounts.google.com/.well-known/openid-configuration")
            print("httpx:", resp.status_code)
    except Exception as e:
        print("httpx error:", type(e), e)

asyncio.run(test())
