import socket

# Save original getaddrinfo
original_getaddrinfo = socket.getaddrinfo

def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    # Force family to AF_INET (IPv4)
    return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = ipv4_only_getaddrinfo

import httpx
import asyncio

async def test():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://accounts.google.com/.well-known/openid-configuration")
            print("httpx:", resp.status_code)
    except Exception as e:
        print("httpx error:", type(e), e)

asyncio.run(test())
