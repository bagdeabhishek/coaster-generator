#!/bin/bash
# OAuth Network Diagnostic Script
# Run this inside your Coolify container

echo "=== Installing Required Tools ==="
apt-get update -qq && apt-get install -y -qq iproute2 curl dnsutils python3 python3-pip 2>/dev/null || \
apk add --no-cache iproute2 curl bind-tools python3 py3-pip 2>/dev/null || \
echo "WARNING: Could not install packages automatically"
echo ""

echo "=== OAuth Network Diagnostics ==="
echo ""

echo "1. DNS Resolution Test:"
getent hosts accounts.google.com || nslookup accounts.google.com || echo "DNS FAILED"
echo ""

echo "2. Network Interface MTU:"
ip addr show | grep -E "mtu|inet" | head -20
echo ""

echo "3. Route to Google:"
ip route get 192.178.211.84 2>/dev/null || echo "No specific route"
echo ""

echo "4. TCP Connection Test (Port 443):"
timeout 5 bash -c "echo >/dev/tcp/accounts.google.com/443" 2>&1 && echo "TCP: SUCCESS" || echo "TCP: FAILED"
echo ""

echo "5. HTTPS with curl (IPv4 only):"
curl -sI -4 --max-time 5 https://accounts.google.com/.well-known/openid-configuration 2>&1 | head -5
echo ""

echo "6. HTTPS with curl (IPv6 only):"
curl -sI -6 --max-time 5 https://accounts.google.com/.well-known/openid-configuration 2>&1 | head -5
echo ""

echo "7. Test with explicit IPv4 address:"
curl -sI --max-time 5 https://192.178.211.84/.well-known/openid-configuration -H "Host: accounts.google.com" 2>&1 | head -5
echo ""

echo "8. Python httpx test:"
python3 -c "
import asyncio
import httpx
import socket

async def test():
    print(f'Python socket.getaddrinfo: {socket.getaddrinfo(\"accounts.google.com\", 443)[0]}')
    try:
        transport = httpx.AsyncHTTPTransport(local_address=\"0.0.0.0\")
        async with httpx.AsyncClient(transport=transport, timeout=5) as client:
            r = await client.get(\"https://accounts.google.com/.well-known/openid-configuration\")
            print(f'httpx with IPv4 transport: {r.status_code}')
    except Exception as e:
        print(f'httpx with IPv4 transport: FAILED - {type(e).__name__}: {e}')
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(\"https://accounts.google.com/.well-known/openid-configuration\")
            print(f'httpx default: {r.status_code}')
    except Exception as e:
        print(f'httpx default: FAILED - {type(e).__name__}: {e}')

asyncio.run(test())
" 2>&1
echo ""

echo "=== End of Diagnostics ==="
