#!/usr/bin/env python3
"""
Google OAuth Flow Debugger
Run this inside your Coolify container to diagnose OAuth issues step by step
"""

import asyncio
import os
import sys
import socket
import json
from datetime import datetime

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")

def print_info(msg):
    print(f"  {msg}")

async def step_1_check_environment():
    """Step 1: Check environment variables"""
    print_section("STEP 1: Environment Variables")
    
    required_vars = {
        'OAUTH_GOOGLE_CLIENT_ID': 'Google OAuth Client ID',
        'OAUTH_GOOGLE_CLIENT_SECRET': 'Google OAuth Client Secret',
        'PUBLIC_BASE_URL': 'Public base URL for callbacks'
    }
    
    all_good = True
    results = {}
    
    for var, description in required_vars.items():
        value = os.environ.get(var, '')
        if value:
            masked = value[:10] + '...' if len(value) > 15 else value
            print_success(f"{var}: {masked}")
            results[var] = 'SET'
        else:
            print_error(f"{var}: NOT SET ({description})")
            results[var] = 'MISSING'
            all_good = False
    
    # Validate PUBLIC_BASE_URL format
    public_url = os.environ.get('PUBLIC_BASE_URL', '')
    if public_url:
        if public_url.endswith('/'):
            print_warning("PUBLIC_BASE_URL has trailing slash - this will be removed")
        if not public_url.startswith(('http://', 'https://')):
            print_error("PUBLIC_BASE_URL must start with http:// or https://")
            all_good = False
    
    return all_good, results

async def step_2_check_network():
    """Step 2: Check network connectivity"""
    print_section("STEP 2: Network Connectivity")
    
    tests = [
        ('DNS Resolution', 'accounts.google.com', 443),
        ('Google OAuth Endpoint', 'oauth2.googleapis.com', 443),
        ('Google Token Endpoint', 'oauth2.googleapis.com', 443),
    ]
    
    all_good = True
    
    for name, host, port in tests:
        try:
            # DNS resolution
            addr_info = socket.getaddrinfo(host, port)
            ip = addr_info[0][4][0]
            print_success(f"{name}: {host} -> {ip}")
            
            # TCP connection test
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5
            )
            writer.close()
            await writer.wait_closed()
            print_success(f"  TCP connection to {host}:{port}: OK")
            
        except socket.gaierror as e:
            print_error(f"{name}: DNS resolution failed - {e}")
            all_good = False
        except OSError as e:
            print_error(f"{name}: Connection failed - {e}")
            print_info(f"  Error code: {e.errno}")
            all_good = False
        except asyncio.TimeoutError:
            print_error(f"{name}: Connection timeout")
            all_good = False
        except Exception as e:
            print_error(f"{name}: Unexpected error - {type(e).__name__}: {e}")
            all_good = False
    
    return all_good

async def step_3_check_https():
    """Step 3: Check HTTPS/SSL connectivity"""
    print_section("STEP 3: HTTPS/SSL Connectivity")
    
    try:
        import httpx
        print_success("httpx is installed")
        print_info(f"Version: {httpx.__version__}")
    except ImportError:
        print_error("httpx is not installed!")
        print_info("Install with: pip install httpx")
        return False
    
    all_good = True
    
    # Test with different configurations
    tests = [
        ("Default httpx client", None),
        ("IPv4-only transport", httpx.AsyncHTTPTransport(local_address="0.0.0.0")),
    ]
    
    for test_name, transport in tests:
        try:
            if transport:
                client = httpx.AsyncClient(transport=transport, timeout=10)
            else:
                client = httpx.AsyncClient(timeout=10)
            
            response = await client.get(
                "https://accounts.google.com/.well-known/openid-configuration"
            )
            await client.aclose()
            
            if response.status_code == 200:
                print_success(f"{test_name}: HTTP {response.status_code}")
                # Extract issuer for verification
                try:
                    data = response.json()
                    print_info(f"  Issuer: {data.get('issuer', 'N/A')}")
                    print_info(f"  Authorization endpoint: {data.get('authorization_endpoint', 'N/A')[:50]}...")
                except:
                    pass
            else:
                print_error(f"{test_name}: HTTP {response.status_code}")
                all_good = False
                
        except Exception as e:
            print_error(f"{test_name}: {type(e).__name__}: {e}")
            all_good = False
    
    return all_good

async def step_4_check_oauth_config():
    """Step 4: Check OAuth configuration"""
    print_section("STEP 4: OAuth Configuration Validation")
    
    client_id = os.environ.get('OAUTH_GOOGLE_CLIENT_ID', '')
    public_url = os.environ.get('PUBLIC_BASE_URL', 'http://localhost:3000')
    
    if public_url.endswith('/'):
        public_url = public_url[:-1]
    
    callback_url = f"{public_url}/auth/callback/google"
    
    print_info(f"Client ID: {client_id[:20]}..." if client_id else "NOT SET")
    print_info(f"Expected callback URL: {callback_url}")
    print_info(f"Login URL: {public_url}/auth/login/google")
    
    # Warn about common mistakes
    if 'localhost' in public_url and '192.168.1.75' not in public_url:
        print_warning("You're using localhost but accessing via 192.168.1.75")
        print_info("  Set PUBLIC_BASE_URL=http://192.168.1.75:3000 for local testing")
    
    if not public_url.startswith('https://') and 'abhishekdoesstuff.com' in public_url:
        print_warning("Your domain uses Cloudflare - consider using https://")
    
    return True

async def step_5_test_full_flow():
    """Step 5: Test the full OAuth flow"""
    print_section("STEP 5: Testing Full OAuth Flow")
    
    try:
        from authlib.integrations.starlette_client import OAuth
        print_success("Authlib is installed")
    except ImportError:
        print_error("Authlib is not installed!")
        print_info("Install with: pip install authlib httpx")
        return False
    
    client_id = os.environ.get('OAUTH_GOOGLE_CLIENT_ID', '')
    client_secret = os.environ.get('OAUTH_GOOGLE_CLIENT_SECRET', '')
    
    if not client_id or not client_secret:
        print_error("Cannot test OAuth flow - credentials not set")
        return False
    
    try:
        oauth = OAuth()
        oauth.register(
            name='google',
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'}
        )
        
        print_success("OAuth client registered successfully")
        
        # Try to load server metadata (this is where it usually fails)
        try:
            # Access internal client to test connection
            client = oauth.google._client
            print_success("Authlib client initialized")
            
            # Try to fetch metadata
            import httpx
            test_client = httpx.AsyncClient(timeout=10)
            response = await test_client.get(
                'https://accounts.google.com/.well-known/openid-configuration'
            )
            await test_client.aclose()
            
            if response.status_code == 200:
                print_success("Can fetch Google OAuth metadata")
            else:
                print_error(f"Failed to fetch metadata: HTTP {response.status_code}")
                
        except AttributeError:
            print_warning("Cannot access internal client for testing")
        except Exception as e:
            print_error(f"Failed to fetch metadata: {type(e).__name__}: {e}")
            return False
            
    except Exception as e:
        print_error(f"Failed to register OAuth: {type(e).__name__}: {e}")
        return False
    
    return True

async def step_6_generate_report():
    """Step 6: Generate diagnostic report"""
    print_section("STEP 6: Diagnostic Summary")
    
    public_url = os.environ.get('PUBLIC_BASE_URL', 'http://localhost:3000')
    if public_url.endswith('/'):
        public_url = public_url[:-1]
    
    print_info("\nTo fix the issue, check these in order:")
    print_info("\n1. Google Cloud Console:")
    print_info(f"   - Add this EXACT redirect URI:")
    print_info(f"     {public_url}/auth/callback/google")
    print_info(f"   - No trailing slash, match protocol exactly")
    
    print_info("\n2. Environment Variables:")
    print_info("   - OAUTH_GOOGLE_CLIENT_ID: Must be set")
    print_info("   - OAUTH_GOOGLE_CLIENT_SECRET: Must be set")
    print_info("   - PUBLIC_BASE_URL: Must match your domain exactly")
    
    print_info("\n3. Network Issues (if connection fails):")
    print_info("   - Check if container can reach accounts.google.com:443")
    print_info("   - Verify iptables FORWARD rules allow outbound traffic")
    print_info("   - Check if LXC host has proper NAT configuration")
    
    print_info("\n4. Quick Test Commands:")
    print_info("   docker exec <container> bash -c '</dev/tcp/accounts.google.com/443 && echo OK'")
    print_info("   curl -I https://accounts.google.com/.well-known/openid-configuration")

async def main():
    print(f"{BLUE}Google OAuth Flow Debugger{RESET}")
    print(f"Started at: {datetime.now().isoformat()}")
    
    results = {
        'environment': False,
        'network': False,
        'https': False,
        'oauth_config': False,
        'full_flow': False
    }
    
    try:
        results['environment'], env_data = await step_1_check_environment()
        results['network'] = await step_2_check_network()
        results['https'] = await step_3_check_https()
        results['oauth_config'] = await step_4_check_oauth_config()
        results['full_flow'] = await step_5_test_full_flow()
        await step_6_generate_report()
        
        print_section("FINAL RESULTS")
        all_passed = all(results.values())
        
        for step, passed in results.items():
            status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
            print(f"  {step.replace('_', ' ').title()}: {status}")
        
        if all_passed:
            print(f"\n{GREEN}All checks passed! OAuth should work.{RESET}")
            return 0
        else:
            print(f"\n{RED}Some checks failed. Review the errors above.{RESET}")
            return 1
            
    except Exception as e:
        print(f"\n{RED}Debugger crashed: {type(e).__name__}: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
