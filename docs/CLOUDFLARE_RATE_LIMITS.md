# Cloudflare Rate Limit Rules (Hobby-Safe Defaults)

These rules protect public app APIs from abuse while keeping normal usage smooth.

## Before You Start

- Keep app-side ownership and quota checks enabled.
- Configure these rules in Cloudflare **WAF -> Rate limiting rules**.
- Start in **simulate/log** mode for 24h, then switch to **managed challenge** (or block for obvious abuse).

## Rule 1: Process Endpoint (Strict)

- **Expression**
  - `(http.request.method eq "POST" and starts_with(http.request.uri.path, "/api/process"))`
- **Threshold**
  - `5 requests per 10 minutes` per IP
- **Action**
  - `Managed Challenge`

## Rule 2: Confirm/Retry Endpoints (Medium)

- **Expression**
  - `(http.request.method eq "POST" and (starts_with(http.request.uri.path, "/api/confirm/") or starts_with(http.request.uri.path, "/api/retry/") or starts_with(http.request.uri.path, "/api/regenerate/")))`
- **Threshold**
  - `20 requests per 10 minutes` per IP
- **Action**
  - `Managed Challenge`

## Rule 3: Status Polling (Higher, but capped)

- **Expression**
  - `(http.request.method eq "GET" and starts_with(http.request.uri.path, "/api/status/"))`
- **Threshold**
  - `120 requests per 10 minutes` per IP
- **Action**
  - `Managed Challenge`

## Rule 4: Download Endpoints (Medium)

- **Expression**
  - `(http.request.method eq "GET" and starts_with(http.request.uri.path, "/api/download/"))`
- **Threshold**
  - `30 requests per 10 minutes` per IP
- **Action**
  - `Managed Challenge`

## Optional Rule 5: API Burst Shield

- **Expression**
  - `starts_with(http.request.uri.path, "/api/")`
- **Threshold**
  - `600 requests per 10 minutes` per IP
- **Action**
  - `Managed Challenge`

## Suggested Exclusions

- Skip or relax for your own trusted IPs if you do frequent load testing.
- Keep `/health` and `/ready` unrestricted only if needed by monitoring.

## Rollout Checklist

1. Add all rules in simulate mode.
2. Run normal user flow and one load test.
3. Review Cloudflare logs for false positives.
4. Switch to Managed Challenge.
5. Move severe abusers to Block if needed.
