#!/usr/bin/env python3
"""
Load test for Coaster Generator - Current Production Load Testing

This script supports two modes:
1. Phase 2 only (default): Bypasses BFL by seeding jobs in "review" state
2. Full flow (with --use-api-key): Tests complete flow with API key authentication

Usage examples:
  # Test Phase 2 only (no BFL, uses existing preview fixture):
  python tools/load_test_current.py \
      --base-url https://coaster.abhishekdoesstuff.com \
      --total-jobs 12 \
      --concurrency 4

  # Test with API key (full flow including BFL generation):
  python tools/load_test_current.py \
      --base-url https://coaster.abhishekdoesstuff.com \
      --total-jobs 5 \
      --concurrency 2 \
      --use-api-key YOUR_API_KEY \
      --test-image ./fixtures/test-image.jpg

  # Test rate limits:
  python tools/load_test_current.py \
      --base-url https://coaster.abhishekdoesstuff.com \
      --total-jobs 20 \
      --concurrency 10 \
      --mode rapid
"""

import argparse
import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

import aiohttp


@dataclass
class JobRunResult:
    job_id: str
    ok: bool
    status: str
    phase: str  # 'submit', 'confirm', 'poll', 'complete'
    duration_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    http_status: Optional[int] = None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def find_fixture_preview(jobs_dir: str, explicit_path: Optional[str]) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Preview fixture not found: {explicit_path}")
        return explicit_path

    candidates = sorted(
        glob(os.path.join(jobs_dir, "*_preview.png")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No preview fixture found in {jobs_dir}. "
            "Run one real generation first or pass --preview-image."
        )
    return candidates[0]


def seed_review_jobs(
    jobs_dir: str,
    preview_fixture: str,
    total_jobs: int,
    params: Dict[str, object],
    prefix: str,
) -> List[str]:
    os.makedirs(jobs_dir, exist_ok=True)
    created_ids: List[str] = []

    with open(preview_fixture, "rb") as f:
        preview_bytes = f.read()

    for i in range(total_jobs):
        job_id = f"{prefix}-{i:04d}-{uuid.uuid4().hex[:8]}"
        created_ids.append(job_id)

        preview_out = os.path.join(jobs_dir, f"{job_id}_preview.png")
        with open(preview_out, "wb") as f:
            f.write(preview_bytes)

        payload: Dict[str, object] = {
            "job_id": job_id,
            "status": "review",
            "progress": 50,
            "message": "Image generated! Please review and confirm to proceed.",
            "files": {},
            "created_at": datetime.now().isoformat(),
            "error": None,
            "preview_image_path": preview_out,
            "params": params,
            "api_key": None,
            "stamp_text": params.get("stamp_text", "Load Test"),
        }

        job_json = os.path.join(jobs_dir, f"{job_id}.json")
        with open(job_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return created_ids


async def submit_job_full_flow(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    test_image_path: str,
    params: Dict[str, object],
) -> Tuple[str, Optional[str]]:
    """Submit a job via full API with image upload."""
    try:
        with open(test_image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        payload = {
            "image_data": image_data,
            "filename": os.path.basename(test_image_path),
            "api_key": api_key,
            **params
        }

        async with session.post(
            f"{base_url}/api/process",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return None, f"HTTP {resp.status}: {text[:300]}"
            
            data = await resp.json()
            return data.get("job_id"), None
            
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


async def confirm_job(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
) -> Tuple[bool, Optional[str], float]:
    """Confirm a job and return success, error, and duration in ms."""
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{base_url}/api/confirm/{job_id}",
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            duration = (time.perf_counter() - t0) * 1000.0
            if resp.status == 200:
                return True, None, duration
            else:
                text = await resp.text()
                return False, f"HTTP {resp.status}: {text[:300]}", duration
    except Exception as e:
        duration = (time.perf_counter() - t0) * 1000.0
        return False, f"{type(e).__name__}: {e}", duration


async def poll_job_status(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    poll_interval: float,
    timeout_s: float,
) -> Tuple[str, Optional[str], float]:
    """Poll job until complete/failed/timeout. Returns status, error, duration."""
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    
    while time.perf_counter() < deadline:
        try:
            async with session.get(
                f"{base_url}/api/status/{job_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    await asyncio.sleep(poll_interval)
                    continue
                    
                data = await resp.json()
                status = data.get("status", "unknown")
                
                if status == "completed":
                    duration = time.perf_counter() - t0
                    return status, None, duration
                elif status == "failed":
                    duration = time.perf_counter() - t0
                    error = data.get("error") or data.get("message", "Unknown error")
                    return status, error, duration
                elif status == "review":
                    # Still waiting for confirmation
                    pass
                    
        except Exception:
            pass
            
        await asyncio.sleep(poll_interval)
    
    return "timeout", f"Timed out after {timeout_s:.1f}s", timeout_s


async def run_job_phase2(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    poll_interval: float,
    timeout_s: float,
) -> JobRunResult:
    """Run a Phase 2 job (already seeded in review state)."""
    durations = {}
    t_start = time.perf_counter()
    
    # Step 1: Confirm the job
    confirm_ok, confirm_err, confirm_ms = await confirm_job(session, base_url, job_id)
    durations['confirm'] = confirm_ms
    
    if not confirm_ok:
        return JobRunResult(
            job_id=job_id,
            ok=False,
            status="confirm_failed",
            phase="confirm",
            duration_ms=durations,
            error=confirm_err,
        )
    
    # Step 2: Poll for completion
    status, poll_err, poll_duration = await poll_job_status(
        session, base_url, job_id, poll_interval, timeout_s
    )
    durations['poll'] = poll_duration * 1000.0
    durations['total'] = (time.perf_counter() - t_start) * 1000.0
    
    if status == "completed":
        return JobRunResult(
            job_id=job_id,
            ok=True,
            status=status,
            phase="complete",
            duration_ms=durations,
        )
    else:
        return JobRunResult(
            job_id=job_id,
            ok=False,
            status=status,
            phase="poll",
            duration_ms=durations,
            error=poll_err,
        )


async def run_job_full_flow(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    test_image_path: str,
    params: Dict[str, object],
    poll_interval: float,
    timeout_s: float,
) -> JobRunResult:
    """Run a full flow job (submit with API key)."""
    durations = {}
    t_start = time.perf_counter()
    
    # Step 1: Submit job
    t0 = time.perf_counter()
    job_id, submit_err = await submit_job_full_flow(
        session, base_url, api_key, test_image_path, params
    )
    durations['submit'] = (time.perf_counter() - t0) * 1000.0
    
    if not job_id:
        return JobRunResult(
            job_id="unknown",
            ok=False,
            status="submit_failed",
            phase="submit",
            duration_ms=durations,
            error=submit_err,
        )
    
    # Step 2: Wait for review state (BFL generation)
    # For now, we skip polling for review and assume it's ready
    # In real scenario, you'd poll until status is "review"
    
    # Step 3: Confirm and poll for completion
    return await run_job_phase2(session, base_url, job_id, poll_interval, timeout_s)


async def run_load_phase2(
    base_url: str,
    job_ids: List[str],
    concurrency: int,
    poll_interval: float,
    timeout_s: float,
) -> List[JobRunResult]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def wrapped(job_id: str) -> JobRunResult:
            async with semaphore:
                return await run_job_phase2(
                    session, base_url, job_id, poll_interval, timeout_s
                )

        return await asyncio.gather(*[wrapped(job_id) for job_id in job_ids])


async def run_load_full_flow(
    base_url: str,
    api_key: str,
    test_image_path: str,
    params: Dict[str, object],
    total_jobs: int,
    concurrency: int,
    poll_interval: float,
    timeout_s: float,
) -> List[JobRunResult]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def wrapped(_: int) -> JobRunResult:
            async with semaphore:
                return await run_job_full_flow(
                    session, base_url, api_key, test_image_path, 
                    params, poll_interval, timeout_s
                )

        return await asyncio.gather(*[wrapped(i) for i in range(total_jobs)])


def cleanup_seeded_jobs(jobs_dir: str, job_ids: List[str]) -> None:
    for job_id in job_ids:
        for suffix in (".json", "_preview.png"):
            path = os.path.join(jobs_dir, f"{job_id}{suffix}")
            if os.path.exists(path):
                os.remove(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coaster Generator Load Test")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--temp-dir", default="./temp", help="TEMP_DIR used by API server")
    parser.add_argument("--preview-image", default=None, help="PNG fixture path (Phase 2 mode)")
    parser.add_argument("--test-image", default=None, help="Test image for full flow mode")
    parser.add_argument("--total-jobs", type=int, default=8, help="Total jobs to run")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent jobs")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Status poll interval (seconds)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-job timeout (seconds)")
    
    # API Key for full flow mode
    parser.add_argument("--use-api-key", default=None, help="API key for full flow testing (bypasses BFL rate limits)")
    
    # Job parameters
    parser.add_argument("--diameter", type=float, default=100.0)
    parser.add_argument("--thickness", type=float, default=5.0)
    parser.add_argument("--logo-depth", type=float, default=0.6)
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--flip-horizontal", dest="flip_horizontal", action="store_true", default=True)
    parser.add_argument("--no-flip-horizontal", dest="flip_horizontal", action="store_false")
    parser.add_argument("--top-rotate", type=int, default=0)
    parser.add_argument("--bottom-rotate", type=int, default=0)
    parser.add_argument("--stamp-text", default="Load Test")
    
    # Cleanup
    parser.add_argument("--keep-seeded-jobs", action="store_true", help="Do not delete seeded review jobs")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_dir = os.path.join(args.temp_dir, "jobs")
    
    # Build params dict
    params = {
        "diameter": args.diameter,
        "thickness": args.thickness,
        "logo_depth": args.logo_depth,
        "scale": args.scale,
        "flip_horizontal": args.flip_horizontal,
        "top_rotate": args.top_rotate,
        "bottom_rotate": args.bottom_rotate,
        "stamp_text": args.stamp_text,
    }
    
    base_url = args.base_url.rstrip("/")
    
    if args.use_api_key:
        # Full flow mode with API key
        if not args.test_image:
            print("Error: --test-image required for full flow mode")
            return
        if not os.path.exists(args.test_image):
            print(f"Error: Test image not found: {args.test_image}")
            return
            
        print("Full Flow Load Test (with API key)")
        print(f"- Base URL: {base_url}")
        print(f"- Total jobs: {args.total_jobs}")
        print(f"- Concurrency: {args.concurrency}")
        print(f"- Test image: {args.test_image}")
        
        started = time.perf_counter()
        results = asyncio.run(
            run_load_full_flow(
                base_url=base_url,
                api_key=args.use_api_key,
                test_image_path=args.test_image,
                params=params,
                total_jobs=args.total_jobs,
                concurrency=args.concurrency,
                poll_interval=args.poll_interval,
                timeout_s=args.timeout,
            )
        )
        
    else:
        # Phase 2 mode (seeded jobs)
        fixture = find_fixture_preview(jobs_dir, args.preview_image)
        prefix = f"loadtest-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        job_ids = seed_review_jobs(
            jobs_dir=jobs_dir,
            preview_fixture=fixture,
            total_jobs=args.total_jobs,
            params=params,
            prefix=prefix,
        )

        print("Phase 2 Load Test (BFL bypass)")
        print(f"- Base URL: {base_url}")
        print(f"- Jobs dir: {jobs_dir}")
        print(f"- Preview fixture: {fixture}")
        print(f"- Seeded jobs: {len(job_ids)}")
        print(f"- Concurrency: {args.concurrency}")

        started = time.perf_counter()
        results = asyncio.run(
            run_load_phase2(
                base_url=base_url,
                job_ids=job_ids,
                concurrency=args.concurrency,
                poll_interval=args.poll_interval,
                timeout_s=args.timeout,
            )
        )
        
        if not args.keep_seeded_jobs:
            cleanup_seeded_jobs(jobs_dir, job_ids)
    
    total_elapsed = time.perf_counter() - started

    # Analyze results
    completed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    
    # Collect metrics
    confirm_times = [r.duration_ms.get('confirm', 0) for r in results if 'confirm' in r.duration_ms]
    poll_times = [r.duration_ms.get('poll', 0) for r in results if 'poll' in r.duration_ms]
    total_times = [r.duration_ms.get('total', 0) for r in results if 'total' in r.duration_ms]

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total jobs: {len(results)}")
    print(f"Completed: {len(completed)} ({len(completed)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"Wall time: {total_elapsed:.2f}s")
    print(f"Throughput: {len(completed)/total_elapsed:.3f} jobs/s ({(len(completed)/total_elapsed)*60:.2f} jobs/min)")
    
    if confirm_times:
        print(f"\nConfirm latency (ms):")
        print(f"  p50: {percentile(confirm_times, 0.5):.1f}")
        print(f"  p95: {percentile(confirm_times, 0.95):.1f}")
        print(f"  p99: {percentile(confirm_times, 0.99):.1f}")
    
    if poll_times:
        print(f"\nProcessing time - poll to completion (s):")
        print(f"  p50: {percentile(poll_times, 0.5)/1000:.2f}")
        print(f"  p95: {percentile(poll_times, 0.95)/1000:.2f}")
        print(f"  p99: {percentile(poll_times, 0.99)/1000:.2f}")
    
    if total_times:
        print(f"\nEnd-to-end total time (s):")
        print(f"  p50: {percentile(total_times, 0.5)/1000:.2f}")
        print(f"  p95: {percentile(total_times, 0.95)/1000:.2f}")
        print(f"  p99: {percentile(total_times, 0.99)/1000:.2f}")

    if failed:
        print(f"\n{'='*60}")
        print("FAILURES")
        print("="*60)
        for r in failed[:10]:
            print(f"- {r.job_id}: {r.status} (phase: {r.phase})")
            if r.error:
                print(f"  Error: {r.error[:100]}")
        if len(failed) > 10:
            print(f"- ... and {len(failed)-10} more")


if __name__ == "__main__":
    main()
