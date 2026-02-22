#!/usr/bin/env python3
"""
Load test for Phase 2 only (vectorization + STL/3MF generation).

This script bypasses BFL completely by pre-seeding jobs directly in the
"review" state, then calling /api/confirm/{job_id} concurrently.

What it measures:
- Confirm request success/failure
- End-to-end time from confirm -> completed/failed
- Throughput and latency percentiles

Usage example:
  python tools/load_test_phase2.py \
      --base-url http://localhost:8000 \
      --total-jobs 12 \
      --concurrency 4
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional

import aiohttp


@dataclass
class JobRunResult:
    job_id: str
    ok: bool
    status: str
    confirm_ms: float
    total_s: float
    error: Optional[str] = None


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
    diameter: float,
    thickness: float,
    logo_depth: float,
    scale: float,
    flip_horizontal: bool,
    top_rotate: int,
    bottom_rotate: int,
    stamp_text: str,
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
            "params": {
                "diameter": diameter,
                "thickness": thickness,
                "logo_depth": logo_depth,
                "scale": scale,
                "flip_horizontal": flip_horizontal,
                "top_rotate": top_rotate,
                "bottom_rotate": bottom_rotate,
            },
            "api_key": None,
            "stamp_text": stamp_text,
        }

        job_json = os.path.join(jobs_dir, f"{job_id}.json")
        with open(job_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return created_ids


async def run_one_job(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    poll_interval: float,
    timeout_s: float,
) -> JobRunResult:
    t0 = time.perf_counter()

    # Trigger phase 2 only.
    confirm_start = time.perf_counter()
    try:
        async with session.post(f"{base_url}/api/confirm/{job_id}") as resp:
            confirm_ms = (time.perf_counter() - confirm_start) * 1000.0
            if resp.status != 200:
                text = await resp.text()
                return JobRunResult(
                    job_id=job_id,
                    ok=False,
                    status="confirm_http_error",
                    confirm_ms=confirm_ms,
                    total_s=time.perf_counter() - t0,
                    error=f"HTTP {resp.status}: {text[:300]}",
                )
    except Exception as exc:
        return JobRunResult(
            job_id=job_id,
            ok=False,
            status="confirm_exception",
            confirm_ms=(time.perf_counter() - confirm_start) * 1000.0,
            total_s=time.perf_counter() - t0,
            error=str(exc),
        )

    deadline = t0 + timeout_s
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{base_url}/api/status/{job_id}") as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    return JobRunResult(
                        job_id=job_id,
                        ok=False,
                        status="status_http_error",
                        confirm_ms=confirm_ms,
                        total_s=time.perf_counter() - t0,
                        error=f"HTTP {resp.status}: {txt[:300]}",
                    )
                data = await resp.json()
        except Exception as exc:
            return JobRunResult(
                job_id=job_id,
                ok=False,
                status="status_exception",
                confirm_ms=confirm_ms,
                total_s=time.perf_counter() - t0,
                error=str(exc),
            )

        status = data.get("status", "unknown")
        if status == "completed":
            return JobRunResult(
                job_id=job_id,
                ok=True,
                status=status,
                confirm_ms=confirm_ms,
                total_s=time.perf_counter() - t0,
            )
        if status == "failed":
            return JobRunResult(
                job_id=job_id,
                ok=False,
                status=status,
                confirm_ms=confirm_ms,
                total_s=time.perf_counter() - t0,
                error=data.get("error") or data.get("message"),
            )

        await asyncio.sleep(poll_interval)

    return JobRunResult(
        job_id=job_id,
        ok=False,
        status="timeout",
        confirm_ms=confirm_ms,
        total_s=time.perf_counter() - t0,
        error=f"Timed out after {timeout_s:.1f}s",
    )


async def run_load(
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
                return await run_one_job(session, base_url, job_id, poll_interval, timeout_s)

        return await asyncio.gather(*[wrapped(job_id) for job_id in job_ids])


def cleanup_seeded_jobs(jobs_dir: str, job_ids: List[str]) -> None:
    for job_id in job_ids:
        for suffix in (".json", "_preview.png"):
            path = os.path.join(jobs_dir, f"{job_id}{suffix}")
            if os.path.exists(path):
                os.remove(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 (no BFL) load test")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--temp-dir", default="./temp", help="TEMP_DIR used by API server")
    parser.add_argument("--preview-image", default=None, help="PNG fixture path (optional)")
    parser.add_argument("--total-jobs", type=int, default=8, help="Total jobs to run")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent confirms")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Status poll interval seconds")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-job timeout seconds")
    parser.add_argument("--diameter", type=float, default=100.0)
    parser.add_argument("--thickness", type=float, default=5.0)
    parser.add_argument("--logo-depth", type=float, default=0.6)
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--flip-horizontal", dest="flip_horizontal", action="store_true")
    parser.add_argument("--no-flip-horizontal", dest="flip_horizontal", action="store_false")
    parser.set_defaults(flip_horizontal=True)
    parser.add_argument("--top-rotate", type=int, default=0)
    parser.add_argument("--bottom-rotate", type=int, default=0)
    parser.add_argument("--stamp-text", default="Load Test")
    parser.add_argument("--keep-seeded-jobs", action="store_true", help="Do not delete seeded review jobs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_dir = os.path.join(args.temp_dir, "jobs")
    fixture = find_fixture_preview(jobs_dir, args.preview_image)

    prefix = f"loadtest-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    job_ids = seed_review_jobs(
        jobs_dir=jobs_dir,
        preview_fixture=fixture,
        total_jobs=args.total_jobs,
        diameter=args.diameter,
        thickness=args.thickness,
        logo_depth=args.logo_depth,
        scale=args.scale,
        flip_horizontal=args.flip_horizontal,
        top_rotate=args.top_rotate,
        bottom_rotate=args.bottom_rotate,
        stamp_text=args.stamp_text,
        prefix=prefix,
    )

    print("Phase-2 load test (BFL bypass)")
    print(f"- Base URL: {args.base_url}")
    print(f"- Jobs dir: {jobs_dir}")
    print(f"- Preview fixture: {fixture}")
    print(f"- Seeded jobs: {len(job_ids)}")
    print(f"- Concurrency: {args.concurrency}")

    started = time.perf_counter()
    results = asyncio.run(
        run_load(
            base_url=args.base_url.rstrip("/"),
            job_ids=job_ids,
            concurrency=args.concurrency,
            poll_interval=args.poll_interval,
            timeout_s=args.timeout,
        )
    )
    total_elapsed = time.perf_counter() - started

    if not args.keep_seeded_jobs:
        cleanup_seeded_jobs(jobs_dir, job_ids)

    completed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    latencies = [r.total_s for r in completed]
    confirm_latencies = [r.confirm_ms for r in results]

    print("\nResults")
    print(f"- Total jobs: {len(results)}")
    print(f"- Completed: {len(completed)}")
    print(f"- Failed: {len(failed)}")
    print(f"- Wall time: {total_elapsed:.2f}s")
    print(f"- Throughput: {len(completed)/total_elapsed:.3f} jobs/s ({(len(completed)/total_elapsed)*60:.2f} jobs/min)")
    print(f"- Confirm latency p50/p95: {percentile(confirm_latencies, 0.5):.1f}ms / {percentile(confirm_latencies, 0.95):.1f}ms")

    if latencies:
        print("- End-to-end completion latency (confirm -> completed)")
        print(
            f"  p50={percentile(latencies, 0.50):.2f}s, "
            f"p95={percentile(latencies, 0.95):.2f}s, "
            f"p99={percentile(latencies, 0.99):.2f}s"
        )

    if failed:
        print("\nFailures")
        for r in failed[:10]:
            print(f"- {r.job_id}: {r.status} ({r.error})")
        if len(failed) > 10:
            print(f"- ... and {len(failed)-10} more")


if __name__ == "__main__":
    main()
