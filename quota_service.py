"""
Quota decision logic for coaster generation.
Determines allowed usage based on user tier and current limits.
"""

import os
import hashlib
import asyncio
from typing import Optional, Dict, Any, Tuple

try:
    from db_store import (
        get_usage_count,
        get_paid_usage_count,
        is_subscription_active,
        record_usage_event,
        get_subscription,
        reserve_usage_event_atomic,
    )
except ImportError:
    from auth_quota_store import (
        get_usage_count,
        get_paid_usage_count,
        is_subscription_active,
        record_usage_event,
        get_subscription,
    )
    reserve_usage_event_atomic = None


# Quota limits (env-configurable)
ANON_FREE_LIMIT = int(os.environ.get("ANON_FREE_LIMIT", "1"))
LOGIN_BONUS_LIMIT = int(os.environ.get("LOGIN_BONUS_LIMIT", "2"))
PAID_MONTHLY_LIMIT = int(os.environ.get("PAID_MONTHLY_LIMIT", "200"))

# Rate limiter lock (to prevent race conditions)
_quota_lock = asyncio.Lock()


def hash_fingerprint(fingerprint: str) -> str:
    """Hash a device fingerprint."""
    return hashlib.sha256(str(fingerprint).encode()).hexdigest()[:32]


def get_anon_principal_ip(fallback: str) -> str:
    """
    Get anonymous principal ID from fingerprint or IP.
    """
    fp = fallback
    
    if not fp or fp == "unknown":
        return "ip:unknown"
    
    return f"anon:{hash_fingerprint(fp)}"


async def check_quota(
    fingerprint: Optional[str],
    user_id: Optional[str],
    ip_header: Optional[str] = None
) -> Tuple[bool, str, int, Dict[str, Any]]:
    """
    Check if user has remaining quota for generation.
    
    Args:
        fingerprint: X-Device-Fingerprint header value
        user_id: User ID from session (None if anonymous)
        ip_header: X-Forwarded-For or similar IP header
    
    Returns:
        (allowed, message, retry_after_seconds, usage_info)
    """
    async with _quota_lock:
        # Determine principal
        if user_id:
            principal_type = "user"
            principal_id = user_id
            is_anon = False
        else:
            # Determine anon principal from fingerprint or IP
            if fingerprint:
                fp = fingerprint
            elif ip_header:
                fp = ip_header
            else:
                fp = "unknown"
            
            principal_type = "anon"
            principal_id = get_anon_principal_ip(fp)
            is_anon = True
        
        # Fetch subscription info for paid users
        subscription_active = False
        paid_period_start = None
        paid_used = 0
        paid_remaining = PAID_MONTHLY_LIMIT
        
        if user_id:
            subscription_active = is_subscription_active(user_id)
            if subscription_active:
                sub = get_subscription(user_id)
                if sub and sub.get('period_start'):
                    paid_period_start = sub['period_start']
                paid_used = get_paid_usage_count(user_id, paid_period_start)
                paid_remaining = max(0, PAID_MONTHLY_LIMIT - paid_used)
        
        # Fetch usage counts
        anon_used = get_usage_count(principal_type, principal_id, "anon_free")
        login_used = get_usage_count(principal_type, principal_id, "login_bonus")
        
        # Build usage info
        usage_info = {
            "tier": "unknown",
            "remaining_anon": max(0, ANON_FREE_LIMIT - anon_used),
            "remaining_login_bonus": max(0, LOGIN_BONUS_LIMIT - login_used),
            "paid_limit": PAID_MONTHLY_LIMIT,
            "paid_used": paid_used,
            "paid_remaining": paid_remaining,
            "remaining_total": 0,
            "next_action": None,
            "message": "",
            "bucket": None,
        }
        
        # Decision logic
        bucket = None
        allowed = False
        message = ""
        retry_after = 0
        
        if user_id and subscription_active and paid_remaining > 0:
            # Paid tier
            allowed = True
            bucket = "paid"
            usage_info["remaining_total"] = paid_remaining
            usage_info["tier"] = "paid"
            message = f"You have {paid_remaining} generations remaining this month."
            
        elif is_anon and anon_used < ANON_FREE_LIMIT:
            # Anonymous free tier
            allowed = True
            bucket = "anon_free"
            usage_info["remaining_total"] = ANON_FREE_LIMIT - anon_used
            usage_info["tier"] = "free"
            message = f"You have {ANON_FREE_LIMIT - anon_used} free generation(s) remaining."
            
        elif user_id and login_used < LOGIN_BONUS_LIMIT:
            # Login bonus tier
            allowed = True
            bucket = "login_bonus"
            usage_info["remaining_total"] = LOGIN_BONUS_LIMIT - login_used
            usage_info["tier"] = "free"
            message = f"You have {LOGIN_BONUS_LIMIT - login_used} bonus generation(s) remaining."
            
        else:
            # Denied
            allowed = False
            usage_info["remaining_total"] = 0
            usage_info["tier"] = "exhausted"
            
            if user_id:
                usage_info["next_action"] = "upgrade"
                usage_info["message"] = (
                    "You've used all your free generations. "
                    "Upgrade to a paid plan for unlimited generation capability."
                )
            else:
                usage_info["next_action"] = "login"
                usage_info["message"] = (
                    "You've used your free generation. "
                    "Sign in to unlock 2 more free generations."
                )
                
        usage_info["bucket"] = bucket
        
        return allowed, message, retry_after, usage_info


async def consume_quota(
    job_id: str,
    fingerprint: Optional[str],
    user_id: Optional[str],
    bucket: str
) -> Optional[str]:
    """
    Consume one quota unit for a generation.
    Records the usage event atomically.
    
    Returns event_id on success, None if consumption failed.
    """
    async with _quota_lock:
        # Determine principal
        if user_id:
            principal_type = "user"
            principal_id = user_id
        else:
            if fingerprint:
                fp = fingerprint
            else:
                fp = "unknown"
            principal_type = "anon"
            principal_id = get_anon_principal_ip(fp)
        
        try:
            event_id = record_usage_event(job_id, principal_type, principal_id, bucket)
            return event_id
        except Exception as e:
            print(f"Failed to record usage event: {e}")
            return None


async def check_and_consume_quota_atomic(
    job_id: str,
    fingerprint: Optional[str],
    user_id: Optional[str],
    ip_header: Optional[str] = None,
) -> Tuple[bool, str, int, Dict[str, Any], Optional[str]]:
    """
    Atomically evaluate quota and reserve one usage event.

    Returns (allowed, message, retry_after_seconds, usage_info, event_id)
    """
    max_attempts = 3
    for _ in range(max_attempts):
        async with _quota_lock:
            # Determine principal context
            is_anon = not bool(user_id)
            if user_id:
                principal_type = "user"
                principal_id = user_id
            else:
                fp = fingerprint or ip_header or "unknown"
                principal_type = "anon"
                principal_id = get_anon_principal_ip(fp)

            # Determine subscription state for user
            subscription_active = False
            paid_period_start = None
            if user_id:
                subscription_active = is_subscription_active(user_id)
                if subscription_active:
                    sub = get_subscription(user_id)
                    if sub and sub.get("period_start"):
                        paid_period_start = sub["period_start"]

            # Current usage snapshot
            anon_used = get_usage_count(principal_type, principal_id, "anon_free")
            login_used = get_usage_count(principal_type, principal_id, "login_bonus")
            paid_used = get_paid_usage_count(user_id, paid_period_start) if user_id and subscription_active else 0

            usage_info = {
                "tier": "unknown",
                "remaining_anon": max(0, ANON_FREE_LIMIT - anon_used),
                "remaining_login_bonus": max(0, LOGIN_BONUS_LIMIT - login_used),
                "paid_limit": PAID_MONTHLY_LIMIT,
                "paid_used": paid_used,
                "paid_remaining": max(0, PAID_MONTHLY_LIMIT - paid_used),
                "remaining_total": 0,
                "next_action": None,
                "message": "",
                "bucket": None,
            }

            bucket = None
            limit = 0
            since = None
            message = ""

            if user_id and subscription_active and paid_used < PAID_MONTHLY_LIMIT:
                bucket = "paid"
                limit = PAID_MONTHLY_LIMIT
                since = paid_period_start
                usage_info["tier"] = "paid"
                usage_info["remaining_total"] = max(0, PAID_MONTHLY_LIMIT - paid_used)
                message = f"You have {usage_info['remaining_total']} generations remaining this month."
            elif is_anon and anon_used < ANON_FREE_LIMIT:
                bucket = "anon_free"
                limit = ANON_FREE_LIMIT
                usage_info["tier"] = "free"
                usage_info["remaining_total"] = max(0, ANON_FREE_LIMIT - anon_used)
                message = f"You have {usage_info['remaining_total']} free generation(s) remaining."
            elif user_id and login_used < LOGIN_BONUS_LIMIT:
                bucket = "login_bonus"
                limit = LOGIN_BONUS_LIMIT
                usage_info["tier"] = "free"
                usage_info["remaining_total"] = max(0, LOGIN_BONUS_LIMIT - login_used)
                message = f"You have {usage_info['remaining_total']} bonus generation(s) remaining."
            else:
                usage_info["tier"] = "exhausted"
                usage_info["remaining_total"] = 0
                if user_id:
                    usage_info["next_action"] = "upgrade"
                    usage_info["message"] = (
                        "You've used all your free generations. "
                        "Upgrade to a paid plan for unlimited generation capability."
                    )
                else:
                    usage_info["next_action"] = "login"
                    usage_info["message"] = (
                        "You've used your free generation. "
                        "Sign in to unlock 2 more free generations."
                    )
                return False, "", 0, usage_info, None

            usage_info["bucket"] = bucket

            # Authoritative reservation in shared DB when available.
            if reserve_usage_event_atomic:
                event_id = reserve_usage_event_atomic(
                    job_id=job_id,
                    principal_type=principal_type,
                    principal_id=principal_id,
                    bucket=bucket,
                    limit=limit,
                    since=since,
                )
            else:
                event_id = record_usage_event(job_id, principal_type, principal_id, bucket)

            if not event_id:
                # Lost a race or hit limit exactly at reservation time; retry snapshot.
                continue

            # We consumed one generation now.
            usage_info["remaining_total"] = max(0, usage_info["remaining_total"] - 1)
            if bucket == "anon_free":
                usage_info["remaining_anon"] = max(0, usage_info["remaining_anon"] - 1)
            elif bucket == "login_bonus":
                usage_info["remaining_login_bonus"] = max(0, usage_info["remaining_login_bonus"] - 1)
            elif bucket == "paid":
                usage_info["paid_used"] += 1
                usage_info["paid_remaining"] = max(0, usage_info["paid_remaining"] - 1)

            return True, message, 0, usage_info, event_id

    # Could not reserve after retries under contention.
    return False, "", 0, {
        "tier": "exhausted",
        "remaining_anon": 0,
        "remaining_login_bonus": 0,
        "paid_limit": PAID_MONTHLY_LIMIT,
        "paid_used": 0,
        "paid_remaining": 0,
        "remaining_total": 0,
        "next_action": "retry",
        "message": "Usage is busy right now. Please retry.",
        "bucket": None,
    }, None
