"""
SQLite-only authentication and quota persistence layer.
Backwards-compatible module surface used by main.py and quota_service.py.
"""

import uuid
import threading
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

from auth_quota_store import (
    get_connection,
    init_db,
    hash_user_id,
    create_user,
    get_user_by_email,
    get_user_by_id,
    get_user_by_oauth,
    link_oauth_identity,
    record_usage_event,
    get_usage_count,
    get_paid_usage_count,
    set_subscription,
    get_subscription,
    is_subscription_active,
    record_webhook,
    is_webhook_processed,
    clear_all_quotas,
)

# Kept for compatibility with existing imports.
USE_POSTGRES = False

# Module-level lock for critical sections across workers/threads.
db_lock = threading.Lock()


def reserve_usage_event_atomic(
    job_id: str,
    principal_type: str,
    principal_id: str,
    bucket: str,
    limit: int,
    since: Optional[str] = None,
) -> Optional[str]:
    """
    Atomically reserve one usage event if under limit.

    Returns event_id when reserved, or None if quota is exhausted.
    """
    if limit <= 0:
        return None

    event_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")

            if since:
                cursor.execute(
                    """
                    SELECT COUNT(*) as count FROM usage_events
                    WHERE principal_type = ? AND principal_id = ? AND bucket = ? AND created_at >= ?
                    """,
                    (principal_type, principal_id, bucket, since),
                )
            else:
                cursor.execute(
                    """
                    SELECT COUNT(*) as count FROM usage_events
                    WHERE principal_type = ? AND principal_id = ? AND bucket = ?
                    """,
                    (principal_type, principal_id, bucket),
                )

            row = cursor.fetchone()
            current_count = int(row["count"] if row else 0)

            if current_count >= limit:
                conn.rollback()
                return None

            cursor.execute(
                """
                INSERT INTO usage_events (id, job_id, principal_type, principal_id, bucket, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, job_id, principal_type, principal_id, bucket, now),
            )
            conn.commit()
            return event_id

        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


@contextmanager
def webhook_processing_lock(_webhook_id: str):
    """Serialize webhook processing with a local lock."""
    with db_lock:
        yield


# Aliases for backward compatibility
get_user_by_oauth_id = get_user_by_oauth
link_identity = link_oauth_identity
