"""
Authentication and quota persistence layer.
Supports both SQLite (development) and PostgreSQL/Supabase (production).
"""

import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
import threading
from contextlib import contextmanager

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_POSTGRES = DATABASE_URL.startswith("postgres") or DATABASE_URL.startswith("postgresql")

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    
    # Connection pool for PostgreSQL
    db_pool = None
    db_lock = threading.Lock()
    
    def get_connection():
        """Get a connection from the pool."""
        global db_pool
        if db_pool is None:
            with db_lock:
                if db_pool is None:
                    db_pool = SimpleConnectionPool(
                        1, 20,
                        DATABASE_URL,
                        cursor_factory=RealDictCursor
                    )
        return db_pool.getconn()
    
    def release_connection(conn):
        """Release connection back to pool."""
        if db_pool:
            db_pool.putconn(conn)
    
    # PostgreSQL uses %s for placeholders
    PARAM_STYLE = "%s"
    
else:
    # SQLite for local development
    import sqlite3
    
    DB_PATH = os.path.join(os.environ.get("TEMP_DIR", "./temp"), "app_data.db")
    db_lock = threading.Lock()
    
    def get_connection():
        """Get a thread-local database connection."""
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn
    
    def release_connection(conn):
        """Close SQLite connection."""
        conn.close()
    
    # SQLite uses ? for placeholders
    PARAM_STYLE = "?"


def get_db():
    """Context manager for database connections."""
    class DBContext:
        def __enter__(self):
            self.conn = get_connection()
            return self.conn
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            release_connection(self.conn)
            return False
    
    return DBContext()


def init_db():
    """Initialize the database schema."""
    with get_db() as conn:
        cursor = conn.cursor()
        lock_id = 438217

        # Serialize schema init across gunicorn workers for PostgreSQL.
        if USE_POSTGRES:
            cursor.execute("SELECT pg_advisory_lock(%s)", (lock_id,))

        try:
            # Users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    email_verified INT DEFAULT 0,
                    name TEXT,
                    avatar_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # OAuth identities table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_identities (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    provider_user_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    access_token TEXT,
                    refresh_token TEXT,
                    token_expires_at INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(provider, provider_user_id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

            # Usage events table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    principal_type TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_principal
                ON usage_events(principal_type, principal_id, bucket, created_at)
                """
            )

            # Subscriptions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    user_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    customer_id TEXT,
                    subscription_id TEXT,
                    status TEXT NOT NULL,
                    period_start TEXT,
                    period_end TEXT,
                    plan_code TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sub_status
                ON subscriptions(status, period_end)
                """
            )

            # Processed webhooks table (idempotency)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_webhooks (
                    webhook_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    received_at TEXT NOT NULL
                )
                """
            )

        finally:
            if USE_POSTGRES:
                cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))

        print(f"Database initialized: {'PostgreSQL (Supabase)' if USE_POSTGRES else 'SQLite'}")


def _format_query(query: str) -> str:
    """Convert ? placeholders to %s for PostgreSQL if needed."""
    if USE_POSTGRES:
        return query.replace("?", "%s")
    return query


def hash_user_id(user_id: str) -> str:
    """Generate a secure hash for user ID."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:32]


def create_user(email: str, name: Optional[str] = None, avatar_url: Optional[str] = None) -> tuple[str, str]:
    """
    Create a new user.
    Returns: (user_id, email)
    Raises: IntegrityError if email already exists
    """
    user_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("""
            INSERT INTO users (id, email, email_verified, name, avatar_url, created_at, updated_at)
            VALUES (?, ?, 0, ?, ?, ?, ?)
        """)
        cursor.execute(query, (user_id, email.lower(), name, avatar_url, now, now))
        return user_id, email.lower()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("SELECT * FROM users WHERE email = ?")
        cursor.execute(query, (email.lower(),))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("SELECT * FROM users WHERE id = ?")
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_oauth(provider: str, provider_user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by OAuth provider and user ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("""
            SELECT u.* FROM users u
            JOIN oauth_identities o ON u.id = o.user_id
            WHERE o.provider = ? AND o.provider_user_id = ?
        """)
        cursor.execute(query, (provider, provider_user_id))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def link_oauth_identity(
    user_id: str,
    provider: str,
    provider_user_id: str,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    token_expires_at: Optional[int] = None
) -> bool:
    """
    Link an OAuth identity to a user.
    Returns True if successful, False if identity already exists.
    """
    now = datetime.utcnow().isoformat()
    identity_id = uuid.uuid4().hex
    
    with get_db() as conn:
        cursor = conn.cursor()

        query = _format_query(
            """
            INSERT INTO oauth_identities
            (id, provider, provider_user_id, user_id, access_token, refresh_token,
             token_expires_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, provider_user_id) DO UPDATE SET
                user_id = excluded.user_id,
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                token_expires_at = excluded.token_expires_at,
                updated_at = excluded.updated_at
            """
        )
        cursor.execute(
            query,
            (
                identity_id,
                provider,
                provider_user_id,
                user_id,
                access_token,
                refresh_token,
                token_expires_at,
                now,
                now,
            ),
        )
        
        return True


def record_usage_event(
    job_id: str,
    principal_type: str,
    principal_id: str,
    bucket: str
) -> str:
    """Record a usage event for quota tracking."""
    event_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("""
            INSERT INTO usage_events (id, job_id, principal_type, principal_id, bucket, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """)
        cursor.execute(query, (event_id, job_id, principal_type, principal_id, bucket, now))
        return event_id


def _advisory_lock_id(lock_key: str) -> int:
    """Build a stable 63-bit advisory lock id from an arbitrary string."""
    digest = hashlib.sha256(lock_key.encode()).hexdigest()
    return int(digest[:15], 16)


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
    conn = get_connection()
    try:
        cursor = conn.cursor()

        if USE_POSTGRES:
            lock_key = f"quota:{principal_type}:{principal_id}:{bucket}:{since or 'all'}"
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (_advisory_lock_id(lock_key),))
        else:
            cursor.execute("BEGIN IMMEDIATE")

        if since:
            count_query = _format_query(
                """
                SELECT COUNT(*) as count FROM usage_events
                WHERE principal_type = ? AND principal_id = ? AND bucket = ? AND created_at >= ?
                """
            )
            cursor.execute(count_query, (principal_type, principal_id, bucket, since))
        else:
            count_query = _format_query(
                """
                SELECT COUNT(*) as count FROM usage_events
                WHERE principal_type = ? AND principal_id = ? AND bucket = ?
                """
            )
            cursor.execute(count_query, (principal_type, principal_id, bucket))

        row = cursor.fetchone()
        used = row["count"] if row else 0
        if used >= limit:
            conn.rollback()
            return None

        insert_query = _format_query(
            """
            INSERT INTO usage_events (id, job_id, principal_type, principal_id, bucket, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """
        )
        cursor.execute(insert_query, (event_id, job_id, principal_type, principal_id, bucket, now))
        conn.commit()
        return event_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def count_usage_in_bucket(
    principal_type: str,
    principal_id: str,
    bucket: str,
    since: str
) -> int:
    """Count usage events in a bucket since a given time."""
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("""
            SELECT COUNT(*) as count FROM usage_events
            WHERE principal_type = ? AND principal_id = ? AND bucket = ? AND created_at >= ?
        """)
        cursor.execute(query, (principal_type, principal_id, bucket, since))
        row = cursor.fetchone()
        return row['count'] if row else 0


def get_usage_count(
    principal_type: str,
    principal_id: str,
    bucket: str,
    since: Optional[str] = None,
) -> int:
    """Get usage count for a principal/bucket, optionally since timestamp."""
    with get_db() as conn:
        cursor = conn.cursor()
        if since:
            query = _format_query(
                """
                SELECT COUNT(*) as count FROM usage_events
                WHERE principal_type = ? AND principal_id = ? AND bucket = ? AND created_at >= ?
                """
            )
            cursor.execute(query, (principal_type, principal_id, bucket, since))
        else:
            query = _format_query(
                """
                SELECT COUNT(*) as count FROM usage_events
                WHERE principal_type = ? AND principal_id = ? AND bucket = ?
                """
            )
            cursor.execute(query, (principal_type, principal_id, bucket))

        row = cursor.fetchone()
        return row['count'] if row else 0


def get_paid_usage_count(
    user_id: str,
    since_timestamp: Optional[str] = None,
) -> int:
    """Get paid usage count, optionally for current billing period."""
    with get_db() as conn:
        cursor = conn.cursor()
        if since_timestamp:
            query = _format_query(
                """
                SELECT COUNT(*) as count FROM usage_events
                WHERE principal_type = 'user' AND principal_id = ?
                AND bucket = 'paid' AND created_at >= ?
                """
            )
            cursor.execute(query, (user_id, since_timestamp))
        else:
            query = _format_query(
                """
                SELECT COUNT(*) as count FROM usage_events
                WHERE principal_type = 'user' AND principal_id = ? AND bucket = 'paid'
                """
            )
            cursor.execute(query, (user_id,))

        row = cursor.fetchone()
        return row['count'] if row else 0


def set_subscription(
    user_id: str,
    provider: str,
    customer_id: Optional[str],
    subscription_id: Optional[str],
    status: str,
    period_start: Optional[str],
    period_end: Optional[str],
    plan_code: Optional[str]
) -> None:
    """Set or update a user's subscription."""
    now = datetime.utcnow().isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("""
            INSERT INTO subscriptions
            (user_id, provider, customer_id, subscription_id, status, period_start, period_end, plan_code, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                provider = excluded.provider,
                customer_id = excluded.customer_id,
                subscription_id = excluded.subscription_id,
                status = excluded.status,
                period_start = excluded.period_start,
                period_end = excluded.period_end,
                plan_code = excluded.plan_code,
                updated_at = excluded.updated_at
        """)
        cursor.execute(query, (
            user_id, provider, customer_id, subscription_id, status,
            period_start, period_end, plan_code, now
        ))


def get_subscription(user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user's subscription."""
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("SELECT * FROM subscriptions WHERE user_id = ?")
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def is_subscription_active(user_id: str) -> bool:
    """Check if user has an active subscription."""
    from datetime import datetime as dt

    sub = get_subscription(user_id)
    if not sub:
        return False

    status = (sub.get('status') or '').lower()
    if status not in ('active', 'trialing'):
        return False

    period_end = sub.get('period_end')
    if period_end:
        try:
            end_date = dt.fromisoformat(str(period_end).replace('Z', '+00:00'))
            if end_date.tzinfo is not None:
                if dt.now(end_date.tzinfo) > end_date:
                    return False
            else:
                if dt.utcnow() > end_date:
                    return False
        except (ValueError, TypeError):
            pass

    return True


def record_webhook(webhook_id: str, event_type: str) -> bool:
    """
    Record a processed webhook for idempotency.
    Returns True if newly recorded, False if already processed.
    """
    now = datetime.utcnow().isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("""
            INSERT INTO processed_webhooks (webhook_id, event_type, received_at)
            VALUES (?, ?, ?)
            ON CONFLICT(webhook_id) DO NOTHING
        """)
        cursor.execute(query, (webhook_id, event_type, now))
        return cursor.rowcount > 0


def is_webhook_processed(webhook_id: str) -> bool:
    """Check if a webhook has already been processed."""
    with get_db() as conn:
        cursor = conn.cursor()
        query = _format_query("SELECT 1 FROM processed_webhooks WHERE webhook_id = ?")
        cursor.execute(query, (webhook_id,))
        return cursor.fetchone() is not None


@contextmanager
def webhook_processing_lock(webhook_id: str):
    """Cross-worker lock for webhook processing, strongest on PostgreSQL."""
    if not USE_POSTGRES:
        with db_lock:
            yield
        return

    conn = get_connection()
    lock_id = _advisory_lock_id(f"webhook:{webhook_id}")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
        yield
    finally:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
        finally:
            release_connection(conn)


def clear_all_quotas():
    """Clear all usage events (for testing)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM usage_events")
        print("All quota usage cleared")


# Aliases for backward compatibility
get_user_by_oauth_id = get_user_by_oauth
link_identity = link_oauth_identity
