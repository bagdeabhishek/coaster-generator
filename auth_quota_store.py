"""
Authentication and quota persistence layer using SQLite.
Thread-safe with lock-based access.
"""

import sqlite3
import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
import threading


DB_PATH = os.path.join(os.environ.get("TEMP_DIR", "./temp"), "app_data.db")
db_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable WAL mode for concurrent reads/writes
    conn.execute("PRAGMA synchronous=NORMAL;") 
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    email_verified INT DEFAULT 0,
                    name TEXT,
                    avatar_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # OAuth identities table
            cursor.execute("""
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
            """)
            
            # Usage events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    principal_type TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_principal 
                ON usage_events(principal_type, principal_id, bucket, created_at)
            """)
            
            # Subscriptions table
            cursor.execute("""
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
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sub_status 
                ON subscriptions(status, period_end)
            """)
            
            # Processed webhooks table (idempotency)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_webhooks (
                    webhook_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    received_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            print(f"Database initialized at: {DB_PATH}")
        except sqlite3.Error as e:
            print(f"Database init error: {e}")
            raise


def hash_user_id(user_id: str) -> str:
    """Generate a secure hash for user ID."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:32]


def create_user(email: str, name: Optional[str] = None, avatar_url: Optional[str] = None) -> tuple[str, str]:
    """
    Create a new user.
    Returns: (user_id, email)
    Raises: sqlite3.IntegrityError if email already exists
    """
    user_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (id, email, email_verified, name, avatar_url, created_at, updated_at)
                VALUES (?, ?, 0, ?, ?, ?, ?)
            """, (user_id, email.lower(), name, avatar_url, now, now))
            conn.commit()
            return user_id, email.lower()
        finally:
            conn.close()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()


def get_user_by_oauth(provider: str, provider_user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by OAuth provider and user ID."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.* FROM users u
                JOIN oauth_identities o ON u.id = o.user_id
                WHERE o.provider = ? AND o.provider_user_id = ?
            """, (provider, provider_user_id))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()


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
    
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
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
            """, (
                identity_id,
                provider,
                provider_user_id,
                user_id,
                access_token,
                refresh_token,
                token_expires_at,
                now,
                now,
            ))
            conn.commit()
            
            return True
        finally:
            conn.close()


def record_usage_event(
    job_id: str,
    principal_type: str,
    principal_id: str,
    bucket: str
) -> str:
    """
    Record a usage event.
    Returns created event ID.
    """
    event_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat()
    
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO usage_events (id, job_id, principal_type, principal_id, bucket, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_id, job_id, principal_type, principal_id, bucket, created_at))
            conn.commit()
            return event_id
        finally:
            conn.close()


def get_usage_count(
    principal_type: str,
    principal_id: str,
    bucket: str,
    since: Optional[str] = None
) -> int:
    """
    Get usage count for a principal/bucket.
    If since is provided, count from that time.
    """
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            
            if since:
                cursor.execute("""
                    SELECT COUNT(*) FROM usage_events
                    WHERE principal_type = ? AND principal_id = ? AND bucket = ? AND created_at >= ?
                """, (principal_type, principal_id, bucket, since))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM usage_events
                    WHERE principal_type = ? AND principal_id = ? AND bucket = ?
                """, (principal_type, principal_id, bucket))
            
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


def get_paid_usage_count(
    user_id: str,
    since_timestamp: Optional[str] = None
) -> int:
    """Get paid usage count, optionally for current billing period."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            
            if since_timestamp:
                cursor.execute("""
                    SELECT COUNT(*) FROM usage_events ue
                    WHERE ue.principal_type = 'user'
                    AND ue.principal_id = ?
                    AND ue.bucket = 'paid'
                    AND ue.created_at >= ?
                """, (user_id, since_timestamp))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM usage_events
                    WHERE principal_type = 'user' AND principal_id = ? AND bucket = 'paid'
                """, (user_id,))
            
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


def set_subscription(
    user_id: str,
    provider: str,
    customer_id: Optional[str],
    subscription_id: Optional[str],
    status: str,
    period_start: Optional[str],
    period_end: Optional[str],
    plan_code: Optional[str]
) -> bool:
    """
    Set or update a user's subscription.
    Returns True if successful.
    """
    updated_at = datetime.utcnow().isoformat()
    
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO subscriptions 
                (user_id, provider, customer_id, subscription_id, status, 
                 period_start, period_end, plan_code, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    provider = excluded.provider,
                    customer_id = COALESCE(excluded.customer_id, subscriptions.customer_id),
                    subscription_id = COALESCE(excluded.subscription_id, subscriptions.subscription_id),
                    status = excluded.status,
                    period_start = COALESCE(excluded.period_start, subscriptions.period_start),
                    period_end = excluded.period_end,
                    plan_code = COALESCE(excluded.plan_code, subscriptions.plan_code),
                    updated_at = excluded.updated_at
                WHERE subscriptions.user_id = ?
            """, (
                user_id, provider, customer_id, subscription_id, status,
                period_start, period_end, plan_code, updated_at, user_id
            ))
            conn.commit()
            return True
        finally:
            conn.close()


def get_subscription(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user's current subscription."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM subscriptions WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()


def is_subscription_active(user_id: str) -> bool:
    """Check if user has an active subscription."""
    from datetime import datetime as dt
    
    sub = get_subscription(user_id)
    if not sub:
        return False
    
    status = sub.get('status', '').lower()
    if status not in ('active', 'trialing'):
        return False
    
    period_end = sub.get('period_end')
    if period_end:
        try:
            end_date = dt.fromisoformat(period_end.replace('Z', '+00:00'))
            if dt.now(end_date.tzinfo) > end_date:
                return False
        except (ValueError, TypeError):
            pass
    
    return True


def record_webhook(webhook_id: str, event_type: str):
    """Record a processed webhook for idempotency."""
    received_at = datetime.utcnow().isoformat()
    
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO processed_webhooks (webhook_id, event_type, received_at)
                VALUES (?, ?, ?)
            """, (webhook_id, event_type, received_at))
            conn.commit()
        finally:
            conn.close()


def is_webhook_processed(webhook_id: str) -> bool:
    """Check if webhook was already processed."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM processed_webhooks WHERE webhook_id = ?", (webhook_id,))
            return cursor.fetchone() is not None
        finally:
            conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialization complete")

def clear_all_quotas():
    """Clear all quota usage (useful for dev mode testing)."""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM usage_events")
            conn.commit()
            return True
        finally:
            conn.close()
