#!/usr/bin/env python3
"""
Migrate data from SQLite to PostgreSQL/Supabase.
Run this once when switching to Supabase.
"""

import os
import sys

# Ensure DATABASE_URL is set to PostgreSQL
if not os.environ.get("DATABASE_URL", "").startswith("postgres"):
    print("Error: DATABASE_URL must be set to PostgreSQL connection string")
    print("Example: export DATABASE_URL='postgresql://user:pass@host:5432/dbname'")
    sys.exit(1)

# Import both database modules
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

SQLITE_PATH = os.path.join(os.environ.get("TEMP_DIR", "./temp"), "app_data.db")
POSTGRES_URL = os.environ.get("DATABASE_URL")

def migrate_table(table_name, columns):
    """Migrate a single table."""
    print(f"Migrating {table_name}...")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    # Get all data from SQLite
    sqlite_cursor.execute(f"SELECT * FROM {table_name}")
    rows = sqlite_cursor.fetchall()
    
    if not rows:
        print(f"  No data in {table_name}")
        sqlite_conn.close()
        return
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(POSTGRES_URL)
    pg_cursor = pg_conn.cursor()
    
    # Insert data
    placeholders = ",".join(["%s"] * len(columns))
    query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    
    count = 0
    for row in rows:
        try:
            pg_cursor.execute(query, tuple(row))
            count += 1
        except Exception as e:
            print(f"  Error migrating row: {e}")
    
    pg_conn.commit()
    pg_conn.close()
    sqlite_conn.close()
    
    print(f"  Migrated {count} rows")

def main():
    print("Starting migration from SQLite to PostgreSQL...")
    print(f"SQLite: {SQLITE_PATH}")
    print(f"PostgreSQL: {POSTGRES_URL[:20]}...")
    print()
    
    # Initialize PostgreSQL schema first
    print("Initializing PostgreSQL schema...")
    from db_store import init_db
    init_db()
    print()
    
    # Migrate tables
    migrate_table("users", ["id", "email", "email_verified", "name", "avatar_url", "created_at", "updated_at"])
    migrate_table("oauth_identities", ["id", "provider", "provider_user_id", "user_id", "access_token", "refresh_token", "token_expires_at", "created_at", "updated_at"])
    migrate_table("subscriptions", ["user_id", "provider", "customer_id", "subscription_id", "status", "period_start", "period_end", "plan_code", "updated_at"])
    migrate_table("usage_events", ["id", "job_id", "principal_type", "principal_id", "bucket", "created_at"])
    migrate_table("processed_webhooks", ["webhook_id", "event_type", "received_at"])
    
    print()
    print("Migration complete!")

if __name__ == "__main__":
    main()
