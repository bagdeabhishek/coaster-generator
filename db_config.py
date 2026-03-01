"""
Database configuration for CoastGen.
Supports both SQLite (development) and PostgreSQL (production/Supabase).
"""

import os

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Check if we're using PostgreSQL (Supabase) or SQLite
USE_POSTGRES = DATABASE_URL.startswith("postgres") or DATABASE_URL.startswith("postgresql")

if USE_POSTGRES:
    # PostgreSQL/Supebase configuration
    print("Using PostgreSQL database (Supabase)")
else:
    # SQLite configuration (fallback for local dev)
    print("Using SQLite database (local development)")
