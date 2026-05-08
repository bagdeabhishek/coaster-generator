"""
Database configuration for CoastGen.
SQLite-only mode.
"""

import os

DB_PATH = os.path.join(os.environ.get("TEMP_DIR", "./temp"), "app_data.db")
USE_POSTGRES = False

print(f"Using SQLite database: {DB_PATH}")
