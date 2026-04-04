"""
CoastGen - 3D Custom Coaster Generator
FastAPI backend with async job processing for converting images to 3D printable coasters.
"""

import os
import io
import base64
import uuid
import asyncio
import logging
import sys
import json
import fcntl
import hashlib
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import mediapipe as mp
except ImportError:
    mp = None

_FACE_CROP_UNAVAILABLE_LOGGED = False

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Request, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from pydantic import BaseModel, Field
import concurrent.futures
from tools.coaster_gen import CoasterGenerator

# Auth and billing
try:
    from db_store import (
        init_db,
        create_user,
        get_user_by_email,
        get_user_by_id,
        get_user_by_oauth,
        link_oauth_identity,
        set_subscription,
        record_webhook,
        is_webhook_processed,
        webhook_processing_lock,
        clear_all_quotas,
        USE_POSTGRES
    )
    print(f"Using {'PostgreSQL' if USE_POSTGRES else 'SQLite'} database")
except ImportError:
    # Fallback to old module for backward compatibility
    from auth_quota_store import (
        init_db,
        create_user,
        get_user_by_email,
        get_user_by_id,
        get_user_by_oauth,
        link_oauth_identity,
        set_subscription,
        record_webhook,
        is_webhook_processed,
        clear_all_quotas,
    )
    from contextlib import nullcontext
    webhook_processing_lock = lambda _webhook_id: nullcontext()
    USE_POSTGRES = False
    print("Using legacy SQLite database")
from quota_service import check_quota, check_and_consume_quota_atomic

# Security imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# OAuth
from authlib.integrations.starlette_client import OAuth

# Billing
from dodopayments import DodoPayments

# Debug flag - read from environment variable (default to False for production)
DEBUG_NO_CLEANUP = os.environ.get("DEBUG_NO_CLEANUP", "false").lower() in ("true", "1", "yes")

# Configure logging based on environment
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if DEBUG_NO_CLEANUP:
    logger.warning("="*60)
    logger.warning("DEBUG MODE: Cleanup is DISABLED - temp files will be preserved!")
    logger.warning("="*60)

logger.info("="*60)
logger.info("Starting CoastGen")
logger.info("="*60)

# Configuration
# Use absolute paths to avoid issues with vtracer (Rust library)
TEMP_DIR = os.path.abspath("./temp")
BFL_API_URL = "https://api.bfl.ai/v1"
MAX_POLLING_ATTEMPTS = 60
POLLING_INTERVAL = 2

# Security Configuration
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production").lower()
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_COOLDOWN_HOURS = int(os.environ.get("RATE_LIMIT_COOLDOWN_HOURS", "168"))  # 1 week default
ALLOW_BYPASS_WITH_API_KEY = os.environ.get("ALLOW_BYPASS_WITH_API_KEY", "true").lower() == "true"
LEGACY_RATE_LIMIT_ENABLED = os.environ.get("LEGACY_RATE_LIMIT_ENABLED", "false").lower() == "true"
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
MAX_RAW_FILE_SIZE_MB = int(os.environ.get("MAX_RAW_FILE_SIZE_MB", "30"))
PREPROCESS_DEFAULT_FACE_CROP = os.environ.get("PREPROCESS_DEFAULT_FACE_CROP", "true").lower() == "true"
PREPROCESS_DEFAULT_AUTO_DOWNSIZE = os.environ.get("PREPROCESS_DEFAULT_AUTO_DOWNSIZE", "true").lower() == "true"
PREPROCESS_DEFAULT_FACE_PADDING = float(os.environ.get("PREPROCESS_DEFAULT_FACE_PADDING", "0.35"))
FACE_DETECT_MIN_CONFIDENCE = float(os.environ.get("FACE_DETECT_MIN_CONFIDENCE", "0.5"))
FACE_DETECT_MAX_EDGE = int(os.environ.get("FACE_DETECT_MAX_EDGE", "1280"))
PREPROCESS_MAX_EDGE = int(os.environ.get("PREPROCESS_MAX_EDGE", "2048"))
PREPROCESS_JPEG_QUALITY = int(os.environ.get("PREPROCESS_JPEG_QUALITY", "90"))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
TRUSTED_PROXY_HOSTS = [
    host.strip()
    for host in os.environ.get("TRUSTED_PROXY_HOSTS", "127.0.0.1,localhost").split(",")
    if host.strip()
]

logger.info(f"TEMP_DIR: {TEMP_DIR}")
logger.info(f"BFL_API_URL: {BFL_API_URL}")
logger.info(f"Rate limiting: {RATE_LIMIT_ENABLED} ({RATE_LIMIT_COOLDOWN_HOURS}h cooldown)")
logger.info(f"Bypass with API key: {ALLOW_BYPASS_WITH_API_KEY}")
logger.info(f"Legacy weekly limiter enabled: {LEGACY_RATE_LIMIT_ENABLED}")
logger.info(f"Image upload limits: raw={MAX_RAW_FILE_SIZE_MB}MB, preprocessed_target={MAX_FILE_SIZE_MB}MB")
logger.info(
    "Preprocess defaults: face_crop=%s auto_downsize=%s face_padding=%.2f",
    PREPROCESS_DEFAULT_FACE_CROP,
    PREPROCESS_DEFAULT_AUTO_DOWNSIZE,
    PREPROCESS_DEFAULT_FACE_PADDING,
)

# Auth Configuration
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
if not SESSION_SECRET:
    if ENVIRONMENT == "production":
        logger.error("SESSION_SECRET is not set! Generating a random one (not secure for production).")
    else:
        logger.warning("SESSION_SECRET is not set; generating a temporary development secret.")
    SESSION_SECRET = uuid.uuid4().hex

OAUTH_GOOGLE_CLIENT_ID = os.environ.get("OAUTH_GOOGLE_CLIENT_ID", "")
OAUTH_GOOGLE_CLIENT_SECRET = os.environ.get("OAUTH_GOOGLE_CLIENT_SECRET", "")

# Billing Configuration
DODO_API_KEY = os.environ.get("DODO_PAYMENTS_API_KEY", "")
DODO_ENV = os.environ.get("DODO_PAYMENTS_ENVIRONMENT", "test_mode")
DODO_WEBHOOK_KEY = os.environ.get("DODO_PAYMENTS_WEBHOOK_KEY", "")
DODO_SUBSCRIPTION_PRODUCT_ID = os.environ.get("DODO_SUBSCRIPTION_PRODUCT_ID", "")
GOOGLE_ANALYTICS_ID = os.environ.get("GOOGLE_ANALYTICS_ID", "").strip()
SUPPORT_CONTACT_EMAIL = os.environ.get("SUPPORT_CONTACT_EMAIL", "admin@abhishekdoesstuff.com").strip()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:3000")
if PUBLIC_BASE_URL.endswith("/"):
    PUBLIC_BASE_URL = PUBLIC_BASE_URL[:-1]

logger.info(f"Session enabled: {bool(SESSION_SECRET)}")
logger.info(f"Google OAuth enabled: {bool(OAUTH_GOOGLE_CLIENT_ID)}")
logger.info(f"Dodo Payments enabled: {bool(DODO_API_KEY)}")
logger.info(f"Google Analytics enabled: {bool(GOOGLE_ANALYTICS_ID)}")


def is_quota_related_error(message: str) -> bool:
    """Return True if the message looks like quota/rate-limit exhaustion."""
    text = (message or "").lower()
    markers = [
        "quota",
        "rate limit",
        "too many requests",
        "insufficient credit",
        "insufficient credits",
        "credit balance",
        "usage limit",
        "billing hard limit",
        "payment required",
        "429",
        "resource_exhausted",
    ]
    return any(marker in text for marker in markers)


def with_support_contact(message: str) -> str:
    """Append support contact guidance once."""
    base = (message or "").strip()
    if not base:
        base = "Quota exceeded."
    if SUPPORT_CONTACT_EMAIL.lower() in base.lower():
        return base
    return f"{base} Contact {SUPPORT_CONTACT_EMAIL} for quota increase help."

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Temp directory ensured: {os.path.abspath(TEMP_DIR)}")

# Job storage directory (shared across all workers)
JOBS_DIR = os.path.join(TEMP_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
logger.info(f"Jobs directory: {JOBS_DIR}")


@dataclass
class Job:
    """Represents a coaster generation job."""
    job_id: str
    status: str = "pending"  # pending, processing_flatten, review, processing_vectorize, processing_3d, completed, failed
    progress: int = 0
    message: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    preview_image_path: Optional[str] = None  # Path to saved preview image
    source_image_path: Optional[str] = None  # Path to original uploaded image for regeneration
    params: Optional['ProcessRequest'] = None  # Store params for confirmation step
    api_key: Optional[str] = None  # User-provided BFL API key (optional)
    uses_own_api_key: bool = False  # Whether user supplied their own BFL key
    stamp_text: str = "Abhishek Does Stuff"  # Text to display on coaster stamp
    owner_user_id: Optional[str] = None  # Authenticated owner id
    owner_anon_id: Optional[str] = None  # Anonymous owner fingerprint hash
    
    def to_dict(self) -> dict:
        """Convert job to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "files": self.files,
            "created_at": self.created_at.isoformat(),
            "error": self.error,
            "preview_image_path": self.preview_image_path,
            "source_image_path": self.source_image_path,
            "params": self.params.dict() if self.params else None,
            "api_key": self.api_key,
            "uses_own_api_key": self.uses_own_api_key,
            "stamp_text": self.stamp_text,
            "owner_user_id": self.owner_user_id,
            "owner_anon_id": self.owner_anon_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        """Create job from dictionary."""
        job = cls(
            job_id=data["job_id"],
            status=data["status"],
            progress=data["progress"],
            message=data["message"],
            files=data.get("files", {}),
            error=data.get("error"),
            preview_image_path=data.get("preview_image_path"),
            source_image_path=data.get("source_image_path"),
            api_key=data.get("api_key"),
            uses_own_api_key=bool(data.get("uses_own_api_key", False)),
            stamp_text=data.get("stamp_text", "Abhishek Does Stuff"),
            owner_user_id=data.get("owner_user_id"),
            owner_anon_id=data.get("owner_anon_id"),
        )
        job.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("params"):
            job.params = ProcessRequest(**data["params"])
        return job


class JobStore:
    """File-based job storage shared across all workers."""
    
    @staticmethod
    def _get_job_path(job_id: str) -> str:
        """Get file path for job data."""
        return os.path.join(JOBS_DIR, f"{job_id}.json")
    
    @staticmethod
    def _get_preview_path(job_id: str) -> str:
        """Get file path for preview image."""
        return os.path.join(JOBS_DIR, f"{job_id}_preview.png")

    @staticmethod
    def _get_source_path(job_id: str) -> str:
        """Get file path for original source image."""
        return os.path.join(JOBS_DIR, f"{job_id}_source.png")
    
    @classmethod
    def save_job(cls, job: Job) -> None:
        """Save job to disk atomically with file locking."""
        job_path = cls._get_job_path(job.job_id)
        temp_path = f"{job_path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(temp_path, 'w') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(job.to_dict(), f)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            os.replace(temp_path, job_path)
            logger.debug(f"Job {job.job_id} saved to disk")
        except Exception as e:
            logger.error(f"Failed to save job {job.job_id}: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    
    @classmethod
    def get_job(cls, job_id: str) -> Optional[Job]:
        """Load job from disk."""
        job_path = cls._get_job_path(job_id)
        if not os.path.exists(job_path):
            return None
        
        try:
            with open(job_path, 'r') as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            job = Job.from_dict(data)
            logger.debug(f"Job {job_id} loaded from disk")
            return job
        except Exception as e:
            logger.error(f"Failed to load job {job_id}: {e}")
            return None
    
    @classmethod
    def save_preview_image(cls, job_id: str, image_bytes: bytes) -> str:
        """Save preview image to disk."""
        preview_path = cls._get_preview_path(job_id)
        try:
            with open(preview_path, 'wb') as f:
                f.write(image_bytes)
            logger.debug(f"Preview image saved for job {job_id}: {preview_path}")
            return preview_path
        except Exception as e:
            logger.error(f"Failed to save preview image for job {job_id}: {e}")
            raise
    
    @classmethod
    def get_preview_image(cls, job_id: str) -> Optional[bytes]:
        """Load preview image from disk."""
        preview_path = cls._get_preview_path(job_id)
        if not os.path.exists(preview_path):
            return None
        
        try:
            with open(preview_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load preview image for job {job_id}: {e}")
            return None

    @classmethod
    def save_source_image(cls, job_id: str, image_bytes: bytes) -> str:
        """Save original uploaded source image to disk."""
        source_path = cls._get_source_path(job_id)
        try:
            with open(source_path, 'wb') as f:
                f.write(image_bytes)
            logger.debug(f"Source image saved for job {job_id}: {source_path}")
            return source_path
        except Exception as e:
            logger.error(f"Failed to save source image for job {job_id}: {e}")
            raise

    @classmethod
    def get_source_image(cls, job_id: str) -> Optional[bytes]:
        """Load original source image from disk."""
        source_path = cls._get_source_path(job_id)
        if not os.path.exists(source_path):
            return None

        try:
            with open(source_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load source image for job {job_id}: {e}")
            return None
    
    @classmethod
    def cleanup_old_jobs(cls, max_age_hours: int = 24) -> None:
        """Clean up job files and generated 3MF/STL files older than specified hours."""
        try:
            now = datetime.now()
            
            # Clean up JSON and preview files in jobs dir
            for filename in os.listdir(JOBS_DIR):
                if filename.endswith('.json') or filename.endswith('_preview.png') or filename.endswith('_source.png'):
                    filepath = os.path.join(JOBS_DIR, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    age_hours = (now - file_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old job file: {filename}")
                        
            # Clean up generated 3MF, STL, SVG, PNG files in temp dir
            # (skip database files)
            for filename in os.listdir(TEMP_DIR):
                if filename.endswith(('.3mf', '.stl', '.svg', '.png')) and not filename.endswith('_preview.png'):
                    filepath = os.path.join(TEMP_DIR, filename)
                    # Exclude directories
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        age_hours = (now - file_time).total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            os.remove(filepath)
                            logger.info(f"Cleaned up old generated file: {filename}")
        except Exception as e:
            logger.error(f"Error during job cleanup: {e}")


# Backward compatibility - use JobStore methods
jobs = JobStore()  # This provides a dict-like interface


class FileBasedRateLimiter:
    """File-based rate limiter that persists across restarts."""
    
    def __init__(self, storage_path: str, cooldown_hours: int = 168):
        self.storage_path = storage_path
        self.cooldown_hours = cooldown_hours
        self.lock = asyncio.Lock()
        # Ensure directory exists
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    async def is_allowed(self, fingerprint: str) -> tuple[bool, str, int]:
        """
        Check if user is allowed to make a request.
        Returns: (allowed: bool, message: str, retry_after_seconds: int)
        """
        async with self.lock:
            # Load existing data
            data = await self._load_data()
            
            now = time.time()
            cooldown_seconds = self.cooldown_hours * 3600
            
            if fingerprint in data:
                last_request = data[fingerprint]
                time_since = now - last_request
                
                if time_since < cooldown_seconds:
                    retry_after = int(cooldown_seconds - time_since)
                    hours_left = retry_after / 3600
                    return (
                        False,
                        f"You've already created a coaster! Come back in {hours_left:.1f} hours.",
                        retry_after
                    )
            
            # Record this request
            data[fingerprint] = now
            await self._save_data(data)
            
            return True, "OK", 0
    
    async def _load_data(self) -> dict:
        """Load rate limit data from file."""
        if not os.path.exists(self.storage_path):
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load rate limit data: {e}")
            return {}
    
    async def _save_data(self, data: dict):
        """Save rate limit data to file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            logger.error(f"Failed to save rate limit data: {e}")
    
    async def cleanup_old(self):
        """Remove entries older than cooldown period."""
        async with self.lock:
            data = await self._load_data()
            now = time.time()
            cutoff = now - (self.cooldown_hours * 3600)
            
            # Filter out old entries
            data = {k: v for k, v in data.items() if v > cutoff}
            await self._save_data(data)
            
            removed = len([k for k, v in data.items() if v <= cutoff])
            if removed > 0:
                logger.info(f"Cleaned up {removed} old rate limit entries")


class ProcessRequest(BaseModel):
    """Request model for coaster generation."""
    diameter: float = Field(default=100.0, ge=40.0, le=180.0)
    thickness: float = Field(default=5.0, ge=2.0, le=12.0)
    logo_depth: float = Field(default=0.6, ge=0.2, le=2.0)
    top_logo_height: float = Field(default=0.0, ge=0.0, le=2.0)
    scale: float = Field(default=0.85, ge=0.3, le=0.98)
    flip_horizontal: bool = True
    top_rotate: int = Field(default=0, ge=0, le=360)
    bottom_rotate: int = Field(default=0, ge=0, le=360)
    nozzle_diameter: float = Field(default=0.4, ge=0.2, le=1.2)
    auto_thicken: bool = True


class StatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: int
    message: str
    download_urls: Optional[Dict[str, str]] = None
    error: Optional[str] = None


# Custom rate limit key function (combines IP + device fingerprint)
def get_rate_limit_key(request: Request) -> str:
    """Generate rate limit key from IP and device fingerprint."""
    # Get client IP, handling proxies
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    elif request.client:
        client_ip = request.client.host
    else:
        client_ip = "unknown"
    
    # Prefer server-issued anonymous session id to avoid spoofable headers
    anon_session_id = request.session.get("anon_id")
    if anon_session_id:
        device_fp = f"session:{anon_session_id}"
    else:
        # Fallback to client-provided fingerprint header
        device_fp = request.headers.get("X-Device-Fingerprint", "")
    
    # Combine for unique key
    if device_fp:
        key = f"{client_ip}:{device_fp}"
    else:
        key = client_ip
    
    # Hash to create consistent length key
    return hashlib.sha256(key.encode()).hexdigest()[:24]

# Initialize file-based rate limiter (persists across restarts)
rate_limiter = None
if RATE_LIMIT_ENABLED and LEGACY_RATE_LIMIT_ENABLED:
    rate_limiter = FileBasedRateLimiter(
        storage_path=os.path.join(TEMP_DIR, "rate_limits.json"),
        cooldown_hours=RATE_LIMIT_COOLDOWN_HOURS
    )
    logger.info(f"Rate limiter initialized with {RATE_LIMIT_COOLDOWN_HOURS}h cooldown")

# Initialize database
init_db()

# Initialize Local Processing Pool
MAX_LOCAL_WORKERS = 4
process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_LOCAL_WORKERS)
active_local_tasks = 0
tasks_lock = asyncio.Lock()

# Initialize Modal (optional fallback)
MODAL_APP_NAME = "coaster-generator"
USE_MODAL_FALLBACK = os.environ.get("USE_MODAL_FALLBACK", "true").lower() == "true"
if USE_MODAL_FALLBACK:
    try:
        import modal
        # We don't deploy here, just prepare to lookup the function later
        logger.info("Modal fallback enabled for burst traffic")
    except ImportError:
        logger.warning("Modal library not installed. Disabling burst to cloud.")
        USE_MODAL_FALLBACK = False


def _lookup_modal_function(app_name: str, function_name: str):
    """Resolve Modal function across client API versions."""
    if not USE_MODAL_FALLBACK:
        raise RuntimeError("Modal fallback is disabled")

    # Older SDKs
    lookup = getattr(modal.Function, "lookup", None)
    if callable(lookup):
        return lookup(app_name, function_name)

    # Newer SDKs
    from_name = getattr(modal.Function, "from_name", None)
    if callable(from_name):
        return from_name(app_name, function_name)

    raise RuntimeError("Unsupported Modal SDK: no Function.lookup/from_name")

# Initialize FastAPI app with optimized settings
app = FastAPI(
    title="CoastGen",
    version="1.0.0",
    docs_url="/docs" if DEBUG_NO_CLEANUP else None,
    redoc_url="/redoc" if DEBUG_NO_CLEANUP else None,
)

# Add CORS middleware
is_dev = ENVIRONMENT == "development"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if is_dev else [origin.strip() for origin in ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["*"] if is_dev else ["GET", "POST"],
    allow_headers=["*"] if is_dev else ["Content-Type", "X-Device-Fingerprint"],
)

# Add Proxy headers middleware for correct IP/Scheme behind reverse proxies (like Coolify/Traefik)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=TRUSTED_PROXY_HOSTS)
logger.info(f"Trusted proxy hosts: {TRUSTED_PROXY_HOSTS}")

# Add session middleware
if SESSION_SECRET:
    app.add_middleware(
        SessionMiddleware,
        secret_key=SESSION_SECRET,
        session_cookie="coaster_session",
        max_age=30 * 24 * 60 * 60,  # 30 days
        same_site="lax",
        https_only=True if ENVIRONMENT == "production" else False,
    )
    logger.info("Session middleware enabled")

# Initialize OAuth
oauth = OAuth()
enabled_providers = []
if OAUTH_GOOGLE_CLIENT_ID and OAUTH_GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=OAUTH_GOOGLE_CLIENT_ID,
        client_secret=OAUTH_GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"}
    )
    enabled_providers.append("google")
    logger.info("Google OAuth registered")

# Initialize Dodo Payments client
dodo_client = None
if DODO_API_KEY:
    dodo_client = DodoPayments(
        bearer_token=DODO_API_KEY,
        environment=DODO_ENV,
        webhook_key=DODO_WEBHOOK_KEY,
    )
    logger.info("Dodo Payments client initialized")

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    csp_header = (
        "default-src 'self'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "script-src 'self' 'unsafe-inline' cdn.tailwindcss.com cdnjs.cloudflare.com unpkg.com threejs.org www.googletagmanager.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "img-src 'self' blob: data: https://lh3.googleusercontent.com https://www.google-analytics.com; "
        "connect-src 'self' https://api.bfl.ai https://auth.bfl.ai https://www.google-analytics.com https://region1.google-analytics.com https://stats.g.doubleclick.net; "
        "font-src 'self' https://fonts.gstatic.com; "
    )
    if not is_dev:
        csp_header += "upgrade-insecure-requests; "
        
    response.headers["Content-Security-Policy"] = csp_header
    return response

# Setup templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "frontend", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend", "static")), name="static")


# ============= AUTH ROUTES =============

@app.get("/api/auth/providers")
async def get_auth_providers():
    """Return list of available OAuth providers."""
    return {"providers": enabled_providers}


@app.get("/api/auth/me")
async def get_auth_me(request: Request):
    """Get current user info from session."""
    user_id = request.session.get("user_id")
    if not user_id:
        return {"authenticated": False}
    
    user = get_user_by_id(user_id)
    if not user:
        request.session.pop("user_id", None)
        return {"authenticated": False}
    
    return {
        "authenticated": True,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url")
        }
    }


@app.post("/api/auth/logout")
async def logout(request: Request):
    """Clear user session."""
    request.session.pop("user_id", None)
    return {"success": True}

@app.post("/api/dev/clear-quotas")
async def dev_clear_quotas():
    """DEV ONLY: Clear all quota usage."""
    if ENVIRONMENT != "development":
        raise HTTPException(status_code=403, detail="Not available in production")
    clear_all_quotas()
    return {"success": True, "message": "All quotas cleared"}


# ============= OAUTH LOGIN ROUTES =============

@app.get("/auth/login/google")
async def login_google(request: Request):
    """Redirect to Google OAuth."""
    if "google" not in enabled_providers:
        raise HTTPException(status_code=404, detail="Google login is not configured")

    try:
        return await oauth.google.authorize_redirect(
            request,
            f"{PUBLIC_BASE_URL}/auth/callback/google",
        )
    except Exception as e:
        logger.error(f"OAuth redirect FAILED: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OAuth error: {type(e).__name__}")


@app.get("/auth/test-oauth-connection")
async def test_oauth_connection():
    """Test OAuth network connectivity directly."""
    if ENVIRONMENT != "development":
        raise HTTPException(status_code=404, detail="Not found")

    import httpx
    import asyncio
    
    results = {}
    
    # Test 1: Fresh httpx client
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://accounts.google.com/.well-known/openid-configuration")
            results["fresh_httpx"] = f"SUCCESS: {r.status_code}"
    except Exception as e:
        results["fresh_httpx"] = f"FAILED: {type(e).__name__}: {e}"
    
    # Test 2: Authlib's internal client
    if "google" in enabled_providers:
        try:
            # Access Authlib's internal client
            from authlib.integrations.httpx_client import AsyncOAuth2Client
            client = oauth.google._client
            r = await client.get("https://accounts.google.com/.well-known/openid-configuration", withhold_token=True)
            results["authlib_client"] = f"SUCCESS: {r.status_code}"
        except Exception as e:
            results["authlib_client"] = f"FAILED: {type(e).__name__}: {e}"
    else:
        results["authlib_client"] = "SKIPPED: OAuth not enabled"
    
    # Test 3: DNS resolution timing
    import socket
    import time
    start = time.time()
    try:
        info = socket.getaddrinfo("accounts.google.com", 443)
        elapsed = time.time() - start
        results["dns_timing"] = f"SUCCESS: {len(info)} results in {elapsed:.3f}s, first: {info[0][4]}"
    except Exception as e:
        results["dns_timing"] = f"FAILED: {type(e).__name__}: {e}"
    
    # Test 4: Raw TCP connection
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("accounts.google.com", 443),
            timeout=5
        )
        writer.close()
        await writer.wait_closed()
        results["raw_tcp"] = "SUCCESS: Connected"
    except Exception as e:
        results["raw_tcp"] = f"FAILED: {type(e).__name__}: {e}"
    
    return results


@app.get("/auth/callback/google")
async def callback_google(request: Request):
    """Handle Google OAuth callback."""
    if "google" not in enabled_providers:
        raise HTTPException(status_code=404, detail="Google login is not configured")

    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo", {})
        if not user_info:
            try:
                user_info = await oauth.google.userinfo(token=token)
            except Exception:
                user_info = {}

        email = (user_info.get("email") or "").strip().lower()
        sub = user_info.get("sub") or token.get("sub") or ""
        name = user_info.get("name")
        avatar_url = user_info.get("picture")

        if not email or not sub:
            raise HTTPException(status_code=400, detail="Google profile is missing required fields")

        user = get_user_by_oauth("google", sub)
        if not user:
            user = get_user_by_email(email)
        if not user:
            user_id, _ = create_user(email=email, name=name, avatar_url=avatar_url)
            user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=500, detail="Unable to create user")

        link_oauth_identity(
            user_id=user["id"],
            provider="google",
            provider_user_id=sub,
            access_token=token.get("access_token"),
            refresh_token=token.get("refresh_token"),
        )

        request.session["user_id"] = user["id"]
        return RedirectResponse(url="/", status_code=302)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Google OAuth callback error: {exc}")
        raise HTTPException(status_code=500, detail="Login failed")


# ============= BILLING ROUTES =============

@app.post("/api/billing/checkout")
async def get_checkout_url(request: Request):
    """Create hosted Dodo checkout session for subscription upgrade."""
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not dodo_client or not DODO_SUBSCRIPTION_PRODUCT_ID:
        raise HTTPException(status_code=500, detail="Billing is not configured")

    user = get_user_by_id(user_id)
    if not user:
        request.session.pop("user_id", None)
        raise HTTPException(status_code=401, detail="User not found")

    session = dodo_client.checkout_sessions.create(
        product_cart=[{"product_id": DODO_SUBSCRIPTION_PRODUCT_ID, "quantity": 1}],
        customer={"email": user["email"], "name": user.get("name")},
        metadata={"user_id": user_id},
        return_url=f"{PUBLIC_BASE_URL}/?billing=return",
    )

    checkout_url = getattr(session, "checkout_url", None) or getattr(session, "url", None)
    session_id = getattr(session, "session_id", None)
    if not checkout_url:
        raise HTTPException(status_code=502, detail="Checkout URL missing from billing provider response")

    return {"checkout_url": checkout_url, "session_id": session_id}


@app.post("/api/billing/webhook")
async def handle_webhook(request: Request):
    """Handle Dodo Payments webhook events."""
    if not dodo_client:
        raise HTTPException(status_code=500, detail="Billing is not configured")

    webhook_id = request.headers.get("webhook-id", "")
    webhook_signature = request.headers.get("webhook-signature", "")
    webhook_timestamp = request.headers.get("webhook-timestamp", "")

    if not webhook_id:
        raise HTTPException(status_code=400, detail="Missing webhook-id")

    raw_body = await request.body()

    try:
        unwrapped = dodo_client.webhooks.unwrap(
            raw_body,
            headers={
                "webhook-id": webhook_id,
                "webhook-signature": webhook_signature,
                "webhook-timestamp": webhook_timestamp,
            },
        )
    except Exception as exc:
        logger.error(f"Webhook signature verification failed: {exc}")
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = unwrapped.model_dump() if hasattr(unwrapped, "model_dump") else unwrapped
    if not isinstance(payload, dict):
        payload = {}

    event_type = payload.get("type", "")
    data = payload.get("data") or {}
    subscription = data.get("subscription") or (data if data.get("payload_type") == "Subscription" else {})

    with webhook_processing_lock(webhook_id):
        if is_webhook_processed(webhook_id):
            return {"received": True}

        if event_type.startswith("subscription.") or subscription:
            metadata = subscription.get("metadata") or {}
            customer = subscription.get("customer") or {}
            email = (customer.get("email") or subscription.get("customer_email") or "").strip().lower()

            user_id = metadata.get("user_id")
            if not user_id and email:
                user = get_user_by_email(email)
                user_id = user["id"] if user else None

            if user_id:
                status = (subscription.get("status") or event_type.replace("subscription.", "") or "unknown").lower()
                set_subscription(
                    user_id=user_id,
                    provider="dodo",
                    customer_id=customer.get("id") or subscription.get("customer_id"),
                    subscription_id=subscription.get("id") or subscription.get("subscription_id"),
                    status=status,
                    period_start=subscription.get("period_start") or subscription.get("current_period_start"),
                    period_end=subscription.get("period_end") or subscription.get("current_period_end"),
                    plan_code=subscription.get("plan_code") or subscription.get("product_id"),
                )

        record_webhook(webhook_id, event_type)
    return {"received": True}


@app.get("/api/usage")
async def get_usage(request: Request):
    """Get user's quota information."""
    user_id = request.session.get("user_id")
    fingerprint = request.headers.get("X-Device-Fingerprint")
    anon_quota_id = None if user_id else f"session:{get_or_create_anon_session_id(request)}"
    ip_header = get_client_ip(request)
    
    allowed, message, retry_after, usage_info = await check_quota(
        anon_quota_id or fingerprint, user_id, ip_header
    )
    
    return {
        "authenticated": bool(user_id),
        "quota_exhausted": not allowed,
        "debug_mode": ENVIRONMENT == "development",
        **usage_info
    }

# Validation functions
def validate_job_id(job_id: str) -> str:
    """Validate job ID format to prevent path traversal attacks."""
    if not re.match(r'^[a-f0-9\-]{36}$', job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    return job_id


def validate_stamp_text(text: str) -> str:
    """Validate and sanitize stamp text."""
    if not text:
        return "Abhishek Does Stuff"
    
    # Max 50 chars
    if len(text) > 50:
        raise HTTPException(status_code=400, detail="Stamp text too long (max 50 characters)")
    
    # Allow alphanumeric, spaces, and basic punctuation
    if not re.match(r'^[\w\s\-\.\'\!\?]+$', text):
        raise HTTPException(status_code=400, detail="Invalid characters in stamp text")
    
    return text.strip()


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    true_ip = request.headers.get("True-Client-IP")
    if true_ip:
        return true_ip.strip()

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    elif request.client:
        return request.client.host
    return "unknown"


def get_anon_owner_id(request: Request) -> str:
    """Build a stable anonymous owner id from fingerprint or client IP."""
    device_fingerprint = request.headers.get("X-Device-Fingerprint")
    source = device_fingerprint or get_client_ip(request) or "unknown"
    return hashlib.sha256(source.encode()).hexdigest()[:32]


def get_or_create_anon_session_id(request: Request) -> str:
    """Get persistent anonymous session id, creating one if missing."""
    anon_id = request.session.get("anon_id")
    if anon_id:
        return anon_id

    anon_id = uuid.uuid4().hex
    request.session["anon_id"] = anon_id
    return anon_id


@dataclass
class PreprocessOptions:
    """User-selectable preprocessing controls for uploaded images."""
    face_crop: bool = PREPROCESS_DEFAULT_FACE_CROP
    face_crop_padding: float = PREPROCESS_DEFAULT_FACE_PADDING
    auto_downsize: bool = PREPROCESS_DEFAULT_AUTO_DOWNSIZE


def _normalize_preprocess_options(
    face_crop: bool,
    face_crop_padding: float,
    auto_downsize: bool,
) -> PreprocessOptions:
    """Clamp user-provided preprocessing options to safe bounds."""
    padding = max(0.0, min(1.0, float(face_crop_padding)))
    return PreprocessOptions(
        face_crop=bool(face_crop),
        face_crop_padding=padding,
        auto_downsize=bool(auto_downsize),
    )


def _resize_with_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Resize image preserving aspect ratio so max dimension is max_edge."""
    if max_edge <= 0:
        return image

    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image

    scale = max_edge / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _detect_largest_face_bbox(image: Image.Image, face_crop_padding: float) -> Optional[Tuple[int, int, int, int]]:
    """Detect the most relevant face bbox (x1, y1, x2, y2) in image coordinates."""
    if mp is None:
        global _FACE_CROP_UNAVAILABLE_LOGGED
        if not _FACE_CROP_UNAVAILABLE_LOGGED:
            logger.warning("mediapipe not installed; skipping face crop preprocessing")
            _FACE_CROP_UNAVAILABLE_LOGGED = True
        return None

    src_w, src_h = image.size
    detection_image = _resize_with_max_edge(image, FACE_DETECT_MAX_EDGE)
    det_w, det_h = detection_image.size
    scale_x = src_w / det_w
    scale_y = src_h / det_h

    rgb = np.asarray(detection_image.convert("RGB"))

    best_bbox = None
    best_score = -1.0
    center_x = det_w / 2.0
    center_y = det_h / 2.0
    norm = max(det_w, det_h) ** 2

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=FACE_DETECT_MIN_CONFIDENCE,
    ) as detector:
        results = detector.process(rgb)

    detections = results.detections if results and results.detections else []
    for detection in detections:
        rel_box = detection.location_data.relative_bounding_box
        x = int(rel_box.xmin * det_w)
        y = int(rel_box.ymin * det_h)
        w = int(rel_box.width * det_w)
        h = int(rel_box.height * det_h)

        if w <= 0 or h <= 0:
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(det_w, x + w)
        y2 = min(det_h, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        area = float((x2 - x1) * (y2 - y1))
        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0
        dist = ((face_cx - center_x) ** 2 + (face_cy - center_y) ** 2) / norm
        score = area * (1.0 - 0.25 * min(dist, 1.0))

        if score > best_score:
            best_score = score
            best_bbox = (x1, y1, x2, y2)

    if not best_bbox:
        return None

    bx1, by1, bx2, by2 = best_bbox
    face_w = bx2 - bx1
    face_h = by2 - by1
    pad_x = int(face_w * max(0.0, face_crop_padding))
    pad_y = int(face_h * max(0.0, face_crop_padding))

    bx1 = max(0, bx1 - pad_x)
    by1 = max(0, by1 - pad_y)
    bx2 = min(det_w, bx2 + pad_x)
    by2 = min(det_h, by2 + pad_y)

    ox1 = int(bx1 * scale_x)
    oy1 = int(by1 * scale_y)
    ox2 = int(bx2 * scale_x)
    oy2 = int(by2 * scale_y)

    ox1 = max(0, min(src_w - 1, ox1))
    oy1 = max(0, min(src_h - 1, oy1))
    ox2 = max(ox1 + 1, min(src_w, ox2))
    oy2 = max(oy1 + 1, min(src_h, oy2))

    return ox1, oy1, ox2, oy2


def _encode_jpeg_under_limit(image: Image.Image, max_bytes: int) -> bytes:
    """Encode image as JPEG under a max byte size with iterative downscaling."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    work = _resize_with_max_edge(image, PREPROCESS_MAX_EDGE)
    quality = max(55, min(PREPROCESS_JPEG_QUALITY, 95))
    best = b""

    for _ in range(8):
        buf = io.BytesIO()
        work.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if not best or len(data) < len(best):
            best = data
        if len(data) <= max_bytes:
            return data

        quality = max(55, quality - 8)
        next_w = max(512, int(work.width * 0.85))
        next_h = max(512, int(work.height * 0.85))
        if next_w == work.width and next_h == work.height:
            break
        work = work.resize((next_w, next_h), Image.LANCZOS)

    return best


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    """Encode image to JPEG without iterative downsizing."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(55, min(quality, 95)), optimize=True)
    return buf.getvalue()


def preprocess_uploaded_image(image_bytes: bytes, options: PreprocessOptions) -> Tuple[bytes, Dict[str, Any]]:
    """Prepare uploaded image for better BFL reliability (face crop + downsize)."""
    try:
        source_image = Image.open(io.BytesIO(image_bytes))
        orientation_value = source_image.getexif().get(274, 1)
        source_image = ImageOps.exif_transpose(source_image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to parse image file") from exc

    face_cropped = False
    if options.face_crop:
        try:
            bbox = _detect_largest_face_bbox(source_image, options.face_crop_padding)
            if bbox:
                source_image = source_image.crop(bbox)
                face_cropped = True
        except Exception as exc:
            logger.warning(f"Face crop failed, continuing with original image: {exc}")

    output_limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    orientation_corrected = orientation_value not in (None, 1)
    if not face_cropped and not orientation_corrected and len(image_bytes) <= output_limit_bytes:
        return image_bytes, {
            "input_bytes": len(image_bytes),
            "output_bytes": len(image_bytes),
            "face_cropped": False,
            "output_format": "original",
            "passthrough": True,
        }

    if options.auto_downsize:
        output_bytes = _encode_jpeg_under_limit(source_image, output_limit_bytes)
    else:
        output_bytes = _encode_jpeg(source_image, PREPROCESS_JPEG_QUALITY)

    if len(output_bytes) > output_limit_bytes:
        detail = (
            f"Image exceeds {MAX_FILE_SIZE_MB}MB after preprocessing. "
            "Enable auto-downsize or upload a smaller image."
        )
        raise HTTPException(status_code=413, detail=detail)

    meta = {
        "input_bytes": len(image_bytes),
        "output_bytes": len(output_bytes),
        "face_cropped": face_cropped,
        "auto_downsize": options.auto_downsize,
        "face_crop": options.face_crop,
        "face_crop_padding": options.face_crop_padding,
        "output_format": "jpeg",
        "passthrough": False,
    }
    return output_bytes, meta


async def preprocess_uploaded_image_async(image_bytes: bytes, options: PreprocessOptions) -> Tuple[bytes, Dict[str, Any]]:
    """Async wrapper for image preprocessing."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, preprocess_uploaded_image, image_bytes, options)


def enforce_job_access(request: Request, job: Job) -> None:
    """Ensure requester is authorized to access this job."""
    session_user_id = request.session.get("user_id")

    # Authenticated jobs must match authenticated owner.
    if job.owner_user_id:
        if session_user_id != job.owner_user_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        return

    # Anonymous jobs must match anonymous owner hash.
    if job.owner_anon_id:
        # Preferred path: persistent anonymous session ownership.
        session_anon_id = request.session.get("anon_id")
        if session_anon_id and session_anon_id == job.owner_anon_id:
            return

        # Backward compatibility for legacy jobs using hash-based owner id.
        if get_anon_owner_id(request) == job.owner_anon_id:
            return

        raise HTTPException(status_code=403, detail="Forbidden")

    # Backward compatibility for old jobs created before ownership fields existed.
    # Keep access behavior unchanged for legacy jobs.
    return


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(JOBS_DIR, exist_ok=True)
    # Clean up old jobs on startup (non-blocking, only on one worker)
    cleanup_lock_file = os.path.join(TEMP_DIR, ".cleanup_lock")
    try:
        # Try to create lock file - if it exists, another worker is doing cleanup
        if not os.path.exists(cleanup_lock_file):
            with open(cleanup_lock_file, 'w') as f:
                f.write(str(os.getpid()))
            # Run cleanup asynchronously so it doesn't block startup
            asyncio.create_task(async_cleanup_old_jobs(cleanup_lock_file))
    except Exception as e:
        logger.warning(f"Could not start cleanup: {e}")
    
    logger.info(f"Application started. Debug mode: {DEBUG_NO_CLEANUP}")


async def async_cleanup_old_jobs(lock_file: str, max_age_hours: int = 24):
    """Async cleanup that runs in background."""
    try:
        await asyncio.sleep(5)  # Wait 5 seconds after startup to let things settle
        JobStore.cleanup_old_jobs(max_age_hours)
        
        # Also cleanup old rate limit entries if rate limiter exists
        if rate_limiter:
            await rate_limiter.cleanup_old()
            logger.info("Rate limit cleanup completed")
    finally:
        # Remove lock file when done
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint for Coolify and monitoring."""
    # Count job files in jobs directory
    job_count = 0
    if os.path.exists(JOBS_DIR):
        job_count = len([f for f in os.listdir(JOBS_DIR) if f.endswith('.json')])
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_jobs": job_count,
        "debug_mode": DEBUG_NO_CLEANUP
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check for Coolify."""
    temp_dir_exists = os.path.exists(TEMP_DIR)
    return {
        "ready": temp_dir_exists,
        "temp_dir": TEMP_DIR,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/rate-limit-status")
async def get_rate_limit_status(request: Request):
    """Check rate limit status for the current user."""
    if not RATE_LIMIT_ENABLED:
        return {
            "limited": False,
            "can_create_job": True,
            "message": "Rate limiting is disabled",
            "bypass_available": ALLOW_BYPASS_WITH_API_KEY
        }
    
    return {
        "limited": False,  # Would need to check actual storage for real status
        "can_create_job": True,
        "message": f"You can create 1 job per {RATE_LIMIT_COOLDOWN_HOURS} hours",
        "cooldown_hours": RATE_LIMIT_COOLDOWN_HOURS,
        "bypass_available": ALLOW_BYPASS_WITH_API_KEY
    }


async def update_job_status(job_id: str, status: str, progress: int, message: str):
    """Update job status on disk."""
    job = JobStore.get_job(job_id)
    if job:
        job.status = status
        job.progress = progress
        job.message = message
        JobStore.save_job(job)


async def bfl_flux_process(image_bytes: bytes, api_key: str, stamp_text: str = "Abhishek Does Stuff") -> bytes:
    """
    Process image through BFL FLUX API to create flat vector illustration.
    Optimized to use aiohttp for true async HTTP requests.

    Args:
        image_bytes: Raw image bytes
        api_key: BFL API key
        stamp_text: Text to display on coaster stamp

    Returns:
        Processed PNG image bytes
    """
    logger.info("="*60)
    logger.info("BFL FLUX PROCESS - Starting image processing")
    logger.info(f"Input image size: {len(image_bytes)} bytes")
    logger.info(f"Stamp text: {stamp_text}")

    # Convert image to base64 (raw base64 string, not data URL)
    logger.debug("Converting image to base64...")
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    logger.debug(f"Base64 encoded length: {len(image_base64)} chars")

    # Load prompt template and substitute stamp text
    template_file_path = os.path.join(os.path.dirname(__file__), "prompt_template.txt")
    logger.debug(f"Loading prompt template from: {template_file_path}")
    try:
        with open(template_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read().strip()
        # Replace {{stamp_text}} with user-provided text
        prompt = prompt_template.replace("{{stamp_text}}", stamp_text)
        logger.info(f"✓ Prompt template loaded: {len(prompt_template)} chars")
        logger.info(f"✓ Prompt generated with stamp text: {len(prompt)} chars")
    except Exception as e:
        logger.error(f"Failed to load prompt template from file: {e}")
        logger.warning("Using fallback prompt")
        prompt = f"flat vector illustration, solid colors, no gradients, high contrast, 2d cartoon style, white background, clean lines suitable for vector tracing. Circular stamp with text '{stamp_text}' on top arc."
    
    # Set parameters based on BFL API documentation
    width = 512
    height = 512
    seed = 432262096973491  # Same seed as ComfyUI workflow for reproducibility
    output_format = "png"  # Get PNG directly to avoid conversion issues
    
    payload = {
        "prompt": prompt,
        "input_image": image_base64,
        "width": width,
        "height": height,
        "seed": seed,
        "output_format": output_format,
    }
    
    logger.info(f"Image dimensions: {width}x{height}")
    logger.info(f"Seed: {seed} (for reproducibility)")
    logger.info(f"Output format: {output_format}")
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-key": api_key
    }
    
    logger.info(f"Submitting job to BFL API: {BFL_API_URL}/flux-2-klein-9b")
    logger.debug(f"Payload keys: {list(payload.keys())}")
    
    # Use aiohttp session for true async HTTP requests
    async with aiohttp.ClientSession() as session:
        # Submit job
        logger.debug("Making POST request to BFL...")
        async with session.post(
            f"{BFL_API_URL}/flux-2-klein-9b",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            logger.debug(f"BFL Response status: {response.status}")
            if response.status != 200:
                text = await response.text()
                logger.error(f"BFL API error: {response.status}")
                logger.error(f"Response text: {text[:500]}")
                provider_error = f"BFL API error: {response.status} - {text}"
                if response.status == 429 or is_quota_related_error(provider_error):
                    provider_error = with_support_contact(provider_error)
                raise Exception(provider_error)
            result = await response.json()
    
    logger.info(f"BFL job submitted successfully")
    logger.debug(f"Response keys: {list(result.keys())}")
    
    # Extract polling URL and request ID from response
    request_id = result.get("id")
    polling_url = result.get("polling_url")
    
    logger.info(f"Request ID: {request_id}")
    logger.info(f"Polling URL: {polling_url}")
    
    if not request_id or not polling_url:
        logger.error("Missing request_id or polling_url in response")
        logger.error(f"Full response: {result}")
        raise Exception("No request ID or polling URL received from BFL API")
    
    # Poll for result using the provided polling_url
    logger.info("Starting polling loop...")
    async with aiohttp.ClientSession() as session:
        for attempt in range(MAX_POLLING_ATTEMPTS):
            await asyncio.sleep(POLLING_INTERVAL)
            logger.debug(f"Polling attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS}...")
            
            try:
                async with session.get(
                    polling_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as poll_response:
                    logger.debug(f"Poll response status: {poll_response.status}")
                    if poll_response.status != 200:
                        logger.warning(f"Poll failed with status {poll_response.status}")
                        continue
                    
                    poll_result = await poll_response.json()
            except asyncio.TimeoutError:
                logger.warning(f"Poll attempt {attempt + 1} timed out, retrying...")
                continue
            except Exception as e:
                logger.warning(f"Poll attempt {attempt + 1} failed: {e}, retrying...")
                continue
            
            status = poll_result.get("status")
            logger.info(f"Poll status: {status}")
            
            if status == "Ready":
                logger.info("BFL job completed! Downloading result...")
                # Extract the sample URL from result
                result_data = poll_result.get("result", {})
                sample_url = result_data.get("sample")
                logger.info(f"Sample URL: {sample_url[:100]}...")
                if sample_url:
                    # Download the result image
                    logger.debug(f"Downloading from signed URL...")
                    try:
                        async with session.get(
                            sample_url,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as image_response:
                            logger.debug(f"Download response status: {image_response.status}")
                            if image_response.status == 200:
                                image_data = await image_response.read()
                                logger.info(f"Downloaded image: {len(image_data)} bytes")
                                return image_data
                            logger.error(f"Download failed: {image_response.status}")
                    except asyncio.TimeoutError:
                        logger.error("Download timed out")
                raise Exception("No image URL in completed result")
            
            elif status in ["Failed", "Error"]:
                provider_error = f"BFL job failed: {poll_result.get('error', 'Unknown error')}"
                if is_quota_related_error(provider_error):
                    provider_error = with_support_contact(provider_error)
                raise Exception(provider_error)
            
            # Continue polling for Pending, Processing, etc.
    
    raise Exception(f"BFL job timed out after {MAX_POLLING_ATTEMPTS * POLLING_INTERVAL} seconds")


async def process_coaster_job(
    job_id: str,
    image_bytes: bytes,
    params: ProcessRequest,
    stamp_text: str,
    api_key: str
):
    """
    Background task - Phase 1: Process image through BFL and wait for confirmation.

    Args:
        job_id: Job identifier
        image_bytes: Uploaded image bytes
        params: Coaster parameters
        stamp_text: Text to display on coaster stamp
        api_key: BFL API key to use
    """
    logger.info("="*60)
    logger.info(f"PHASE 1 STARTED - Job ID: {job_id}")
    logger.info(f"Input image size: {len(image_bytes)} bytes")
    logger.info(f"Parameters: {params}")
    logger.info(f"Stamp text: {stamp_text}")

    try:
        # Get job object
        job = JobStore.get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")

        # Validate API key
        if not api_key:
            logger.error("No BFL API key available!")
            raise Exception("No BFL API key available. Please provide your API key in the form or set BFL_API_KEY environment variable.")

        logger.info(f"API key present: {bool(api_key)}")
        
        # Step 1: Flatten image with BFL FLUX
        logger.info("STEP 1/2: Flattening image with BFL FLUX API...")
        await update_job_status(job_id, "processing_flatten", 50, "Flattening image with AI...")
        logger.info("Calling bfl_flux_process...")
        flattened_image = await bfl_flux_process(image_bytes, api_key, stamp_text)
        logger.info(f"✓ Flattening complete: {len(flattened_image)} bytes")
        
        # Save preview image to disk for persistence across workers
        preview_path = JobStore.save_preview_image(job_id, flattened_image)
        
        # Update job with paths and params for confirmation step
        job.preview_image_path = preview_path
        job.params = params
        job.status = "review"
        job.progress = 50
        job.message = "Image generated! Please review and confirm to proceed."
        JobStore.save_job(job)
        
        logger.info(f"✓✓✓ JOB {job_id} READY FOR REVIEW ✓✓✓")
        logger.info("="*60)
        
    except Exception as e:
        error_message = str(e)
        if is_quota_related_error(error_message):
            error_message = with_support_contact(error_message)

        logger.error("="*60)
        logger.error(f"JOB {job_id} FAILED!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {error_message}")
        logger.exception("Full traceback:")
        logger.error("="*60)
        
        job = JobStore.get_job(job_id)
        if job:
            job.status = "failed"
            job.progress = 0
            job.message = f"Error: {error_message}"
            job.error = error_message
            JobStore.save_job(job)

def run_3d_processing_pipeline(flattened_image: bytes, params: ProcessRequest, job_id: str):
    """
    Pure synchronous function that runs the heavy CPU bounds tasks.
    Suitable for running inside a ProcessPoolExecutor.
    """
    generator = CoasterGenerator(
        diameter=params.diameter,
        thickness=params.thickness,
        logo_depth=params.logo_depth,
        top_logo_height=params.top_logo_height,
        scale=params.scale,
        flip_horizontal=params.flip_horizontal,
        top_rotate=params.top_rotate,
        bottom_rotate=params.bottom_rotate,
        nozzle_diameter=params.nozzle_diameter,
        auto_thicken=params.auto_thicken,
    )

    timestamp = datetime.now().strftime("%H%M%S")
    file_prefix = f"{job_id}_{timestamp}"
    temp_input_path = os.path.join(TEMP_DIR, f"{file_prefix}_input.png")

    with open(temp_input_path, "wb") as f:
        f.write(flattened_image)

    try:
        return generator.generate_coaster(
            input_image_path=temp_input_path,
            output_dir=TEMP_DIR,
            stamp_text="",
            is_preview=False,
            file_prefix=file_prefix,
        )
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


async def process_vectorization_3d(job_id: str):
    """
    Background task - Phase 2: Vectorization and 3D generation (after confirmation).
    
    Args:
        job_id: Job identifier
    """
    global active_local_tasks
    logger.info("="*60)
    logger.info(f"PHASE 2 STARTED - Job ID: {job_id}")
    
    try:
        job = JobStore.get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Load preview image from disk
        flattened_image = JobStore.get_preview_image(job_id)
        params = job.params
        stamp_text = job.stamp_text or "Abhishek Does Stuff"
        
        if not flattened_image or not params:
            raise Exception("Missing preview image or parameters")
            
        await update_job_status(job_id, "processing_3d", 75, "Starting 3D generation...")
        
        # Check active local tasks to decide routing
        local_slot_reserved = False
        force_local = bool(getattr(job, "uses_own_api_key", False))
        async with tasks_lock:
            current_active = active_local_tasks
            if force_local:
                active_local_tasks += 1
                local_slot_reserved = True
                route_to_modal = False
            elif current_active < MAX_LOCAL_WORKERS:
                active_local_tasks += 1
                local_slot_reserved = True
                route_to_modal = False
            else:
                route_to_modal = True
                
        try:
            if route_to_modal and USE_MODAL_FALLBACK:
                logger.info("ROUTING TO CLOUD: Local pool full, bursting to Modal...")
                await update_job_status(job_id, "processing_3d", 80, "Generating on cloud GPU/CPU...")
                
                # Convert image to bytes to send over wire
                if isinstance(flattened_image, (bytes, bytearray)):
                    img_bytes = bytes(flattened_image)
                else:
                    img_byte_arr = io.BytesIO()
                    flattened_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                
                # Call Modal function synchronously (it's network IO bound from our perspective)
                loop = asyncio.get_running_loop()
                modal_func = _lookup_modal_function(MODAL_APP_NAME, "generate_3d_coaster")
                params_payload = params.model_dump() if hasattr(params, "model_dump") else params.dict()
                combined_bytes, body_bytes, logos_bytes = await loop.run_in_executor(
                    None, 
                    modal_func.remote, 
                    img_bytes, 
                    params_payload,
                    stamp_text
                )
                
                # Save received bytes to local disk
                timestamp = datetime.now().strftime("%H%M%S")
                base_name = f"{job_id}_{timestamp}"
                
                coaster_3mf_path = os.path.join(TEMP_DIR, f"{base_name}_coaster.3mf")
                body_stl_path = os.path.join(TEMP_DIR, f"{base_name}_Body.stl")
                logos_stl_path = os.path.join(TEMP_DIR, f"{base_name}_Logos.stl")
                
                with open(coaster_3mf_path, "wb") as f: f.write(combined_bytes)
                with open(body_stl_path, "wb") as f: f.write(body_bytes)
                with open(logos_stl_path, "wb") as f: f.write(logos_bytes)
                
                logger.info(f"✓ Cloud 3D generation complete via Modal")
                
            else:
                if force_local:
                    logger.info("ROUTING LOCALLY: User-supplied BFL API key forces local processing.")
                elif route_to_modal:
                    logger.warning("ROUTING LOCALLY: Local pool full, but Modal fallback disabled. Queuing locally...")
                else:
                    logger.info("ROUTING LOCALLY: Processing on M900 Tiny...")
                    
                await update_job_status(job_id, "processing_3d", 85, "Generating 3D model locally...")
                
                # Run the heavy CPU math in the local ProcessPoolExecutor
                loop = asyncio.get_running_loop()
                coaster_3mf_path, body_stl_path, logos_stl_path = await loop.run_in_executor(
                    process_pool,
                    run_3d_processing_pipeline,
                    flattened_image,
                    params,
                    job_id
                )
                logger.info(f"✓ Local 3D generation complete")
        finally:
            if local_slot_reserved:
                async with tasks_lock:
                    active_local_tasks = max(0, active_local_tasks - 1)
        
        logger.info(f"  - 3MF: {coaster_3mf_path}")
        logger.info(f"  - Body STL: {body_stl_path}")
        logger.info(f"  - Logos STL: {logos_stl_path}")
        
        # Update job with file paths
        logger.info("Updating job status to completed...")
        job.files = {
            "combined_3mf": coaster_3mf_path,
            "body": body_stl_path,
            "logos": logos_stl_path
        }
        job.status = "completed"
        job.progress = 100
        job.message = "Coaster generation complete!"
        JobStore.save_job(job)
        
        logger.info(f"✓✓✓ JOB {job_id} COMPLETED SUCCESSFULLY ✓✓✓")
        logger.info("="*60)
        
    except Exception as e:
        error_message = str(e)
        if is_quota_related_error(error_message):
            error_message = with_support_contact(error_message)

        logger.error("="*60)
        logger.error(f"JOB {job_id} FAILED in Phase 2!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {error_message}")
        logger.exception("Full traceback:")
        logger.error("="*60)
        
        job = JobStore.get_job(job_id)
        if job:
            job.status = "failed"
            job.progress = 0
            job.message = f"Error: {error_message}"
            job.error = error_message
            JobStore.save_job(job)


def _resolve_identity_context(request: Request):
    """Resolve identity and network context used for quota and ownership checks."""
    user_id = request.session.get("user_id")
    device_fingerprint = request.headers.get("X-Device-Fingerprint")
    anon_quota_id = None if user_id else f"session:{get_or_create_anon_session_id(request)}"
    client_ip = get_client_ip(request)
    return user_id, device_fingerprint, anon_quota_id, client_ip


def _build_process_params(
    diameter: float,
    thickness: float,
    logo_depth: float,
    top_logo_height: float,
    scale: float,
    flip_horizontal: bool,
    top_rotate: int,
    bottom_rotate: int,
    nozzle_diameter: float,
    auto_thicken: bool,
) -> ProcessRequest:
    """Create normalized ProcessRequest model from endpoint inputs."""
    return ProcessRequest(
        diameter=diameter,
        thickness=thickness,
        logo_depth=logo_depth,
        top_logo_height=top_logo_height,
        scale=scale,
        flip_horizontal=flip_horizontal,
        top_rotate=top_rotate,
        bottom_rotate=bottom_rotate,
        nozzle_diameter=nozzle_diameter,
        auto_thicken=auto_thicken,
    )


async def _consume_quota_or_raise(
    job_id: str,
    user_id: Optional[str],
    device_fingerprint: Optional[str],
    anon_quota_id: Optional[str],
    client_ip: str,
    bypass_limit: bool,
) -> Dict[str, Any]:
    """Run product quota checks and consume quota for a job unless bypassing."""
    if bypass_limit:
        logger.info("Bypassing quota consumption because user provided their own BFL API key")
        return {}

    usage_info: Dict[str, Any] = {}
    quota_allowed, quota_message, _, usage_info, quota_event_id = await check_and_consume_quota_atomic(
        job_id=job_id,
        fingerprint=anon_quota_id or device_fingerprint,
        user_id=user_id,
        ip_header=client_ip,
    )
    if not quota_allowed:
        quota_message = usage_info.get("message") or quota_message
        quota_message = with_support_contact(quota_message)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": quota_message,
                "next_action": usage_info.get("next_action"),
                "usage": usage_info,
            },
        )

    if not quota_event_id:
        raise HTTPException(status_code=500, detail="Unable to reserve quota at this time")

    return usage_info


def _resolve_effective_api_key(api_key: str) -> str:
    """Resolve user-provided key or server fallback key."""
    effective_api_key = api_key if api_key else os.environ.get("BFL_API_KEY")
    if not effective_api_key:
        raise HTTPException(status_code=500, detail="No API key configured")
    return effective_api_key


def _queue_phase1_job(
    background_tasks: BackgroundTasks,
    job_id: str,
    image_bytes: bytes,
    params: ProcessRequest,
    stamp_text: str,
    effective_api_key: str,
) -> None:
    """Queue phase-1 processing task."""
    background_tasks.add_task(process_coaster_job, job_id, image_bytes, params, stamp_text, effective_api_key)


@app.post("/api/process")
async def process_image(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    diameter: float = Form(100.0),
    thickness: float = Form(5.0),
    logo_depth: float = Form(0.6),
    top_logo_height: float = Form(0.0),
    scale: float = Form(0.85),
    flip_horizontal: bool = Form(True),
    top_rotate: int = Form(0),
    bottom_rotate: int = Form(0),
    nozzle_diameter: float = Form(0.4),
    auto_thicken: bool = Form(True),
    preprocess_face_crop: bool = Form(PREPROCESS_DEFAULT_FACE_CROP),
    preprocess_face_padding: float = Form(PREPROCESS_DEFAULT_FACE_PADDING),
    preprocess_auto_downsize: bool = Form(PREPROCESS_DEFAULT_AUTO_DOWNSIZE),
    api_key: str = Form(""),
    stamp_text: str = Form("Abhishek Does Stuff")
):
    """
    Start a new coaster generation job.

    Accepts an image file and parameters, returns job ID for tracking.
    Rate limited to 1 job per week unless user provides their own API key.
    """
    logger.info("="*60)
    logger.info("API REQUEST: POST /api/process - New coaster job")
    logger.info(f"Image filename: {image.filename}")
    logger.info(f"Image content-type: {image.content_type}")
    logger.info(f"Parameters: diameter={diameter}, thickness={thickness}, "
                f"logo_depth={logo_depth}, scale={scale}, "
                f"flip_horizontal={flip_horizontal}, top_rotate={top_rotate}, "
                f"bottom_rotate={bottom_rotate}, nozzle_diameter={nozzle_diameter}, "
                f"auto_thicken={auto_thicken}")

    # Validate stamp text
    stamp_text = validate_stamp_text(stamp_text)
    logger.info(f"Stamp text validated: {stamp_text}")

    # Resolve identity context
    user_id, device_fingerprint, anon_quota_id, client_ip = _resolve_identity_context(request)

    # Check if bypassing with own API key
    bypass_limit = bool(api_key) and ALLOW_BYPASS_WITH_API_KEY
    using_own_key = bool(api_key)

    # Check rate limit (if enabled and not bypassing)
    if RATE_LIMIT_ENABLED and not bypass_limit and rate_limiter:
        # Create fingerprint
        fingerprint = get_rate_limit_key(request)
        
        # Check if user is allowed
        allowed, message, retry_after = await rate_limiter.is_allowed(fingerprint)
        
        if not allowed:
            logger.warning(f"Rate limit hit for fingerprint: {fingerprint[:16]}...")
            rate_limit_message = with_support_contact(message or "Rate limit exceeded")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": rate_limit_message,
                    "retry_after": retry_after,
                    "bypass_available": ALLOW_BYPASS_WITH_API_KEY
                }
            )
        
        logger.info(f"Rate limit check passed for fingerprint: {fingerprint[:16]}...")

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        logger.error(f"Invalid content type: {image.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    logger.info("✓ Content type validated")

    # Read and validate image bytes
    logger.debug("Reading image bytes...")
    raw_image_bytes = await image.read()
    if len(raw_image_bytes) == 0:
        logger.error("Empty image file received")
        raise HTTPException(status_code=400, detail="Empty image file")

    raw_max_size = MAX_RAW_FILE_SIZE_MB * 1024 * 1024
    if len(raw_image_bytes) > raw_max_size:
        logger.error(f"Raw upload too large: {len(raw_image_bytes)} bytes (max {raw_max_size})")
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_RAW_FILE_SIZE_MB}MB)")

    preprocess_options = _normalize_preprocess_options(
        face_crop=preprocess_face_crop,
        face_crop_padding=preprocess_face_padding,
        auto_downsize=preprocess_auto_downsize,
    )
    image_bytes, preprocess_meta = await preprocess_uploaded_image_async(raw_image_bytes, preprocess_options)
    logger.info(
        "✓ Image preprocessed: %s -> %s bytes (face_cropped=%s, auto_downsize=%s)",
        preprocess_meta.get("input_bytes"),
        preprocess_meta.get("output_bytes"),
        preprocess_meta.get("face_cropped"),
        preprocess_meta.get("auto_downsize"),
    )

    job_id = str(uuid.uuid4())

    # Quota checks + consumption (authoritative product limits)
    await _consume_quota_or_raise(
        job_id=job_id,
        user_id=user_id,
        device_fingerprint=device_fingerprint,
        anon_quota_id=anon_quota_id,
        client_ip=client_ip,
        bypass_limit=bypass_limit,
    )

    effective_api_key = _resolve_effective_api_key(api_key)

    # Create job only after quota and key checks pass.
    job = Job(
        job_id=job_id,
        stamp_text=stamp_text,
        uses_own_api_key=using_own_key,
        owner_user_id=user_id,
        owner_anon_id=None if user_id else get_or_create_anon_session_id(request),
    )

    source_path = JobStore.save_source_image(job_id, image_bytes)
    job.source_image_path = source_path
    JobStore.save_job(job)
    logger.info(f"✓ Job created: {job_id} with stamp: {stamp_text}")

    # Create parameters object
    params = _build_process_params(
        diameter=diameter,
        thickness=thickness,
        logo_depth=logo_depth,
        top_logo_height=top_logo_height,
        scale=scale,
        flip_horizontal=flip_horizontal,
        top_rotate=top_rotate,
        bottom_rotate=bottom_rotate,
        nozzle_diameter=nozzle_diameter,
        auto_thicken=auto_thicken,
    )
    logger.info("✓ Parameters object created")

    # Start background processing (pass API key separately, not stored)
    logger.info("Starting background processing task...")
    _queue_phase1_job(background_tasks, job_id, image_bytes, params, stamp_text, effective_api_key)
    logger.info(f"✓✓✓ Job {job_id} queued successfully")
    logger.info("="*60)
    
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str, request: Request):
    """Get the current status of a job."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)
    
    response = StatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        error=job.error
    )
    
    # Add download URLs if completed
    if job.status == "completed" and job.files:
        response.download_urls = {
            "combined": f"/api/download/{job_id}",
            "body": f"/api/download/{job_id}/body",
            "logos": f"/api/download/{job_id}/logos"
        }

    return response


@app.get("/api/download/{job_id}")
async def download_coaster(job_id: str, request: Request):
    """Download the combined coaster 3MF file."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)

    if job.status != "completed" or "combined_3mf" not in job.files:
        raise HTTPException(status_code=400, detail="Coaster file not available")

    file_path = job.files["combined_3mf"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        file_path,
        media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        filename=f"coaster_{job_id}.3mf"
    )


@app.get("/api/download/{job_id}/body")
async def download_body_stl(job_id: str, request: Request):
    """Download the coaster body STL file (for viewer)."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)

    if job.status != "completed" or "body" not in job.files:
        raise HTTPException(status_code=400, detail="Body file not available")

    file_path = job.files["body"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=f"coaster_{job_id}_Body.stl"
    )


@app.get("/api/download/{job_id}/logos")
async def download_logos_stl(job_id: str, request: Request):
    """Download the coaster logos STL file (for viewer)."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)

    if job.status != "completed" or "logos" not in job.files:
        raise HTTPException(status_code=400, detail="Logos file not available")

    file_path = job.files["logos"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=f"coaster_{job_id}_Logos.stl"
    )


@app.get("/api/preview-image/{job_id}")
async def get_preview_image(job_id: str, request: Request):
    """Get the BFL generated image for review (before confirmation)."""
    from starlette.responses import StreamingResponse
    
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)
    
    if job.status != "review":
        raise HTTPException(status_code=400, detail="Preview image not available")
    
    # Load preview image from disk
    preview_image = JobStore.get_preview_image(job_id)
    if not preview_image:
        raise HTTPException(status_code=400, detail="Preview image not available")
    
    # Use StreamingResponse for in-memory bytes
    return StreamingResponse(
        io.BytesIO(preview_image),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=preview_{job_id}.png"}
    )


@app.post("/api/confirm/{job_id}")
async def confirm_job(job_id: str, request: Request, background_tasks: BackgroundTasks):
    """Confirm the generated image and proceed with vectorization and 3D generation."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)
    
    if job.status != "review":
        raise HTTPException(status_code=400, detail="Job is not in review state")
    
    logger.info(f"Job {job_id} confirmed by user, starting Phase 2...")
    
    # Start Phase 2 (vectorization + 3D) as background task
    background_tasks.add_task(process_vectorization_3d, job_id)
    
    return {"job_id": job_id, "status": "processing_vectorize", "message": "Processing vectorization and 3D generation..."}


@app.post("/api/regenerate/{job_id}")
async def regenerate_job(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Form(""),
    stamp_text: Optional[str] = Form(None),
    diameter: Optional[float] = Form(None),
    thickness: Optional[float] = Form(None),
    logo_depth: Optional[float] = Form(None),
    top_logo_height: Optional[float] = Form(None),
    scale: Optional[float] = Form(None),
    flip_horizontal: Optional[bool] = Form(None),
    top_rotate: Optional[int] = Form(None),
    bottom_rotate: Optional[int] = Form(None),
    nozzle_diameter: Optional[float] = Form(None),
    auto_thicken: Optional[bool] = Form(None),
):
    """Regenerate preview image from original source image without re-upload."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)

    if job.status != "review":
        raise HTTPException(status_code=400, detail="Job is not in review state")

    if not job.params:
        raise HTTPException(status_code=400, detail="Job parameters missing")

    # Allow users to tweak prompt text/settings before regeneration.
    effective_stamp_text = validate_stamp_text(stamp_text if stamp_text is not None else (job.stamp_text or "Abhishek Does Stuff"))
    current_params = job.params
    job.params = ProcessRequest(
        diameter=diameter if diameter is not None else current_params.diameter,
        thickness=thickness if thickness is not None else current_params.thickness,
        logo_depth=logo_depth if logo_depth is not None else current_params.logo_depth,
        top_logo_height=top_logo_height if top_logo_height is not None else current_params.top_logo_height,
        scale=scale if scale is not None else current_params.scale,
        flip_horizontal=flip_horizontal if flip_horizontal is not None else current_params.flip_horizontal,
        top_rotate=top_rotate if top_rotate is not None else current_params.top_rotate,
        bottom_rotate=bottom_rotate if bottom_rotate is not None else current_params.bottom_rotate,
        nozzle_diameter=nozzle_diameter if nozzle_diameter is not None else current_params.nozzle_diameter,
        auto_thicken=auto_thicken if auto_thicken is not None else current_params.auto_thicken,
    )
    job.stamp_text = effective_stamp_text

    image_bytes = JobStore.get_source_image(job_id)
    if not image_bytes:
        # Backward compatibility: older jobs may not have source image stored.
        image_bytes = JobStore.get_preview_image(job_id)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Original image not available for regeneration")

    user_id = request.session.get("user_id")
    device_fingerprint = request.headers.get("X-Device-Fingerprint")
    anon_quota_id = None if user_id else f"session:{get_or_create_anon_session_id(request)}"
    client_ip = get_client_ip(request)

    bypass_limit = bool(api_key) and ALLOW_BYPASS_WITH_API_KEY
    await _consume_quota_or_raise(
        job_id=job_id,
        user_id=user_id,
        device_fingerprint=device_fingerprint,
        anon_quota_id=anon_quota_id,
        client_ip=client_ip,
        bypass_limit=bypass_limit,
    )

    # Reset Phase-1 status while preserving params, owner, source image and stamp.
    job.status = "pending"
    job.progress = 0
    job.message = "Regenerating image with AI..."
    job.preview_image_path = None
    job.error = None
    job.uses_own_api_key = bool(api_key) if api_key else job.uses_own_api_key
    JobStore.save_job(job)

    effective_api_key = _resolve_effective_api_key(api_key)
    _queue_phase1_job(background_tasks, job_id, image_bytes, job.params, job.stamp_text, effective_api_key)

    return {"job_id": job_id, "status": "processing", "message": "Regenerating image..."}


@app.post("/api/retry/{job_id}")
async def retry_job(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    diameter: float = Form(100.0),
    thickness: float = Form(5.0),
    logo_depth: float = Form(0.6),
    top_logo_height: float = Form(0.0),
    scale: float = Form(0.85),
    flip_horizontal: bool = Form(True),
    top_rotate: int = Form(0),
    bottom_rotate: int = Form(0),
    nozzle_diameter: float = Form(0.4),
    auto_thicken: bool = Form(True),
    preprocess_face_crop: bool = Form(PREPROCESS_DEFAULT_FACE_CROP),
    preprocess_face_padding: float = Form(PREPROCESS_DEFAULT_FACE_PADDING),
    preprocess_auto_downsize: bool = Form(PREPROCESS_DEFAULT_AUTO_DOWNSIZE),
):
    """Retry with a different image, keeping the same job ID."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    enforce_job_access(request, job)
    
    if job.status != "review":
        raise HTTPException(status_code=400, detail="Job is not in review state")
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image bytes
    raw_image_bytes = await image.read()
    if len(raw_image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    raw_max_size = MAX_RAW_FILE_SIZE_MB * 1024 * 1024
    if len(raw_image_bytes) > raw_max_size:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_RAW_FILE_SIZE_MB}MB)")

    preprocess_options = _normalize_preprocess_options(
        face_crop=preprocess_face_crop,
        face_crop_padding=preprocess_face_padding,
        auto_downsize=preprocess_auto_downsize,
    )
    image_bytes, preprocess_meta = await preprocess_uploaded_image_async(raw_image_bytes, preprocess_options)
    logger.info(
        "Retry image preprocessed: %s -> %s bytes (face_cropped=%s, auto_downsize=%s)",
        preprocess_meta.get("input_bytes"),
        preprocess_meta.get("output_bytes"),
        preprocess_meta.get("face_cropped"),
        preprocess_meta.get("auto_downsize"),
    )

    user_id, device_fingerprint, anon_quota_id, client_ip = _resolve_identity_context(request)
    
    await _consume_quota_or_raise(
        job_id=job_id,
        user_id=user_id,
        device_fingerprint=device_fingerprint,
        anon_quota_id=anon_quota_id,
        client_ip=client_ip,
        bypass_limit=False,
    )
    
    # Create parameters object
    params = _build_process_params(
        diameter=diameter,
        thickness=thickness,
        logo_depth=logo_depth,
        top_logo_height=top_logo_height,
        scale=scale,
        flip_horizontal=flip_horizontal,
        top_rotate=top_rotate,
        bottom_rotate=bottom_rotate,
        nozzle_diameter=nozzle_diameter,
        auto_thicken=auto_thicken,
    )
    
    effective_api_key = _resolve_effective_api_key("")

    # Persist updated state only after quota/key checks pass.
    job.status = "pending"
    job.progress = 0
    job.message = "Restarting with new image..."
    job.preview_image_path = None
    source_path = JobStore.save_source_image(job_id, image_bytes)
    job.source_image_path = source_path
    job.error = None
    JobStore.save_job(job)

    # Start background processing with required args
    stamp_text = job.stamp_text or "Abhishek Does Stuff"
    _queue_phase1_job(background_tasks, job_id, image_bytes, params, stamp_text, effective_api_key)
    
    return {"job_id": job_id, "status": "processing", "message": "Restarting with new image..."}


@app.get("/", response_class=HTMLResponse)
async def get_frontend(request: Request):
    """Serve the HTML frontend using Jinja2 template."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": {
            "app_name": "CoastGen",
            "default_stamp": "Abhishek Does Stuff",
            "max_stamp_length": 50,
            "google_analytics_id": GOOGLE_ANALYTICS_ID,
            "support_contact_email": SUPPORT_CONTACT_EMAIL,
            "preprocess_default_face_crop": PREPROCESS_DEFAULT_FACE_CROP,
            "preprocess_default_face_padding": PREPROCESS_DEFAULT_FACE_PADDING,
            "preprocess_default_auto_downsize": PREPROCESS_DEFAULT_AUTO_DOWNSIZE,
            "max_preprocessed_size_mb": MAX_FILE_SIZE_MB,
        }
    })


@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    """Serve favicon.ico - prevents 404 errors in browser console."""
    return FileResponse("frontend/static/favicon.png", media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
