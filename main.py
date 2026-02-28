"""
3D Coaster Generator Web Application
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
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

import aiohttp
import trimesh
import vtracer
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Request, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

# Auth and billing
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
)
from quota_service import check_quota, consume_quota, PAID_MONTHLY_LIMIT

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
logger.info("Starting 3D Coaster Generator")
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
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

logger.info(f"TEMP_DIR: {TEMP_DIR}")
logger.info(f"BFL_API_URL: {BFL_API_URL}")
logger.info(f"Rate limiting: {RATE_LIMIT_ENABLED} ({RATE_LIMIT_COOLDOWN_HOURS}h cooldown)")
logger.info(f"Bypass with API key: {ALLOW_BYPASS_WITH_API_KEY}")
logger.info(f"Legacy weekly limiter enabled: {LEGACY_RATE_LIMIT_ENABLED}")

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
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:3000")
if PUBLIC_BASE_URL.endswith("/"):
    PUBLIC_BASE_URL = PUBLIC_BASE_URL[:-1]

logger.info(f"Session enabled: {bool(SESSION_SECRET)}")
logger.info(f"Google OAuth enabled: {bool(OAUTH_GOOGLE_CLIENT_ID)}")
logger.info(f"Dodo Payments enabled: {bool(DODO_API_KEY)}")

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
    params: Optional['ProcessRequest'] = None  # Store params for confirmation step
    api_key: Optional[str] = None  # User-provided BFL API key (optional)
    stamp_text: str = "Abhishek Does Stuff"  # Text to display on coaster stamp
    
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
            "params": self.params.dict() if self.params else None,
            "api_key": self.api_key,
            "stamp_text": self.stamp_text
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
            api_key=data.get("api_key"),
            stamp_text=data.get("stamp_text", "Abhishek Does Stuff")
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
    
    @classmethod
    def save_job(cls, job: Job) -> None:
        """Save job to disk with file locking."""
        job_path = cls._get_job_path(job.job_id)
        try:
            with open(job_path, 'w') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(job.to_dict(), f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            logger.debug(f"Job {job.job_id} saved to disk")
        except Exception as e:
            logger.error(f"Failed to save job {job.job_id}: {e}")
    
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
    def cleanup_old_jobs(cls, max_age_hours: int = 24) -> None:
        """Clean up job files older than specified hours."""
        try:
            now = datetime.now()
            for filename in os.listdir(JOBS_DIR):
                if filename.endswith('.json') or filename.endswith('_preview.png'):
                    filepath = os.path.join(JOBS_DIR, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    age_hours = (now - file_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old job file: {filename}")
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
    diameter: float = 100.0
    thickness: float = 5.0
    logo_depth: float = 0.6
    scale: float = 0.85
    flip_horizontal: bool = True
    top_rotate: int = 0
    bottom_rotate: int = 0


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
    
    # Get device fingerprint from header
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

# Initialize FastAPI app with optimized settings
app = FastAPI(
    title="3D Coaster Generator",
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
        client_kwargs={"scope": "openid email profile"},
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
        "script-src 'self' 'unsafe-inline' cdn.tailwindcss.com cdnjs.cloudflare.com unpkg.com threejs.org; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "img-src 'self' blob: data: https://lh3.googleusercontent.com; "
        "connect-src 'self' https://api.bfl.ai https://auth.bfl.ai; "
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


# ============= OAUTH LOGIN ROUTES =============

@app.get("/auth/login/google")
async def login_google(request: Request):
    """Redirect to Google OAuth."""
    if "google" not in enabled_providers:
        raise HTTPException(status_code=404, detail="Google login is not configured")

    return await oauth.google.authorize_redirect(
        request,
        f"{PUBLIC_BASE_URL}/auth/callback/google",
    )


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

    if is_webhook_processed(webhook_id):
        return {"received": True}

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
    ip_header = request.headers.get("X-Forwarded-For")
    
    allowed, message, retry_after, usage_info = await check_quota(
        fingerprint, user_id, ip_header
    )
    
    return {
        "authenticated": bool(user_id),
        "quota_exhausted": not allowed,
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
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    elif request.client:
        return request.client.host
    return "unknown"


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
                raise Exception(f"BFL API error: {response.status} - {text}")
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
                raise Exception(f"BFL job failed: {poll_result.get('error', 'Unknown error')}")
            
            # Continue polling for Pending, Processing, etc.
    
    raise Exception(f"BFL job timed out after {MAX_POLLING_ATTEMPTS * POLLING_INTERVAL} seconds")


def vectorize_image(image_bytes: bytes) -> str:
    """
    Convert PNG image to SVG using vtracer.
    
    Args:
        image_bytes: PNG image bytes
    
    Returns:
        SVG string content
    """
    logger.info("="*60)
    logger.info("VECTORIZE IMAGE - Starting vectorization")
    logger.info(f"Input image size: {len(image_bytes)} bytes")
    
    # BFL now returns PNG directly (output_format="png"), no conversion needed
    # vtracer requires PNG format, so this works perfectly
    
    # Save image to temp file (vtracer needs file path)
    # Use absolute paths to avoid issues with vtracer (Rust library)
    temp_png_path = os.path.abspath(os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}.png"))
    temp_svg_path = temp_png_path.replace(".png", ".svg")
    
    logger.info(f"Temp PNG path: {temp_png_path}")
    logger.info(f"Temp SVG path: {temp_svg_path}")
    
    try:
        # Save PNG bytes directly (already PNG from BFL)
        logger.debug("Writing PNG image to temp file...")
        with open(temp_png_path, "wb") as f:
            f.write(image_bytes)
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Saved temp PNG: {os.path.getsize(temp_png_path)} bytes")
        
        # Verify file exists
        if not os.path.exists(temp_png_path):
            logger.error(f"Temp PNG file does not exist after write!")
            raise Exception("Failed to write temp PNG file")
        logger.debug(f"Temp file exists: {os.path.exists(temp_png_path)}")
        logger.debug(f"Temp file size: {os.path.getsize(temp_png_path)} bytes")
        logger.debug(f"Absolute path: {os.path.abspath(temp_png_path)}")
        
        # Convert to SVG using vtracer
        logger.info("Calling vtracer.convert_image_to_svg_py...")
        logger.debug(f"Input file (absolute): {temp_png_path}")
        logger.debug(f"Output file: {temp_svg_path}")
        vtracer.convert_image_to_svg_py(
            temp_png_path,
            temp_svg_path,
            colormode="binary",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=4,
            color_precision=6,
            layer_difference=0,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3
        )
        logger.info("vtracer conversion completed!")
        
        # Check if SVG was created
        if not os.path.exists(temp_svg_path):
            logger.error(f"SVG file was not created at: {temp_svg_path}")
            raise Exception("SVG file was not created by vtracer")
        logger.info(f"SVG file created: {os.path.getsize(temp_svg_path)} bytes")
        
        # Read SVG content
        logger.debug("Reading SVG content...")
        with open(temp_svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        logger.info(f"Read SVG content: {len(svg_content)} chars")
        
        return svg_content
        
    except Exception as e:
        logger.exception("Error in vectorize_image:")
        raise
    finally:
        # Cleanup temp files (disabled in debug mode)
        if DEBUG_NO_CLEANUP:
            logger.warning(f"DEBUG MODE: Preserving temp files:")
            logger.warning(f"  - PNG: {temp_png_path}")
            logger.warning(f"  - SVG: {temp_svg_path}")
        else:
            logger.debug("Cleaning up temp files...")
            if os.path.exists(temp_png_path):
                os.remove(temp_png_path)
                logger.debug(f"Deleted temp PNG: {temp_png_path}")
            if os.path.exists(temp_svg_path):
                os.remove(temp_svg_path)
                logger.debug(f"Deleted temp SVG: {temp_svg_path}")
            logger.info("Vectorization cleanup complete")


def generate_3d_coaster(
    svg_string: str,
    params: ProcessRequest,
    job_id: str
) -> tuple[str, str, str]:
    """
    Generate 3D coaster files from SVG.
    
    Args:
        svg_string: SVG content as string
        params: Coaster parameters
        job_id: Job identifier
    
    Returns:
        Tuple of (3mf_path, body_stl_path, logos_stl_path)
    """
    logger.info("="*60)
    logger.info("GENERATE 3D COASTER - Starting 3D generation")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Parameters: diameter={params.diameter}, thickness={params.thickness}, "
                f"logo_depth={params.logo_depth}, scale={params.scale}, "
                f"flip_horizontal={params.flip_horizontal}, "
                f"top_rotate={params.top_rotate}, bottom_rotate={params.bottom_rotate}")
    logger.info(f"SVG string length: {len(svg_string)} chars")
    
    timestamp = datetime.now().strftime("%H%M%S")
    base_name = f"{job_id}_{timestamp}"
    logger.info(f"Base filename: {base_name}")
    
    # Define output path for 3MF
    output_3mf_path = os.path.join(TEMP_DIR, f"{base_name}_coaster.3mf")
    logger.debug(f"Output path: {output_3mf_path}")
    
    # Create base cylinder
    logger.debug(f"Creating base cylinder: radius={params.diameter/2}, height={params.thickness}")
    base = trimesh.creation.cylinder(
        radius=params.diameter / 2,
        height=params.thickness,
        sections=120
    )
    logger.info(f"Base cylinder created: {len(base.vertices)} vertices, {len(base.faces)} faces")
    
    # Load SVG paths using BytesIO
    logger.debug("Loading SVG from BytesIO...")
    svg_bytes = svg_string.encode('utf-8')
    try:
        path_obj = trimesh.load_path(io.BytesIO(svg_bytes), file_type='svg')
        logger.info(f"SVG loaded successfully")
        logger.debug(f"Path object type: {type(path_obj)}")
    except Exception as e:
        logger.error(f"Failed to load SVG: {e}")
        raise Exception(f"Failed to load SVG: {str(e)}")
    
    # Get polygons from paths
    logger.debug("Extracting polygons from SVG paths...")
    polygons = []
    if hasattr(path_obj, 'polygons_full') and path_obj.polygons_full:
        polygons = path_obj.polygons_full
        logger.info(f"Found {len(polygons)} polygons_full")
    elif hasattr(path_obj, 'polygons_closed') and path_obj.polygons_closed:
        polygons = path_obj.polygons_closed
        logger.info(f"Found {len(polygons)} polygons_closed")
    else:
        logger.error("No polygons_full or polygons_closed found in path_obj")
        logger.debug(f"Path object attributes: {dir(path_obj)}")
    
    if not polygons:
        raise Exception("No valid polygons found in SVG")
    
    # Process polygons to handle holes (subtract contained polygons)
    logger.info(f"Initial polygons: {len(polygons)}")
    
    # Filter out empty and compute areas
    polys = [p for p in polygons if not p.is_empty]
    polys_sorted = sorted(polys, key=lambda p: abs(p.area), reverse=True)
    
    # Drop only true canvas-sized background artifacts from tracing.
    # The previous "drop largest polygon" heuristic can remove real face geometry.
    if polys_sorted:
        global_min_x = min(p.bounds[0] for p in polys_sorted)
        global_min_y = min(p.bounds[1] for p in polys_sorted)
        global_max_x = max(p.bounds[2] for p in polys_sorted)
        global_max_y = max(p.bounds[3] for p in polys_sorted)
        global_w = max(global_max_x - global_min_x, 1e-9)
        global_h = max(global_max_y - global_min_y, 1e-9)

        filtered_polys = []
        dropped_count = 0
        for poly in polys_sorted:
            min_x, min_y, max_x, max_y = poly.bounds
            poly_w = max_x - min_x
            poly_h = max_y - min_y
            bbox_area = max(poly_w * poly_h, 1e-9)
            area_ratio_in_bbox = abs(poly.area) / bbox_area
            coverage_x = poly_w / global_w
            coverage_y = poly_h / global_h

            # vtracer sometimes emits a canvas-sized ring/solid path that is not part
            # of printable logo geometry.
            is_canvas_sized = coverage_x >= 0.995 and coverage_y >= 0.995
            is_ring_like = area_ratio_in_bbox < 0.20
            is_solid_canvas = len(poly.interiors) == 0 and area_ratio_in_bbox > 0.85

            if is_canvas_sized and (is_ring_like or is_solid_canvas):
                dropped_count += 1
                logger.info(
                    "Dropping canvas-sized background polygon: "
                    f"area={abs(poly.area):.2f}, "
                    f"coverage=({coverage_x:.3f},{coverage_y:.3f}), "
                    f"area_ratio={area_ratio_in_bbox:.3f}, "
                    f"holes={len(poly.interiors)}"
                )
                continue

            filtered_polys.append(poly)

        polys_sorted = filtered_polys
        if dropped_count:
            logger.info(f"Dropped {dropped_count} background polygon(s)")

    # Subtraction logic to preserve holes
    used = [False] * len(polys_sorted)
    processed_polys = []

    for i, outer in enumerate(polys_sorted):
        if used[i]:
            continue

        holes = []
        for j in range(i + 1, len(polys_sorted)):
            if used[j]:
                continue
            inner = polys_sorted[j]
            # Give a small buffer to handle floating point imprecision
            if outer.buffer(1e-5).contains(inner):
                holes.append(inner)
                used[j] = True

        if holes:
            hole_union = unary_union(holes)
            carved = outer.difference(hole_union)
            if carved.is_empty:
                continue
            if carved.geom_type == "Polygon":
                processed_polys.append(carved)
            elif carved.geom_type == "MultiPolygon":
                processed_polys.extend(list(carved.geoms))
        else:
            processed_polys.append(outer)

    logger.info(f"Extruding {len(processed_polys)} processed polygons (holes preserved)...")
    logo_meshes = []
    target_size = params.diameter * params.scale
    
    for i, poly in enumerate(processed_polys):
        try:
            # Extrude the polygon
            extruded = trimesh.creation.extrude_polygon(
                poly.buffer(0),
                height=params.logo_depth
            )
            logo_meshes.append(extruded)
            logger.debug(f"Extruded polygon {i}: {len(extruded.vertices)} vertices")
        except Exception as e:
            logger.warning(f"Failed to extrude polygon {i}: {e}")
            continue
    
    if not logo_meshes:
        raise Exception("No valid logo meshes could be created")
    
    logger.info(f"Successfully extruded {len(logo_meshes)} polygons")
    
    # Combine all logo meshes
    logger.debug("Combining logo meshes...")
    logos_combined = trimesh.util.concatenate(logo_meshes)
    logger.info(f"Combined logos: {len(logos_combined.vertices)} vertices")
    
    # Calculate scaling
    bounds = logos_combined.bounds
    current_size_x = bounds[1][0] - bounds[0][0]
    current_size_y = bounds[1][1] - bounds[0][1]
    current_size = max(current_size_x, current_size_y)
    logger.info(f"Current logo size: {current_size:.2f}mm (x: {current_size_x:.2f}, y: {current_size_y:.2f})")
    
    if current_size == 0:
        raise Exception("Invalid logo bounds: zero size")
    
    scale_factor = target_size / current_size
    mirror_x = -1 if params.flip_horizontal else 1
    logger.info(f"Scale factor: {scale_factor:.4f}, Mirror X: {mirror_x}")
    
    # Apply scaling transformation
    logger.debug("Applying scale and mirror transformation...")
    matrix = np.eye(4)
    matrix[0, 0] *= (mirror_x * target_size / current_size)
    matrix[1, 1] *= (target_size / current_size)
    
    logos_combined.apply_transform(matrix)
    logger.info("Scale/mirror transformation applied")
    
    # Center (Visual Bounding Box Center)
    logger.debug("Centering logo...")
    new_bounds = logos_combined.bounds
    center_x = (new_bounds[0][0] + new_bounds[1][0]) / 2
    center_y = (new_bounds[0][1] + new_bounds[1][1]) / 2
    logger.info(f"Center: ({center_x:.2f}, {center_y:.2f})")
    
    trans = np.eye(4)
    trans[0, 3] = -center_x
    trans[1, 3] = -center_y
    logos_combined.apply_transform(trans)
    logger.info("Logo centered")

    # Flush Positioning
    logger.info("Creating positioned logos...")
    
    # Top Logo
    logger.debug("Creating top logo...")
    top_logo = logos_combined.copy()
    if params.top_rotate != 0:
        logger.debug(f"Rotating top logo by {params.top_rotate} degrees")
        rz = trimesh.transformations.rotation_matrix(np.radians(params.top_rotate), [0, 0, 1])
        top_logo.apply_transform(rz)
        
    top_z = (params.thickness / 2) - params.logo_depth
    logger.info(f"Top logo Z position: {top_z:.2f}")
    top_logo.apply_translation([0, 0, top_z])
    
    # Bottom Logo
    logger.debug("Creating bottom logo...")
    bottom_logo = logos_combined.copy()
    if params.bottom_rotate != 0:
        logger.debug(f"Rotating bottom logo by {params.bottom_rotate} degrees")
        rz = trimesh.transformations.rotation_matrix(np.radians(params.bottom_rotate), [0, 0, 1])
        bottom_logo.apply_transform(rz)
        
    logger.debug("Applying X-axis flip (180 degrees)...")
    rx = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    bottom_logo.apply_transform(rx)
    
    bot_z = (-params.thickness / 2) + params.logo_depth
    logger.info(f"Bottom logo Z position: {bot_z:.2f}")
    bottom_logo.apply_translation([0, 0, bot_z])

    # Combine all logos
    logger.debug("Combining top and bottom logos...")
    final_logos = trimesh.util.concatenate([top_logo, bottom_logo])
    logger.info(f"Final combined logos: {len(final_logos.vertices)} vertices")
    
    # Export both 3MF (for download) and STLs (for viewer)
    logger.info(f"Exporting 3MF file and STL files...")
    
    # Define STL paths (for viewer)
    body_stl_path = os.path.join(TEMP_DIR, f"{base_name}_Body.stl")
    logos_stl_path = os.path.join(TEMP_DIR, f"{base_name}_Logos.stl")
    
    # Validate meshes before export
    logger.info(f"Body mesh: {len(base.vertices)} vertices, {len(base.faces)} faces")
    logger.info(f"Logos mesh: {len(final_logos.vertices)} vertices, {len(final_logos.faces)} faces")
    
    if len(base.vertices) == 0 or len(final_logos.vertices) == 0:
        raise Exception("Generated meshes are empty - cannot export")
    
    # Export STLs for viewer
    base.export(body_stl_path)
    final_logos.export(logos_stl_path)
    logger.debug(f"STL files exported for viewer")
    
    # Export 3MF with proper build section for slicer compatibility
    logger.debug(f"Output 3MF: {output_3mf_path}")
    
    try:
        # Create 3MF file manually with proper build section
        import zipfile
        import uuid
        
        # Create XML content manually with proper format
        xml_content = ['<?xml version="1.0" encoding="utf-8"?>']
        xml_content.append('<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" unit="millimeter" xml:lang="en-US">')
        xml_content.append('  <resources>')
        
        # Add body object
        body_id = 1
        xml_content.append(f'    <object id="{body_id}" name="coaster_body" type="model">')
        xml_content.append('      <mesh>')
        xml_content.append('        <vertices>')
        
        # Write body vertices
        for vertex in base.vertices:
            xml_content.append(f'          <vertex x="{vertex[0]}" y="{vertex[1]}" z="{vertex[2]}" />')
        
        xml_content.append('        </vertices>')
        xml_content.append('        <triangles>')
        
        # Write body faces (triangles)
        for face in base.faces:
            xml_content.append(f'          <triangle v1="{face[0]}" v2="{face[1]}" v3="{face[2]}" />')
        
        xml_content.append('        </triangles>')
        xml_content.append('      </mesh>')
        xml_content.append('    </object>')
        
        # Add logos object
        logos_id = 2
        xml_content.append(f'    <object id="{logos_id}" name="coaster_logos" type="model">')
        xml_content.append('      <mesh>')
        xml_content.append('        <vertices>')
        
        # Write logos vertices
        for vertex in final_logos.vertices:
            xml_content.append(f'          <vertex x="{vertex[0]}" y="{vertex[1]}" z="{vertex[2]}" />')
        
        xml_content.append('        </vertices>')
        xml_content.append('        <triangles>')
        
        # Write logos faces (triangles)
        for face in final_logos.faces:
            xml_content.append(f'          <triangle v1="{face[0]}" v2="{face[1]}" v3="{face[2]}" />')
        
        xml_content.append('        </triangles>')
        xml_content.append('      </mesh>')
        xml_content.append('    </object>')

        # Add composite object with two components
        composite_id = 3
        xml_content.append(f'    <object id="{composite_id}" name="coaster" type="model">')
        xml_content.append('      <components>')
        xml_content.append(f'        <component objectid="{body_id}" />')
        xml_content.append(f'        <component objectid="{logos_id}" />')
        xml_content.append('      </components>')
        xml_content.append('    </object>')
        
        xml_content.append('  </resources>')
        xml_content.append('  <build>')
        xml_content.append(f'    <item objectid="{composite_id}" />')
        xml_content.append('  </build>')
        xml_content.append('</model>')
        
        # Create 3MF zip file
        with zipfile.ZipFile(output_3mf_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add the model file
            zf.writestr('3D/3dmodel.model', '\n'.join(xml_content))
            
            # Add required _rels/.rels file
            rels_content = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>'''
            zf.writestr('_rels/.rels', rels_content)
            
            # Add [Content_Types].xml
            content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
</Types>'''
            zf.writestr('[Content_Types].xml', content_types)
        
        file_size = os.path.getsize(output_3mf_path)
        logger.info(f"3MF exported: {file_size} bytes ({file_size/1024:.1f} KB)")
        
    except Exception as e:
        logger.error(f"Failed to export 3MF: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to export 3MF: {str(e)}")

    # Save SVG for debugging
    if DEBUG_NO_CLEANUP:
        debug_svg_path = os.path.join(TEMP_DIR, f"{base_name}_debug.svg")
        with open(debug_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        logger.info(f"Debug SVG saved: {debug_svg_path}")

    logger.info("="*60)
    logger.info("3D COASTER GENERATION COMPLETE")
    logger.info(f"Files: 3MF={output_3mf_path}, Body={body_stl_path}, Logos={logos_stl_path}")

    # Return all three paths - 3MF for download, STLs for viewer
    return output_3mf_path, body_stl_path, logos_stl_path


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
        logger.error("="*60)
        logger.error(f"JOB {job_id} FAILED!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception("Full traceback:")
        logger.error("="*60)
        
        job = JobStore.get_job(job_id)
        if job:
            job.status = "failed"
            job.progress = 0
            job.message = f"Error: {str(e)}"
            job.error = str(e)
            JobStore.save_job(job)


async def process_vectorization_3d(job_id: str):
    """
    Background task - Phase 2: Vectorization and 3D generation (after confirmation).
    
    Args:
        job_id: Job identifier
    """
    logger.info("="*60)
    logger.info(f"PHASE 2 STARTED - Job ID: {job_id}")
    
    try:
        job = JobStore.get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Load preview image from disk
        flattened_image = JobStore.get_preview_image(job_id)
        params = job.params
        
        if not flattened_image or not params:
            raise Exception("Missing preview image or parameters")
        
        # Step 2: Vectorize
        logger.info("STEP 2/2: Converting to vector (SVG)...")
        await update_job_status(job_id, "processing_vectorize", 75, "Converting to vector...")
        logger.info("Calling vectorize_image...")
        svg_string = vectorize_image(flattened_image)
        logger.info(f"✓ Vectorization complete: {len(svg_string)} chars")
        
        # Step 3: Generate 3D
        logger.info("STEP 2/2: Generating 3D model...")
        await update_job_status(job_id, "processing_3d", 90, "Generating 3D model...")
        logger.info("Calling generate_3d_coaster...")
        coaster_3mf_path, body_stl_path, logos_stl_path = generate_3d_coaster(svg_string, params, job_id)
        logger.info(f"✓ 3D generation complete")
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
        logger.error("="*60)
        logger.error(f"JOB {job_id} FAILED in Phase 2!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception("Full traceback:")
        logger.error("="*60)
        
        job = JobStore.get_job(job_id)
        if job:
            job.status = "failed"
            job.progress = 0
            job.message = f"Error: {str(e)}"
            job.error = str(e)
            JobStore.save_job(job)


@app.post("/api/process")
async def process_image(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    diameter: float = Form(100.0),
    thickness: float = Form(5.0),
    logo_depth: float = Form(0.6),
    scale: float = Form(0.85),
    flip_horizontal: bool = Form(True),
    top_rotate: int = Form(0),
    bottom_rotate: int = Form(0),
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
                f"bottom_rotate={bottom_rotate}")

    # Validate stamp text
    stamp_text = validate_stamp_text(stamp_text)
    logger.info(f"Stamp text validated: {stamp_text}")

    # Resolve identity context
    user_id = request.session.get("user_id")
    device_fingerprint = request.headers.get("X-Device-Fingerprint")
    client_ip = get_client_ip(request)

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
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": message,
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
    image_bytes = await image.read()

    # Check file size
    max_size = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_size:
        logger.error(f"File too large: {len(image_bytes)} bytes (max {max_size})")
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    logger.info(f"✓ Image read: {len(image_bytes)} bytes")

    if len(image_bytes) == 0:
        logger.error("Empty image file received")
        raise HTTPException(status_code=400, detail="Empty image file")

    # Quota checks (authoritative product limits)
    quota_allowed, quota_message, _, usage_info = await check_quota(
        device_fingerprint,
        user_id,
        client_ip,
    )
    if not quota_allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": usage_info.get("message") or quota_message,
                "next_action": usage_info.get("next_action"),
                "usage": usage_info,
            },
        )

    # Create job (don't store API key in job for security)
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, stamp_text=stamp_text)
    JobStore.save_job(job)
    logger.info(f"✓ Job created: {job_id} with stamp: {stamp_text}")

    # Consume quota after job creation
    quota_bucket = usage_info.get("bucket")
    if not quota_bucket:
        raise HTTPException(status_code=500, detail="Quota decision failed")
    quota_event_id = await consume_quota(job_id, device_fingerprint, user_id, quota_bucket)
    if not quota_event_id:
        raise HTTPException(status_code=500, detail="Unable to reserve quota at this time")

    # Use provided API key or env var
    effective_api_key = api_key if api_key else os.environ.get("BFL_API_KEY")
    if not effective_api_key:
        raise HTTPException(status_code=500, detail="No API key configured")

    # Create parameters object
    params = ProcessRequest(
        diameter=diameter,
        thickness=thickness,
        logo_depth=logo_depth,
        scale=scale,
        flip_horizontal=flip_horizontal,
        top_rotate=top_rotate,
        bottom_rotate=bottom_rotate
    )
    logger.info("✓ Parameters object created")

    # Start background processing (pass API key separately, not stored)
    logger.info("Starting background processing task...")
    background_tasks.add_task(process_coaster_job, job_id, image_bytes, params, stamp_text, effective_api_key)
    logger.info(f"✓✓✓ Job {job_id} queued successfully")
    logger.info("="*60)
    
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Get the current status of a job."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
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
async def download_coaster(job_id: str):
    """Download the combined coaster 3MF file."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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
async def download_body_stl(job_id: str):
    """Download the coaster body STL file (for viewer)."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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
async def download_logos_stl(job_id: str):
    """Download the coaster logos STL file (for viewer)."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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
async def get_preview_image(job_id: str):
    """Get the BFL generated image for review (before confirmation)."""
    from starlette.responses import StreamingResponse
    
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
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
async def confirm_job(job_id: str, background_tasks: BackgroundTasks):
    """Confirm the generated image and proceed with vectorization and 3D generation."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "review":
        raise HTTPException(status_code=400, detail="Job is not in review state")
    
    logger.info(f"Job {job_id} confirmed by user, starting Phase 2...")
    
    # Start Phase 2 (vectorization + 3D) as background task
    background_tasks.add_task(process_vectorization_3d, job_id)
    
    return {"job_id": job_id, "status": "processing_vectorize", "message": "Processing vectorization and 3D generation..."}


@app.post("/api/retry/{job_id}")
async def retry_job(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    diameter: float = Form(100.0),
    thickness: float = Form(5.0),
    logo_depth: float = Form(0.6),
    scale: float = Form(0.85),
    flip_horizontal: bool = Form(True),
    top_rotate: int = Form(0),
    bottom_rotate: int = Form(0)
):
    """Retry with a different image, keeping the same job ID."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "review":
        raise HTTPException(status_code=400, detail="Job is not in review state")
    
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image bytes
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    user_id = request.session.get("user_id")
    device_fingerprint = request.headers.get("X-Device-Fingerprint")
    client_ip = get_client_ip(request)

    quota_allowed, quota_message, _, usage_info = await check_quota(
        device_fingerprint,
        user_id,
        client_ip,
    )
    if not quota_allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": usage_info.get("message") or quota_message,
                "next_action": usage_info.get("next_action"),
                "usage": usage_info,
            },
        )
    
    # Reset job state
    job.status = "pending"
    job.progress = 0
    job.message = "Restarting with new image..."
    job.preview_image_path = None
    job.error = None
    JobStore.save_job(job)

    quota_bucket = usage_info.get("bucket")
    if not quota_bucket:
        raise HTTPException(status_code=500, detail="Quota decision failed")
    quota_event_id = await consume_quota(job_id, device_fingerprint, user_id, quota_bucket)
    if not quota_event_id:
        raise HTTPException(status_code=500, detail="Unable to reserve quota at this time")
    
    # Create parameters object
    params = ProcessRequest(
        diameter=diameter,
        thickness=thickness,
        logo_depth=logo_depth,
        scale=scale,
        flip_horizontal=flip_horizontal,
        top_rotate=top_rotate,
        bottom_rotate=bottom_rotate
    )
    
    # Start background processing with required args
    effective_api_key = os.environ.get("BFL_API_KEY")
    if not effective_api_key:
        raise HTTPException(status_code=500, detail="No API key configured")

    stamp_text = job.stamp_text or "Abhishek Does Stuff"
    background_tasks.add_task(process_coaster_job, job_id, image_bytes, params, stamp_text, effective_api_key)
    
    return {"job_id": job_id, "status": "processing", "message": "Restarting with new image..."}


@app.get("/", response_class=HTMLResponse)
async def get_frontend(request: Request):
    """Serve the HTML frontend using Jinja2 template."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": {
            "app_name": "3D Coaster Generator",
            "default_stamp": "Abhishek Does Stuff",
            "max_stamp_length": 50
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
