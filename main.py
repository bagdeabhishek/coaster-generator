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
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

import aiohttp
import trimesh
import vtracer
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

logger.info(f"TEMP_DIR: {TEMP_DIR}")
logger.info(f"BFL_API_URL: {BFL_API_URL}")

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
            "api_key": self.api_key
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
            api_key=data.get("api_key")
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


# Initialize FastAPI app with optimized settings
app = FastAPI(
    title="3D Coaster Generator",
    version="1.0.0",
    docs_url="/docs" if DEBUG_NO_CLEANUP else None,
    redoc_url="/redoc" if DEBUG_NO_CLEANUP else None,
)


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


async def update_job_status(job_id: str, status: str, progress: int, message: str):
    """Update job status on disk."""
    job = JobStore.get_job(job_id)
    if job:
        job.status = status
        job.progress = progress
        job.message = message
        JobStore.save_job(job)


async def bfl_flux_process(image_bytes: bytes, api_key: str) -> bytes:
    """
    Process image through BFL FLUX API to create flat vector illustration.
    Optimized to use aiohttp for true async HTTP requests.
    
    Args:
        image_bytes: Raw image bytes
        api_key: BFL API key
    
    Returns:
        Processed PNG image bytes
    """
    logger.info("="*60)
    logger.info("BFL FLUX PROCESS - Starting image processing")
    logger.info(f"Input image size: {len(image_bytes)} bytes")
    
    # Convert image to base64 (raw base64 string, not data URL)
    logger.debug("Converting image to base64...")
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    logger.debug(f"Base64 encoded length: {len(image_base64)} chars")
    
    # Load prompt from file
    prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    logger.debug(f"Loading prompt from: {prompt_file_path}")
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        logger.info(f"✓ Prompt loaded from file: {len(prompt)} chars")
    except Exception as e:
        logger.error(f"Failed to load prompt from file: {e}")
        logger.warning("Using fallback prompt")
        prompt = "flat vector illustration, solid colors, no gradients, high contrast, 2d cartoon style, white background, clean lines suitable for vector tracing"
    
    # Set parameters based on BFL API documentation
    width = 1024
    height = 1024
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
) -> tuple:
    """
    Generate 3D coaster STL files from SVG.
    
    Args:
        svg_string: SVG content as string
        params: Coaster parameters
        job_id: Job identifier
    
    Returns:
        Tuple of (body_stl_path, logos_stl_path, preview_path)
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
    
    # Define output paths
    body_stl_path = os.path.join(TEMP_DIR, f"{base_name}_Body.stl")
    logos_stl_path = os.path.join(TEMP_DIR, f"{base_name}_Logos.stl")
    logger.debug(f"Output paths: Body={body_stl_path}, Logos={logos_stl_path}")
    
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
    
    # Extrude polygons to create logo meshes
    logger.info(f"Extruding {len(polygons)} polygons...")
    logo_meshes = []
    target_size = params.diameter * params.scale
    
    for i, poly in enumerate(polygons):
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
    
    # Export STL files
    logger.info(f"Exporting STL files...")
    logger.debug(f"Body STL: {body_stl_path}")
    logger.debug(f"Logos STL: {logos_stl_path}")
    
    base.export(body_stl_path)
    body_size = os.path.getsize(body_stl_path)
    logger.info(f"Body STL exported: {body_size} bytes ({body_size/1024:.1f} KB)")
    
    final_logos.export(logos_stl_path)
    logos_size = os.path.getsize(logos_stl_path)
    logger.info(f"Logos STL exported: {logos_size} bytes ({logos_size/1024:.1f} KB)")
    
    # Save SVG for debugging
    if DEBUG_NO_CLEANUP:
        debug_svg_path = os.path.join(TEMP_DIR, f"{base_name}_debug.svg")
        with open(debug_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        logger.info(f"Debug SVG saved: {debug_svg_path}")
    
    # Generate preview image
    logger.info("Generating preview image...")
    preview_path = generate_preview(base, final_logos, base_name)
    logger.info(f"Preview generated: {preview_path}")
    
    logger.info("="*60)
    logger.info("3D COASTER GENERATION COMPLETE")
    logger.info(f"Files: Body={body_stl_path}, Logos={logos_stl_path}, Preview={preview_path}")
    
    return body_stl_path, logos_stl_path, preview_path


def generate_preview(base: trimesh.Trimesh, logos: trimesh.Trimesh, base_name: str) -> str:
    """
    Generate preview image showing 3D coaster.
    
    Args:
        base: Base cylinder mesh
        logos: Logo meshes
        base_name: Base filename
    
    Returns:
        Path to preview image
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Side view
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot base
    base_faces = base.faces
    base_verts = base.vertices
    ax1.plot_trisurf(base_verts[:, 0], base_verts[:, 1], base_verts[:, 2],
                     triangles=base_faces, alpha=0.3, color='gray', edgecolor='none')
    
    # Plot logos
    logo_faces = logos.faces
    logo_verts = logos.vertices
    ax1.plot_trisurf(logo_verts[:, 0], logo_verts[:, 1], logo_verts[:, 2],
                     triangles=logo_faces, alpha=0.9, color='red', edgecolor='none')
    
    ax1.set_title("Side View")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=0, azim=0)
    
    # Top view
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot base
    ax2.plot_trisurf(base_verts[:, 0], base_verts[:, 1], base_verts[:, 2],
                     triangles=base_faces, alpha=0.3, color='gray', edgecolor='none')
    
    # Plot logos
    ax2.plot_trisurf(logo_verts[:, 0], logo_verts[:, 1], logo_verts[:, 2],
                     triangles=logo_faces, alpha=0.9, color='red', edgecolor='none')
    
    ax2.set_title("Top View")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=90, azim=-90)
    
    plt.tight_layout()
    
    preview_path = os.path.join(TEMP_DIR, f"{base_name}_preview.png")
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return preview_path


async def process_coaster_job(
    job_id: str,
    image_bytes: bytes,
    params: ProcessRequest
):
    """
    Background task - Phase 1: Process image through BFL and wait for confirmation.
    
    Args:
        job_id: Job identifier
        image_bytes: Uploaded image bytes
        params: Coaster parameters
    """
    logger.info("="*60)
    logger.info(f"PHASE 1 STARTED - Job ID: {job_id}")
    logger.info(f"Input image size: {len(image_bytes)} bytes")
    logger.info(f"Parameters: {params}")
    
    try:
        # Get job object to access stored API key
        job = JobStore.get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Get API key - use job's key if provided, otherwise fall back to env var
        api_key = job.api_key or os.environ.get("BFL_API_KEY")
        logger.info(f"API key source: {'user-provided' if job.api_key else 'environment'}")
        logger.info(f"API key present: {bool(api_key)}")
        if api_key:
            logger.debug(f"API key (first 10 chars): {api_key[:10]}...")
        
        if not api_key:
            logger.error("No BFL API key available! Please provide an API key or set BFL_API_KEY environment variable.")
            raise Exception("No BFL API key available. Please provide your API key in the form or set BFL_API_KEY environment variable.")
        
        # Step 1: Flatten image with BFL FLUX
        logger.info("STEP 1/2: Flattening image with BFL FLUX API...")
        await update_job_status(job_id, "processing_flatten", 50, "Flattening image with AI...")
        logger.info("Calling bfl_flux_process...")
        flattened_image = await bfl_flux_process(image_bytes, api_key)
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
        body_path, logos_path, preview_path = generate_3d_coaster(svg_string, params, job_id)
        logger.info(f"✓ 3D generation complete")
        logger.info(f"  - Body STL: {body_path}")
        logger.info(f"  - Logos STL: {logos_path}")
        logger.info(f"  - Preview: {preview_path}")
        
        # Update job with file paths
        logger.info("Updating job status to completed...")
        job.files = {
            "body": body_path,
            "logos": logos_path,
            "preview": preview_path
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
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    diameter: float = Form(100.0),
    thickness: float = Form(5.0),
    logo_depth: float = Form(0.6),
    scale: float = Form(0.85),
    flip_horizontal: bool = Form(True),
    top_rotate: int = Form(0),
    bottom_rotate: int = Form(0),
    api_key: str = Form("")
):
    """
    Start a new coaster generation job.
    
    Accepts an image file and parameters, returns job ID for tracking.
    If api_key is provided, it will be used instead of the environment variable.
    """
    logger.info("="*60)
    logger.info("API REQUEST: POST /api/process - New coaster job")
    logger.info(f"Image filename: {image.filename}")
    logger.info(f"Image content-type: {image.content_type}")
    logger.info(f"Parameters: diameter={diameter}, thickness={thickness}, "
                f"logo_depth={logo_depth}, scale={scale}, "
                f"flip_horizontal={flip_horizontal}, top_rotate={top_rotate}, "
                f"bottom_rotate={bottom_rotate}")
    logger.info(f"API key provided: {bool(api_key)}")
    
    # Validate file type
    if not image.content_type.startswith("image/"):
        logger.error(f"Invalid content type: {image.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    logger.info("✓ Content type validated")
    
    # Create job
    job_id = str(uuid.uuid4())
    # Use provided API key or None (will fallback to env var later)
    job = Job(job_id=job_id, api_key=api_key if api_key else None)
    JobStore.save_job(job)
    logger.info(f"✓ Job created: {job_id}")
    
    # Read image bytes
    logger.debug("Reading image bytes...")
    image_bytes = await image.read()
    logger.info(f"✓ Image read: {len(image_bytes)} bytes")
    
    if len(image_bytes) == 0:
        logger.error("Empty image file received")
        raise HTTPException(status_code=400, detail="Empty image file")
    
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
    
    # Start background processing
    logger.info("Starting background processing task...")
    background_tasks.add_task(process_coaster_job, job_id, image_bytes, params)
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
            "body": f"/api/download/{job_id}/body",
            "logos": f"/api/download/{job_id}/logos",
            "preview": f"/api/download/{job_id}/preview"
        }
    
    return response


@app.get("/api/download/{job_id}/body")
async def download_body(job_id: str):
    """Download the coaster body STL file."""
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
async def download_logos(job_id: str):
    """Download the coaster logos STL file."""
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


@app.get("/api/download/{job_id}/preview")
async def download_preview(job_id: str):
    """Download the coaster preview PNG file."""
    job = JobStore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed" or "preview" not in job.files:
        raise HTTPException(status_code=400, detail="Preview not available")
    
    file_path = job.files["preview"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=f"coaster_{job_id}_preview.png"
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


# Update StatusResponse to include review status
class StatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: int
    message: str
    download_urls: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    show_preview: bool = False  # Indicates if preview image is available


@app.post("/api/retry/{job_id}")
async def retry_job(
    job_id: str,
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
    
    # Reset job state
    job.status = "pending"
    job.progress = 0
    job.message = "Restarting with new image..."
    job.preview_image_path = None
    job.error = None
    JobStore.save_job(job)
    
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
    
    # Start background processing
    background_tasks.add_task(process_coaster_job, job_id, image_bytes, params)
    
    return {"job_id": job_id, "status": "processing", "message": "Restarting with new image..."}


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the HTML frontend."""
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Coaster Generator</title>
    <!-- Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <!-- STLLoader -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
    <!-- OrbitControls -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .content {
            padding: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        input[type="file"],
        input[type="number"],
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input[type="file"]:focus,
        input[type="number"]:focus,
        select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        .checkbox-group label {
            margin: 0;
            cursor: pointer;
        }
        
        .parameters-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 600px) {
            .parameters-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .parameter-note {
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }
        
        .generate-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        .generate-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* Progress Section */
        .progress-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            display: none;
        }
        
        .progress-section.active {
            display: block;
        }
        
        .progress-bar-container {
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            text-align: center;
            font-weight: 600;
            color: #333;
        }
        
        .status-message {
            text-align: center;
            margin-top: 10px;
            color: #666;
        }
        
        /* Review Section */
        .review-section {
            margin-top: 30px;
            padding: 25px;
            background: #fff3e0;
            border-radius: 8px;
            display: none;
            border: 2px solid #ff9800;
        }
        
        .review-section.active {
            display: block;
        }
        
        .review-section h3 {
            margin-bottom: 15px;
            color: #e65100;
            text-align: center;
            font-size: 1.5em;
        }
        
        .review-image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .review-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .review-actions {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .approve-btn {
            padding: 14px 32px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.2s;
        }
        
        .approve-btn:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        
        .approve-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .retry-btn {
            padding: 14px 32px;
            background: #ff9800;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.2s;
        }
        
        .retry-btn:hover {
            background: #f57c00;
            transform: translateY(-2px);
        }
        
        .retry-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .retry-form {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border: 2px dashed #ff9800;
        }
        
        .retry-form.active {
            display: block;
        }
        
        .retry-form h4 {
            margin-bottom: 15px;
            color: #e65100;
        }
        
        .retry-file-input {
            margin-bottom: 15px;
        }
        
        .retry-submit-btn {
            padding: 12px 24px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .retry-submit-btn:hover {
            background: #1976d2;
        }
        
        .retry-submit-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        /* 3D Viewer Section */
        .viewer-section {
            margin-top: 30px;
            padding: 25px;
            background: #e3f2fd;
            border-radius: 8px;
            display: none;
            border: 2px solid #2196f3;
        }
        
        .viewer-section.active {
            display: block;
        }
        
        .viewer-section h3 {
            margin-bottom: 15px;
            color: #1565c0;
            text-align: center;
            font-size: 1.5em;
        }
        
        .viewer-container {
            width: 100%;
            height: 400px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
            position: relative;
        }
        
        .viewer-controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .viewer-btn {
            padding: 10px 20px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .viewer-btn:hover {
            background: #1976d2;
        }
        
        .viewer-btn.active {
            background: #1565c0;
        }
        
        .viewer-info {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        /* Downloads Section */
        .downloads-section {
            margin-top: 30px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 8px;
            display: none;
        }
        
        .downloads-section.active {
            display: block;
        }
        
        .downloads-section h3 {
            margin-bottom: 15px;
            color: #2e7d32;
        }
        
        .download-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .download-btn {
            padding: 12px 24px;
            background: #4caf50;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            transition: background 0.2s;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        
        .download-btn:hover {
            background: #45a049;
        }
        
        /* Error Section */
        .error-section {
            margin-top: 30px;
            padding: 20px;
            background: #ffebee;
            border-radius: 8px;
            display: none;
        }
        
        .error-section.active {
            display: block;
        }
        
        .error-section h3 {
            color: #c62828;
            margin-bottom: 10px;
        }
        
        .note {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
        }
        
        .note p {
            color: #856404;
            font-size: 0.95em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D Coaster Generator</h1>
            <p>Transform images into 3D printable coasters</p>
        </div>
        
        <div class="content">
            <form id="coasterForm">
                <div class="form-group">
                    <label for="image">Upload Image</label>
                    <input type="file" id="image" name="image" accept="image/*" required>
                    <p class="parameter-note">Recommended: High contrast images with clear subjects</p>
                </div>
                
                <div class="form-group">
                    <label for="api_key">BFL API Key (optional)</label>
                    <input type="password" id="api_key" name="api_key" placeholder="Enter your BFL API key or leave blank to use server key">
                    <p class="parameter-note">Leave empty to use server-provided API key. Enter your own key for personal usage tracking.</p>
                </div>
                
                <div class="parameters-grid">
                    <div class="form-group">
                        <label for="diameter">Diameter (mm)</label>
                        <input type="number" id="diameter" name="diameter" value="100.0" min="50" max="200" step="1">
                    </div>
                    
                    <div class="form-group">
                        <label for="thickness">Thickness (mm)</label>
                        <input type="number" id="thickness" name="thickness" value="5.0" min="2" max="20" step="0.5">
                    </div>
                    
                    <div class="form-group">
                        <label for="logo_depth">Logo Depth (mm)</label>
                        <input type="number" id="logo_depth" name="logo_depth" value="0.6" min="0.1" max="5" step="0.1">
                    </div>
                    
                    <div class="form-group">
                        <label for="scale">Logo Scale</label>
                        <input type="number" id="scale" name="scale" value="0.85" min="0.1" max="1.5" step="0.05">
                        <p class="parameter-note">Relative to coaster diameter</p>
                    </div>
                    
                    <div class="form-group">
                        <label for="top_rotate">Top Rotation (degrees)</label>
                        <input type="number" id="top_rotate" name="top_rotate" value="0" min="0" max="360" step="1">
                    </div>
                    
                    <div class="form-group">
                        <label for="bottom_rotate">Bottom Rotation (degrees)</label>
                        <input type="number" id="bottom_rotate" name="bottom_rotate" value="0" min="0" max="360" step="1">
                    </div>
                </div>
                
                <div class="form-group checkbox-group">
                    <input type="checkbox" id="flip_horizontal" name="flip_horizontal" checked>
                    <label for="flip_horizontal">Flip Horizontal (Mirror)</label>
                </div>
                
                <button type="submit" class="generate-btn" id="generateBtn">Generate 3D Coaster</button>
            </form>
            
            <!-- Progress Section -->
            <div class="progress-section" id="progressSection">
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                <div class="progress-text" id="progressText">0%</div>
                <div class="status-message" id="statusMessage">Initializing...</div>
            </div>
            
            <!-- Review Section -->
            <div class="review-section" id="reviewSection">
                <h3>Review Generated Image</h3>
                <div class="review-image-container">
                    <img id="reviewImage" class="review-image" src="" alt="Generated Preview">
                </div>
                <div class="review-actions">
                    <button class="approve-btn" id="approveBtn">Approve & Continue</button>
                    <button class="retry-btn" id="retryBtn">Try Again</button>
                </div>
                <div class="retry-form" id="retryForm">
                    <h4>Upload a Different Image</h4>
                    <input type="file" id="retryImageInput" class="retry-file-input" accept="image/*">
                    <button class="retry-submit-btn" id="retrySubmitBtn">Submit New Image</button>
                </div>
            </div>
            
            <!-- 3D Viewer Section -->
            <div class="viewer-section" id="viewerSection">
                <h3>3D Preview</h3>
                <div class="viewer-controls">
                    <button class="viewer-btn active" id="viewBodyBtn" data-model="body">View Body</button>
                    <button class="viewer-btn" id="viewLogosBtn" data-model="logos">View Logos</button>
                    <button class="viewer-btn" id="viewBothBtn" data-model="both">View Both</button>
                </div>
                <div class="viewer-container" id="viewerContainer"></div>
                <p class="viewer-info">Click and drag to rotate • Scroll to zoom • Right-click to pan</p>
            </div>
            
            <!-- Downloads Section -->
            <div class="downloads-section" id="downloadsSection">
                <h3>Download Your Files</h3>
                <div class="download-buttons" id="downloadButtons">
                    <a href="#" class="download-btn" id="downloadBody">Download Body STL</a>
                    <a href="#" class="download-btn" id="downloadLogos">Download Logos STL</a>
                    <a href="#" class="download-btn" id="downloadPreview">Download Preview</a>
                </div>
            </div>
            
            <!-- Error Section -->
            <div class="error-section" id="errorSection">
                <h3>Error</h3>
                <p id="errorMessage"></p>
            </div>
            
            <div class="note">
                <p><strong>How it works:</strong> Upload an image and we'll generate a 3D coaster using AI to create a clean black & white design. You'll have a chance to review before the 3D model is created. The workflow involves: Image Generation (AI processing) → Review (your approval) → Vectorization → 3D Modeling → Completion.</p>
                <p style="margin-top: 10px;"><strong>API Key:</strong> You can use your own BFL API key for personal usage tracking, or leave it blank to use the server's key. <a href="https://api.bfl.ai" target="_blank">Get your API key here</a>.</p>
            </div>
        </div>
    </div>

    <script>
        let pollingInterval = null;
        let currentJobId = null;
        let scene = null;
        let camera = null;
        let renderer = null;
        let controls = null;
        let stlLoader = null;
        let currentMeshes = [];
        let downloadUrls = null;
        
        // Initialize Three.js scene
        function init3DViewer() {
            const container = document.getElementById('viewerContainer');
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 0, 150);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(50, 100, 50);
            scene.add(directionalLight);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight2.position.set(-50, -50, 50);
            scene.add(directionalLight2);
            
            // STL Loader
            stlLoader = new THREE.STLLoader();
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize, false);
            
            // Start animation loop
            animate();
        }
        
        function onWindowResize() {
            const container = document.getElementById('viewerContainer');
            if (camera && renderer) {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
        }
        
        function animate() {
            requestAnimationFrame(animate);
            if (controls) {
                controls.update();
            }
            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        }
        
        function clearMeshes() {
            currentMeshes.forEach(mesh => {
                scene.remove(mesh);
                if (mesh.geometry) mesh.geometry.dispose();
                if (mesh.material) mesh.material.dispose();
            });
            currentMeshes = [];
        }
        
        function loadSTL(url, material) {
            return new Promise((resolve, reject) => {
                stlLoader.load(url,
                    function(geometry) {
                        geometry.computeVertexNormals();
                        geometry.center();
                        
                        const mesh = new THREE.Mesh(geometry, material);
                        
                        // Scale to fit view
                        const box = new THREE.Box3().setFromObject(mesh);
                        const size = box.getSize(new THREE.Vector3());
                        const maxDim = Math.max(size.x, size.y, size.z);
                        const scale = 80 / maxDim;
                        mesh.scale.set(scale, scale, scale);
                        
                        scene.add(mesh);
                        currentMeshes.push(mesh);
                        resolve(mesh);
                    },
                    function(xhr) {
                        console.log((xhr.loaded / xhr.total * 100) + '% loaded');
                    },
                    function(error) {
                        console.error('Error loading STL:', error);
                        reject(error);
                    }
                );
            });
        }
        
        async function show3DViewer(urls) {
            downloadUrls = urls;
            document.getElementById('progressSection').classList.remove('active');
            document.getElementById('viewerSection').classList.add('active');
            document.getElementById('downloadsSection').classList.add('active');
            
            // Update download links
            document.getElementById('downloadBody').href = urls.body;
            document.getElementById('downloadLogos').href = urls.logos;
            document.getElementById('downloadPreview').href = urls.preview;
            
            // Initialize viewer if not already done
            if (!renderer) {
                init3DViewer();
            }
            
            // Load both models by default
            await loadBothModels();
        }
        
        async function loadBodyModel() {
            clearMeshes();
            const material = new THREE.MeshPhongMaterial({ 
                color: 0x3498db,
                specular: 0x444444,
                shininess: 60
            });
            try {
                await loadSTL(downloadUrls.body, material);
            } catch (error) {
                console.error('Failed to load body model:', error);
            }
        }
        
        async function loadLogosModel() {
            clearMeshes();
            const material = new THREE.MeshPhongMaterial({ 
                color: 0xe74c3c,
                specular: 0x444444,
                shininess: 60
            });
            try {
                await loadSTL(downloadUrls.logos, material);
            } catch (error) {
                console.error('Failed to load logos model:', error);
            }
        }
        
        async function loadBothModels() {
            clearMeshes();
            
            const bodyMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x3498db,
                specular: 0x444444,
                shininess: 60,
                transparent: true,
                opacity: 0.9
            });
            
            const logosMaterial = new THREE.MeshPhongMaterial({ 
                color: 0xe74c3c,
                specular: 0x444444,
                shininess: 60
            });
            
            try {
                await Promise.all([
                    loadSTL(downloadUrls.body, bodyMaterial),
                    loadSTL(downloadUrls.logos, logosMaterial)
                ]);
            } catch (error) {
                console.error('Failed to load models:', error);
            }
        }
        
        // Viewer button handlers
        document.getElementById('viewBodyBtn').addEventListener('click', async function() {
            setActiveViewerButton(this);
            await loadBodyModel();
        });
        
        document.getElementById('viewLogosBtn').addEventListener('click', async function() {
            setActiveViewerButton(this);
            await loadLogosModel();
        });
        
        document.getElementById('viewBothBtn').addEventListener('click', async function() {
            setActiveViewerButton(this);
            await loadBothModels();
        });
        
        function setActiveViewerButton(btn) {
            document.querySelectorAll('.viewer-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
        
        // Form submission
        document.getElementById('coasterForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Reset UI
            document.getElementById('errorSection').classList.remove('active');
            document.getElementById('downloadsSection').classList.remove('active');
            document.getElementById('reviewSection').classList.remove('active');
            document.getElementById('viewerSection').classList.remove('active');
            document.getElementById('generateBtn').disabled = true;
            document.getElementById('generateBtn').textContent = 'Processing...';
            
            // Clear any existing meshes
            if (scene) {
                clearMeshes();
            }
            
            // Get form data
            const formData = new FormData(e.target);
            
            // Convert checkbox values to boolean strings
            formData.set('flip_horizontal', document.getElementById('flip_horizontal').checked);
            
            try {
                // Start processing
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to start processing');
                }
                
                const data = await response.json();
                currentJobId = data.job_id;
                
                // Show progress section
                document.getElementById('progressSection').classList.add('active');
                
                // Start polling
                startPolling(currentJobId);
                
            } catch (error) {
                showError(error.message);
                resetForm();
            }
        });
        
        function startPolling(jobId) {
            // Clear any existing interval
            if (pollingInterval) {
                clearInterval(pollingInterval);
            }
            
            // Poll immediately
            pollStatus(jobId);
            
            // Then poll every second
            pollingInterval = setInterval(() => pollStatus(jobId), 1000);
        }
        
        async function pollStatus(jobId) {
            try {
                const response = await fetch(`/api/status/${jobId}`);
                
                if (!response.ok) {
                    throw new Error('Failed to get status');
                }
                
                const data = await response.json();
                
                // Update progress
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressText').textContent = data.progress + '%';
                document.getElementById('statusMessage').textContent = data.message;
                
                // Handle different statuses
                if (data.status === 'review') {
                    // Show review section
                    clearInterval(pollingInterval);
                    showReviewSection(jobId);
                } else if (data.status === 'completed') {
                    clearInterval(pollingInterval);
                    show3DViewer(data.download_urls);
                    resetForm();
                } else if (data.status === 'failed') {
                    clearInterval(pollingInterval);
                    showError(data.error || 'Processing failed');
                    resetForm();
                }
                
            } catch (error) {
                console.error('Polling error:', error);
            }
        }
        
        function showReviewSection(jobId) {
            document.getElementById('progressSection').classList.remove('active');
            document.getElementById('reviewSection').classList.add('active');
            
            // Set the review image
            document.getElementById('reviewImage').src = `/api/preview-image/${jobId}?t=${Date.now()}`;
            
            // Enable buttons
            document.getElementById('approveBtn').disabled = false;
            document.getElementById('retryBtn').disabled = false;
            
            // Reset retry form
            document.getElementById('retryForm').classList.remove('active');
            document.getElementById('retryImageInput').value = '';
        }
        
        // Approve button handler
        document.getElementById('approveBtn').addEventListener('click', async function() {
            if (!currentJobId) return;
            
            this.disabled = true;
            document.getElementById('retryBtn').disabled = true;
            this.textContent = 'Processing...';
            
            try {
                const response = await fetch(`/api/confirm/${currentJobId}`, {
                    method: 'POST'
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to confirm');
                }
                
                // Hide review section and show progress
                document.getElementById('reviewSection').classList.remove('active');
                document.getElementById('progressSection').classList.add('active');
                
                // Continue polling
                startPolling(currentJobId);
                
            } catch (error) {
                showError(error.message);
                this.disabled = false;
                document.getElementById('retryBtn').disabled = false;
                this.textContent = 'Approve & Continue';
            }
        });
        
        // Try Again button handler
        document.getElementById('retryBtn').addEventListener('click', function() {
            document.getElementById('retryForm').classList.add('active');
            document.getElementById('retryImageInput').focus();
        });
        
        // Retry submit button handler
        document.getElementById('retrySubmitBtn').addEventListener('click', async function() {
            if (!currentJobId) return;
            
            const fileInput = document.getElementById('retryImageInput');
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select an image file');
                return;
            }
            
            this.disabled = true;
            this.textContent = 'Uploading...';
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            try {
                const response = await fetch(`/api/retry/${currentJobId}`, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to retry');
                }
                
                // Hide review section and show progress
                document.getElementById('reviewSection').classList.remove('active');
                document.getElementById('progressSection').classList.add('active');
                
                // Continue polling
                startPolling(currentJobId);
                
            } catch (error) {
                showError(error.message);
                this.disabled = false;
                this.textContent = 'Submit New Image';
            }
        });
        
        function showError(message) {
            document.getElementById('progressSection').classList.remove('active');
            document.getElementById('reviewSection').classList.remove('active');
            document.getElementById('errorSection').classList.add('active');
            document.getElementById('errorMessage').textContent = message;
        }
        
        function resetForm() {
            document.getElementById('generateBtn').disabled = false;
            document.getElementById('generateBtn').textContent = 'Generate 3D Coaster';
            document.getElementById('approveBtn').textContent = 'Approve & Continue';
            document.getElementById('retrySubmitBtn').disabled = false;
            document.getElementById('retrySubmitBtn').textContent = 'Submit New Image';
        }
    </script>
</body>
</html>
'''
    return html_content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
