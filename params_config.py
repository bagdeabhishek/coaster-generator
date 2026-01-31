# BFL FLUX.2 Klein 9B API Parameters
# Based on ComfyUI workflow analysis and BFL API documentation

# Current API endpoint: /v1/flux-2-klein-9b
# Documentation: https://api.bfl.ai/docs

# Parameters used in API call:
PROMPT_FILE = "prompt.txt"  # Loaded from external file for easy editing

# Image dimensions (must be multiples of 16)
# From ComfyUI workflow: 1024x1024
WIDTH = 1024
HEIGHT = 1024

# Seed for reproducibility
# Using same seed as ComfyUI workflow for consistency
SEED = 432262096973491

# Output format
# "png" - Returns PNG directly (no JPEG conversion needed)
# "jpeg" - Returns JPEG (default, requires conversion)
OUTPUT_FORMAT = "png"

# Note: The following parameters are NOT available for klein endpoint:
# - guidance (CFG scale) - [flex only]
# - steps (inference steps) - [flex only]
# - sampler - Not exposed in API
# 
# To use these parameters, switch to /v1/flux-2-flex endpoint
# But flex may have different model behavior/quality

# ComfyUI vs BFL API differences:
# ComfyUI: Local fp8 model with full control (steps, CFG, sampler)
# BFL API: Hosted full-precision model with limited parameters
# Results will differ but using same seed helps with consistency
