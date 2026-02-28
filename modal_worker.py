import modal
from typing import Dict, Any, Tuple
import io

# Define the container environment Modal needs to run the 3D generation.
# We include all the heavy mathematical libraries here.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trimesh[easy]==4.0.5",
        "vtracer==0.6.11",
        "Pillow==10.2.0",
        "numpy==1.26.3",
        "scipy==1.11.4",
        "shapely==2.0.2"
    )
)

# Define the Modal application
app = modal.App("coaster-generator", image=image)

@app.function(cpu=2.0, memory=2048)  # Give it 2 CPU cores and 2GB RAM
def generate_3d_coaster(
    image_bytes: bytes, 
    params: Dict[str, Any], 
    stamp_text: str
) -> Tuple[bytes, bytes, bytes]:
    """
    Serverless endpoint to generate the 3D coaster.
    This runs completely remotely when triggered!
    
    Args:
        image_bytes: The original generated image bytes
        params: Dictionary containing diameter, thickness, scale, etc.
        stamp_text: Text for the stamp
        
    Returns:
        Tuple of (combined_3mf_bytes, body_stl_bytes, logos_stl_bytes)
    """
    # Import the heavy mathematical processing code INSIDE the function,
    # so it only loads inside the Modal container when executing.
    from tools.coaster_gen import CoasterGenerator
    import tempfile
    import os
    
    # Initialize the generator with parameters
    generator = CoasterGenerator(
        diameter=params.get("diameter", 100),
        thickness=params.get("thickness", 5),
        logo_depth=params.get("logo_depth", 0.6),
        scale=params.get("scale", 0.85),
        flip_horizontal=params.get("flip_horizontal", True),
        top_rotate=params.get("top_rotate", 0),
        bottom_rotate=params.get("bottom_rotate", 0)
    )
    
    # Use a temporary directory inside the container
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.png")
        with open(input_path, "wb") as f:
            f.write(image_bytes)
            
        # Run the heavy 3D math processing
        combined_3mf, body_stl, logos_stl = generator.generate_coaster(
            input_image_path=input_path,
            output_dir=temp_dir,
            stamp_text=stamp_text,
            is_preview=False
        )
        
        # Read the resulting files into memory to send back over the network
        with open(combined_3mf, "rb") as f:
            combined_bytes = f.read()
            
        with open(body_stl, "rb") as f:
            body_bytes = f.read()
            
        with open(logos_stl, "rb") as f:
            logos_bytes = f.read()
            
    # Return the binary data back to the calling server
    return combined_bytes, body_bytes, logos_bytes
