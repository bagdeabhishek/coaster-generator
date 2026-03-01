#!/usr/bin/env python3
"""
Test script to verify 3MF export functionality
Generates a simple test coaster and exports as 3MF
"""

import os
import sys
import trimesh

# Create output directory
output_dir = "/home/abhishek/Documents/CoasterWebService/test_output"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("3MF Export Test Script")
print("="*60)

# Create simple test meshes
print("\n1. Creating test meshes...")

# Body - simple cylinder
body = trimesh.creation.cylinder(radius=50, height=5, sections=64)
print(f"   Body: {len(body.vertices)} vertices, {len(body.faces)} faces")

# Logos - simple box (simulating extruded logo)
logos = trimesh.creation.box(extents=[30, 30, 2])
# Position logos on top of body
logos.apply_translation([0, 0, 3.5])
print(f"   Logos: {len(logos.vertices)} vertices, {len(logos.faces)} faces")

# Method 1: Export individual STLs (for debugging)
print("\n2. Exporting individual STL files...")
body_stl = os.path.join(output_dir, "test_body.stl")
logos_stl = os.path.join(output_dir, "test_logos.stl")

body.export(body_stl)
logos.export(logos_stl)

print(f"   ✓ Body STL: {os.path.getsize(body_stl)} bytes")
print(f"   ✓ Logos STL: {os.path.getsize(logos_stl)} bytes")

# Method 2: Export as 3MF using Scene
print("\n3. Exporting as 3MF (Scene method)...")
output_3mf = os.path.join(output_dir, "test_coaster.3mf")

scene = trimesh.Scene()
scene.add_geometry(body, node_name='coaster_body')
scene.add_geometry(logos, node_name='coaster_logos')

print(f"   Scene geometry count: {len(scene.geometry)}")

try:
    scene.export(output_3mf, file_type='3mf')
    file_size = os.path.getsize(output_3mf)
    print(f"   ✓ 3MF exported: {file_size} bytes ({file_size/1024:.1f} KB)")
    
    # Verify 3MF contents
    print("\n4. Verifying 3MF file...")
    import zipfile
    with zipfile.ZipFile(output_3mf, 'r') as zf:
        files = zf.namelist()
        print(f"   Files in archive: {files}")
        
        # Check the main model file
        if '3D/3dmodel.model' in files:
            model_content = zf.read('3D/3dmodel.model').decode('utf-8')[:500]
            print(f"\n   First 500 chars of 3dmodel.model:")
            print(f"   {model_content}")
    
    # Try to load it back
    print("\n5. Attempting to load 3MF back...")
    loaded = trimesh.load(output_3mf)
    print(f"   ✓ Loaded successfully")
    print(f"   Geometry count: {len(loaded.geometry)}")
    for name, geom in loaded.geometry.items():
        print(f"   - {name}: {len(geom.vertices)} vertices")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Method 3: Test the new build-section fix
print("\n6. Testing 3MF with build section fix...")
try:
    # Copy the function from main.py
    import sys
    sys.path.insert(0, '/home/abhishek/Documents/CoasterWebService')
    
    # Import the fix function
    from main import _fix_3mf_build_section
    
    # Create a 3MF using scene
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name='coaster_body')
    scene.add_geometry(logos, node_name='coaster_logos')
    
    output_3mf3 = os.path.join(output_dir, "test_coaster_fixed.3mf")
    scene.export(output_3mf3, file_type='3mf')
    
    # Apply the fix
    _fix_3mf_build_section(output_3mf3)
    
    file_size3 = os.path.getsize(output_3mf3)
    print(f"   ✓ Fixed 3MF: {file_size3} bytes ({file_size3/1024:.1f} KB)")
    
    # Verify it has build section
    with zipfile.ZipFile(output_3mf3, 'r') as zf:
        content = zf.read('3D/3dmodel.model').decode('utf-8')
        if '<build>' in content:
            print("   ✓ Build section present!")
        else:
            print("   ✗ Build section missing")
            
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n7. Comparing all 3MF files...")
for filename in ['test_coaster.3mf', 'test_coaster_combined.3mf', 'test_coaster_fixed.3mf']:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        with zipfile.ZipFile(filepath, 'r') as zf:
            content = zf.read('3D/3dmodel.model').decode('utf-8')
            has_build = '<build>' in content
            print(f"   {filename}: {'✓ has build' if has_build else '✗ no build'}")

print("\n" + "="*60)
print(f"Test files saved to: {output_dir}")
print("="*60)
print("\nPlease check:")
print("1. test_body.stl - Should load in any 3D viewer")
print("2. test_logos.stl - Should load in any 3D viewer")
print("3. test_coaster.3mf - Try opening in OrcaSlicer")
print("4. test_coaster_combined.3mf - Alternative 3MF format")
print("\nIf test_coaster.3mf works but the main app doesn't,")
print("the issue is in how meshes are being created in the main app.")
