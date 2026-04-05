"""Vase-safe coaster holder generation helpers.

Phase 1 intentionally creates a closed cylindrical container (no finger slit)
so users can print in spiral vase mode.
"""

from __future__ import annotations

import math
from typing import Tuple

import trimesh
from shapely.geometry import Point


def _circle(radius: float, resolution: int = 64):
    return Point(0, 0).buffer(radius, resolution=max(16, resolution))


def resolve_holder_dimensions(
    coaster_diameter: float,
    coaster_thickness: float,
    coaster_count: int,
    nozzle_diameter: float,
    radial_clearance_mm: float = 1.2,
    top_clearance_mm: float = 1.0,
    bottom_thickness_mm: float = 0.8,
) -> dict:
    """Resolve holder dimensions from user-friendly inputs.

    Wall thickness is chosen from nozzle diameter to target a single-wall print.
    """
    count = max(1, int(coaster_count))
    wall_thickness = max(0.35, min(float(nozzle_diameter), 1.0))

    inner_radius = (float(coaster_diameter) / 2.0) + float(radial_clearance_mm)
    outer_radius = inner_radius + wall_thickness

    inner_height = (count * float(coaster_thickness)) + float(top_clearance_mm)
    total_height = inner_height + float(bottom_thickness_mm)

    return {
        "coaster_count": count,
        "wall_thickness": wall_thickness,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "inner_height": inner_height,
        "bottom_thickness": float(bottom_thickness_mm),
        "total_height": total_height,
        "radial_clearance": float(radial_clearance_mm),
        "top_clearance": float(top_clearance_mm),
    }


def build_vase_safe_holder_mesh(
    coaster_diameter: float,
    coaster_thickness: float,
    coaster_count: int,
    nozzle_diameter: float,
    radial_clearance_mm: float = 1.2,
    top_clearance_mm: float = 1.0,
    bottom_thickness_mm: float = 0.8,
    polygon_resolution: int = 64,
) -> Tuple[trimesh.Trimesh, dict]:
    """Create holder mesh + resolved dimensions.

    Mesh coordinates are generated with base at z=0.
    """
    dims = resolve_holder_dimensions(
        coaster_diameter=coaster_diameter,
        coaster_thickness=coaster_thickness,
        coaster_count=coaster_count,
        nozzle_diameter=nozzle_diameter,
        radial_clearance_mm=radial_clearance_mm,
        top_clearance_mm=top_clearance_mm,
        bottom_thickness_mm=bottom_thickness_mm,
    )

    outer = _circle(dims["outer_radius"], resolution=polygon_resolution)
    inner = _circle(dims["inner_radius"], resolution=polygon_resolution)

    wall_ring = outer.difference(inner)
    wall_mesh = trimesh.creation.extrude_polygon(wall_ring, height=dims["total_height"])

    bottom_disk = inner
    bottom_mesh = trimesh.creation.extrude_polygon(bottom_disk, height=dims["bottom_thickness"])

    holder_mesh = trimesh.util.concatenate([wall_mesh, bottom_mesh])
    return holder_mesh, dims
