"""Core coaster generation logic used by Modal worker.

This module is intentionally standalone (no FastAPI/main imports) so it can run
inside Modal containers without app startup side effects.
"""

from __future__ import annotations

import io
import os
import uuid
import zipfile
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import trimesh
import vtracer
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


@dataclass
class CoasterParams:
    diameter: float = 100.0
    thickness: float = 5.0
    top_logo_depth: float = 0.6
    bottom_logo_depth: float = 0.6
    top_logo_height: float = 0.0
    scale: float = 0.85
    flip_horizontal: bool = True
    top_rotate: int = 0
    bottom_rotate: int = 0
    nozzle_diameter: float = 0.4
    auto_thicken: bool = True


class CoasterGenerator:
    def __init__(
        self,
        diameter: float = 100.0,
        thickness: float = 5.0,
        logo_depth: float = 0.6,
        top_logo_depth: float | None = None,
        bottom_logo_depth: float | None = None,
        top_logo_height: float = 0.0,
        scale: float = 0.85,
        flip_horizontal: bool = True,
        top_rotate: int = 0,
        bottom_rotate: int = 0,
        nozzle_diameter: float = 0.4,
        auto_thicken: bool = True,
    ) -> None:
        resolved_bottom_logo_depth = logo_depth if bottom_logo_depth is None else bottom_logo_depth
        resolved_top_logo_depth = logo_depth if top_logo_depth is None else top_logo_depth

        self.params = CoasterParams(
            diameter=diameter,
            thickness=thickness,
            top_logo_depth=resolved_top_logo_depth,
            bottom_logo_depth=resolved_bottom_logo_depth,
            top_logo_height=top_logo_height,
            scale=scale,
            flip_horizontal=flip_horizontal,
            top_rotate=top_rotate,
            bottom_rotate=bottom_rotate,
            nozzle_diameter=nozzle_diameter,
            auto_thicken=auto_thicken,
        )

    @staticmethod
    def _explode_polygons(geom) -> List[Polygon]:
        """Normalize geometry into a flat list of polygons."""
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, Polygon):
            return [geom]
        if isinstance(geom, MultiPolygon):
            return [p for p in geom.geoms if not p.is_empty]
        return []

    @staticmethod
    def _safe_polygon(poly: Polygon):
        """Attempt to repair invalid polygon and return geometry."""
        try:
            return poly.buffer(0)
        except Exception:
            return poly

    @staticmethod
    def _minimum_clearance(poly: Polygon) -> float:
        """Best-effort minimum-clearance metric for thin-feature detection."""
        try:
            val = float(poly.minimum_clearance)
            if np.isfinite(val):
                return max(0.0, val)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _filter_small_holes(poly: Polygon, min_hole_area: float) -> Polygon:
        """Remove holes below threshold area from a polygon."""
        kept_holes = []
        for ring in poly.interiors:
            hole = Polygon(ring)
            if abs(hole.area) >= min_hole_area:
                kept_holes.append(ring.coords)
        return Polygon(poly.exterior.coords, holes=kept_holes)

    def _printability_thresholds(self, scale_factor: float):
        """Return threshold values in both mm and source SVG units."""
        n = max(0.2, float(self.params.nozzle_diameter))

        # Conservative, printability-only thresholds.
        min_feature_mm = max(n * 1.6, 0.5)
        min_island_area_mm2 = max((n * 1.25) ** 2, 0.25)
        min_hole_area_mm2 = max((n * 1.0) ** 2, 0.2)
        max_thicken_mm = max(n * 0.6, 0.15)

        safe_scale = max(scale_factor, 1e-9)
        inv = 1.0 / safe_scale
        inv2 = inv * inv

        return {
            "min_feature_mm": min_feature_mm,
            "min_island_area_mm2": min_island_area_mm2,
            "min_hole_area_mm2": min_hole_area_mm2,
            "max_thicken_mm": max_thicken_mm,
            "min_feature_raw": min_feature_mm * inv,
            "min_island_area_raw": min_island_area_mm2 * inv2,
            "min_hole_area_raw": min_hole_area_mm2 * inv2,
            "max_thicken_raw": max_thicken_mm * inv,
        }

    def _needs_printability_fix(self, polygons: List[Polygon], t: dict) -> bool:
        """Check whether printability preprocessing is needed at all."""
        min_feature = t["min_feature_raw"]
        min_island = t["min_island_area_raw"]
        min_hole = t["min_hole_area_raw"]

        for poly in polygons:
            if abs(poly.area) < min_island:
                return True

            for ring in poly.interiors:
                if abs(Polygon(ring).area) < min_hole:
                    return True

            mc = self._minimum_clearance(poly)
            if mc > 0 and mc < min_feature:
                return True

        return False

    def _preprocess_polygons_for_printability(
        self,
        polygons: List[Polygon],
        scale_factor: float,
    ) -> List[Polygon]:
        """
        Minimal printability-only preprocessing in SVG space.

        Principle: no-op for already-printable geometry; targeted fixes only.
        """
        if not polygons:
            return polygons

        t = self._printability_thresholds(scale_factor)
        if not self._needs_printability_fix(polygons, t):
            return polygons

        min_feature = t["min_feature_raw"]
        min_island = t["min_island_area_raw"]
        min_hole = t["min_hole_area_raw"]
        max_thicken = t["max_thicken_raw"]

        processed: List[Polygon] = []
        for src_poly in polygons:
            geom = self._safe_polygon(src_poly)
            for poly in self._explode_polygons(geom):
                if abs(poly.area) < min_island:
                    continue

                poly = self._filter_small_holes(poly, min_hole)
                geom2 = self._safe_polygon(poly)

                # Auto-thicken only when specifically enabled and needed.
                if self.params.auto_thicken:
                    for p in self._explode_polygons(geom2):
                        mc = self._minimum_clearance(p)
                        if mc > 0 and mc < min_feature:
                            delta = min((min_feature - mc) * 0.5, max_thicken)
                            if delta > 0:
                                p = p.buffer(delta, join_style="mitre", cap_style="square")
                            p = self._safe_polygon(p)
                            for p2 in self._explode_polygons(p):
                                if abs(p2.area) >= min_island:
                                    processed.append(p2)
                        else:
                            if abs(p.area) >= min_island:
                                processed.append(p)
                else:
                    for p in self._explode_polygons(geom2):
                        if abs(p.area) >= min_island:
                            processed.append(p)

        return processed if processed else polygons

    def _vectorize_image(self, image_bytes: bytes, output_dir: str) -> str:
        temp_png_path = os.path.join(output_dir, f"temp_{uuid.uuid4().hex}.png")
        temp_svg_path = temp_png_path.replace(".png", ".svg")

        try:
            with open(temp_png_path, "wb") as f:
                f.write(image_bytes)

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
                path_precision=3,
            )

            with open(temp_svg_path, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            if os.path.exists(temp_png_path):
                os.remove(temp_png_path)
            if os.path.exists(temp_svg_path):
                os.remove(temp_svg_path)

    def vectorize_image(self, image_bytes: bytes, output_dir: str) -> str:
        """Public wrapper for PNG/JPG bytes -> SVG string."""
        return self._vectorize_image(image_bytes, output_dir)

    def _build_meshes_from_svg(self, svg_string: str) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        p = self.params

        base = trimesh.creation.cylinder(
            radius=p.diameter / 2,
            height=p.thickness,
            sections=120,
        )

        path_obj = trimesh.load_path(io.BytesIO(svg_string.encode("utf-8")), file_type="svg")

        polygons = []
        if hasattr(path_obj, "polygons_full") and path_obj.polygons_full:
            polygons = list(path_obj.polygons_full)
        elif hasattr(path_obj, "polygons_closed") and path_obj.polygons_closed:
            polygons = list(path_obj.polygons_closed)

        if not polygons:
            raise RuntimeError("No valid polygons found in SVG")

        polys = [poly for poly in polygons if not poly.is_empty]
        polys_sorted = sorted(polys, key=lambda poly: abs(poly.area), reverse=True)

        if polys_sorted:
            global_min_x = min(poly.bounds[0] for poly in polys_sorted)
            global_min_y = min(poly.bounds[1] for poly in polys_sorted)
            global_max_x = max(poly.bounds[2] for poly in polys_sorted)
            global_max_y = max(poly.bounds[3] for poly in polys_sorted)
            global_w = max(global_max_x - global_min_x, 1e-9)
            global_h = max(global_max_y - global_min_y, 1e-9)

            filtered_polys = []
            for poly in polys_sorted:
                min_x, min_y, max_x, max_y = poly.bounds
                poly_w = max_x - min_x
                poly_h = max_y - min_y
                bbox_area = max(poly_w * poly_h, 1e-9)
                area_ratio_in_bbox = abs(poly.area) / bbox_area
                coverage_x = poly_w / global_w
                coverage_y = poly_h / global_h

                is_canvas_sized = coverage_x >= 0.995 and coverage_y >= 0.995
                is_ring_like = area_ratio_in_bbox < 0.20
                is_solid_canvas = len(poly.interiors) == 0 and area_ratio_in_bbox > 0.85

                if is_canvas_sized and (is_ring_like or is_solid_canvas):
                    continue
                filtered_polys.append(poly)

            polys_sorted = filtered_polys

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
                if outer.buffer(1e-5).contains(inner):
                    holes.append(inner)
                    used[j] = True

            if holes:
                carved = outer.difference(unary_union(holes))
                if carved.is_empty:
                    continue
                if carved.geom_type == "Polygon":
                    processed_polys.append(carved)
                elif carved.geom_type == "MultiPolygon":
                    processed_polys.extend(list(carved.geoms))
            else:
                processed_polys.append(outer)

        if not processed_polys:
            raise RuntimeError("No valid polygons after SVG cleanup")

        # Compute current SVG size to derive mm conversion factor.
        global_min_x = min(poly.bounds[0] for poly in processed_polys)
        global_min_y = min(poly.bounds[1] for poly in processed_polys)
        global_max_x = max(poly.bounds[2] for poly in processed_polys)
        global_max_y = max(poly.bounds[3] for poly in processed_polys)
        current_size_x = global_max_x - global_min_x
        current_size_y = global_max_y - global_min_y
        current_size = max(current_size_x, current_size_y)
        if current_size <= 0:
            raise RuntimeError("Invalid logo bounds: zero size")

        target_size = p.diameter * p.scale
        mirror_x = -1 if p.flip_horizontal else 1
        scale_factor = target_size / current_size

        # Minimal printability preprocessing in SVG domain (only if needed).
        processed_polys = self._preprocess_polygons_for_printability(processed_polys, scale_factor)

        # Recompute bounds after preprocessing and scale to target size.
        global_min_x = min(poly.bounds[0] for poly in processed_polys)
        global_min_y = min(poly.bounds[1] for poly in processed_polys)
        global_max_x = max(poly.bounds[2] for poly in processed_polys)
        global_max_y = max(poly.bounds[3] for poly in processed_polys)
        current_size_x = global_max_x - global_min_x
        current_size_y = global_max_y - global_min_y
        current_size = max(current_size_x, current_size_y)
        if current_size <= 0:
            raise RuntimeError("Invalid logo bounds after preprocessing")

        scale_factor = target_size / current_size

        transformed_polys: List[Polygon] = []
        for poly in processed_polys:
            s = affinity.scale(poly, xfact=mirror_x * scale_factor, yfact=scale_factor, origin=(0, 0))
            transformed_polys.extend(self._explode_polygons(self._safe_polygon(s)))

        if not transformed_polys:
            raise RuntimeError("No valid polygons after scaling")

        # Center the transformed logo around origin.
        tx_min_x = min(poly.bounds[0] for poly in transformed_polys)
        tx_min_y = min(poly.bounds[1] for poly in transformed_polys)
        tx_max_x = max(poly.bounds[2] for poly in transformed_polys)
        tx_max_y = max(poly.bounds[3] for poly in transformed_polys)
        center_x = (tx_min_x + tx_max_x) / 2.0
        center_y = (tx_min_y + tx_max_y) / 2.0

        centered_polys: List[Polygon] = []
        for poly in transformed_polys:
            centered = affinity.translate(poly, xoff=-center_x, yoff=-center_y)
            centered_polys.extend(self._explode_polygons(self._safe_polygon(centered)))

        top_logo_depth = max(0.0, float(p.top_logo_depth))
        bottom_logo_depth = max(0.0, float(p.bottom_logo_depth))
        top_logo_height = max(0.0, float(p.top_logo_height))

        top_extrude_height = top_logo_depth + top_logo_height
        bottom_extrude_height = bottom_logo_depth

        top_logo = None
        if top_extrude_height > 0:
            top_meshes = []
            for poly in centered_polys:
                try:
                    top_meshes.append(trimesh.creation.extrude_polygon(poly, height=top_extrude_height))
                except Exception:
                    continue
            if top_meshes:
                top_logo = trimesh.util.concatenate(top_meshes)
                if p.top_rotate != 0:
                    top_logo.apply_transform(
                        trimesh.transformations.rotation_matrix(np.radians(p.top_rotate), [0, 0, 1])
                    )
                # Anchor top logo base at surface - top_logo_depth.
                top_logo.apply_translation([0, 0, (p.thickness / 2) - top_logo_depth])

        bottom_logo = None
        if bottom_extrude_height > 0:
            bottom_meshes = []
            for poly in centered_polys:
                try:
                    bottom_meshes.append(trimesh.creation.extrude_polygon(poly, height=bottom_extrude_height))
                except Exception:
                    continue
            if bottom_meshes:
                bottom_logo = trimesh.util.concatenate(bottom_meshes)
                if p.bottom_rotate != 0:
                    bottom_logo.apply_transform(
                        trimesh.transformations.rotation_matrix(np.radians(p.bottom_rotate), [0, 0, 1])
                    )
                bottom_logo.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
                bottom_logo.apply_translation([0, 0, (-p.thickness / 2) + bottom_logo_depth])

        logo_parts = [m for m in (top_logo, bottom_logo) if m is not None]
        if not logo_parts:
            raise RuntimeError("No valid top/bottom logo meshes could be created")

        final_logos = trimesh.util.concatenate(logo_parts)
        return base, final_logos

    def _export_three_mf(self, base: trimesh.Trimesh, logos: trimesh.Trimesh, output_3mf_path: str) -> None:
        xml_content = ['<?xml version="1.0" encoding="utf-8"?>']
        xml_content.append('<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" unit="millimeter" xml:lang="en-US">')
        xml_content.append("  <resources>")

        body_id = 1
        logos_id = 2
        composite_id = 3

        xml_content.append(f'    <object id="{body_id}" name="coaster_body" type="model">')
        xml_content.append("      <mesh>")
        xml_content.append("        <vertices>")
        for vertex in base.vertices:
            xml_content.append(f'          <vertex x="{vertex[0]}" y="{vertex[1]}" z="{vertex[2]}" />')
        xml_content.append("        </vertices>")
        xml_content.append("        <triangles>")
        for face in base.faces:
            xml_content.append(f'          <triangle v1="{face[0]}" v2="{face[1]}" v3="{face[2]}" />')
        xml_content.append("        </triangles>")
        xml_content.append("      </mesh>")
        xml_content.append("    </object>")

        xml_content.append(f'    <object id="{logos_id}" name="coaster_logos" type="model">')
        xml_content.append("      <mesh>")
        xml_content.append("        <vertices>")
        for vertex in logos.vertices:
            xml_content.append(f'          <vertex x="{vertex[0]}" y="{vertex[1]}" z="{vertex[2]}" />')
        xml_content.append("        </vertices>")
        xml_content.append("        <triangles>")
        for face in logos.faces:
            xml_content.append(f'          <triangle v1="{face[0]}" v2="{face[1]}" v3="{face[2]}" />')
        xml_content.append("        </triangles>")
        xml_content.append("      </mesh>")
        xml_content.append("    </object>")

        xml_content.append(f'    <object id="{composite_id}" name="coaster" type="model">')
        xml_content.append("      <components>")
        xml_content.append(f'        <component objectid="{body_id}" />')
        xml_content.append(f'        <component objectid="{logos_id}" />')
        xml_content.append("      </components>")
        xml_content.append("    </object>")

        xml_content.append("  </resources>")
        xml_content.append("  <build>")
        xml_content.append(f'    <item objectid="{composite_id}" />')
        xml_content.append("  </build>")
        xml_content.append("</model>")

        with zipfile.ZipFile(output_3mf_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("3D/3dmodel.model", "\n".join(xml_content))
            zf.writestr(
                "_rels/.rels",
                """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\" />
</Relationships>""",
            )
            zf.writestr(
                "[Content_Types].xml",
                """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\" />
  <Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\" />
</Types>""",
            )

    def generate_from_svg(
        self,
        svg_string: str,
        output_dir: str,
        file_prefix: str | None = None,
    ) -> Tuple[str, str, str]:
        """Generate 3MF + STL files from SVG string."""
        base_mesh, logos_mesh = self._build_meshes_from_svg(svg_string)

        base_name = file_prefix or uuid.uuid4().hex[:12]
        output_3mf_path = os.path.join(output_dir, f"{base_name}_coaster.3mf")
        body_stl_path = os.path.join(output_dir, f"{base_name}_Body.stl")
        logos_stl_path = os.path.join(output_dir, f"{base_name}_Logos.stl")

        base_mesh.export(body_stl_path)
        logos_mesh.export(logos_stl_path)
        self._export_three_mf(base_mesh, logos_mesh, output_3mf_path)

        return output_3mf_path, body_stl_path, logos_stl_path

    def generate_coaster(
        self,
        input_image_path: str,
        output_dir: str,
        stamp_text: str = "",
        is_preview: bool = False,
        file_prefix: str | None = None,
    ) -> Tuple[str, str, str]:
        del stamp_text, is_preview

        with open(input_image_path, "rb") as f:
            image_bytes = f.read()

        svg_string = self._vectorize_image(image_bytes, output_dir)
        return self.generate_from_svg(svg_string, output_dir, file_prefix=file_prefix)
