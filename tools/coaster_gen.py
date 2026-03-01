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
from typing import Tuple

import numpy as np
import trimesh
import vtracer
from shapely.ops import unary_union


@dataclass
class CoasterParams:
    diameter: float = 100.0
    thickness: float = 5.0
    logo_depth: float = 0.6
    scale: float = 0.85
    flip_horizontal: bool = True
    top_rotate: int = 0
    bottom_rotate: int = 0


class CoasterGenerator:
    def __init__(
        self,
        diameter: float = 100.0,
        thickness: float = 5.0,
        logo_depth: float = 0.6,
        scale: float = 0.85,
        flip_horizontal: bool = True,
        top_rotate: int = 0,
        bottom_rotate: int = 0,
    ) -> None:
        self.params = CoasterParams(
            diameter=diameter,
            thickness=thickness,
            logo_depth=logo_depth,
            scale=scale,
            flip_horizontal=flip_horizontal,
            top_rotate=top_rotate,
            bottom_rotate=bottom_rotate,
        )

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

        logo_meshes = []
        for poly in processed_polys:
            try:
                logo_meshes.append(trimesh.creation.extrude_polygon(poly.buffer(0), height=p.logo_depth))
            except Exception:
                continue

        if not logo_meshes:
            raise RuntimeError("No valid logo meshes could be created")

        logos_combined = trimesh.util.concatenate(logo_meshes)

        bounds = logos_combined.bounds
        current_size_x = bounds[1][0] - bounds[0][0]
        current_size_y = bounds[1][1] - bounds[0][1]
        current_size = max(current_size_x, current_size_y)
        if current_size == 0:
            raise RuntimeError("Invalid logo bounds: zero size")

        target_size = p.diameter * p.scale
        mirror_x = -1 if p.flip_horizontal else 1

        matrix = np.eye(4)
        matrix[0, 0] *= mirror_x * target_size / current_size
        matrix[1, 1] *= target_size / current_size
        logos_combined.apply_transform(matrix)

        new_bounds = logos_combined.bounds
        center_x = (new_bounds[0][0] + new_bounds[1][0]) / 2
        center_y = (new_bounds[0][1] + new_bounds[1][1]) / 2
        trans = np.eye(4)
        trans[0, 3] = -center_x
        trans[1, 3] = -center_y
        logos_combined.apply_transform(trans)

        top_logo = logos_combined.copy()
        if p.top_rotate != 0:
            top_logo.apply_transform(
                trimesh.transformations.rotation_matrix(np.radians(p.top_rotate), [0, 0, 1])
            )
        top_logo.apply_translation([0, 0, (p.thickness / 2) - p.logo_depth])

        bottom_logo = logos_combined.copy()
        if p.bottom_rotate != 0:
            bottom_logo.apply_transform(
                trimesh.transformations.rotation_matrix(np.radians(p.bottom_rotate), [0, 0, 1])
            )
        bottom_logo.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        bottom_logo.apply_translation([0, 0, (-p.thickness / 2) + p.logo_depth])

        final_logos = trimesh.util.concatenate([top_logo, bottom_logo])
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

    def generate_coaster(
        self,
        input_image_path: str,
        output_dir: str,
        stamp_text: str = "",
        is_preview: bool = False,
    ) -> Tuple[str, str, str]:
        del stamp_text, is_preview

        with open(input_image_path, "rb") as f:
            image_bytes = f.read()

        svg_string = self._vectorize_image(image_bytes, output_dir)
        base_mesh, logos_mesh = self._build_meshes_from_svg(svg_string)

        base_name = uuid.uuid4().hex[:12]
        output_3mf_path = os.path.join(output_dir, f"{base_name}_coaster.3mf")
        body_stl_path = os.path.join(output_dir, f"{base_name}_Body.stl")
        logos_stl_path = os.path.join(output_dir, f"{base_name}_Logos.stl")

        base_mesh.export(body_stl_path)
        logos_mesh.export(logos_stl_path)
        self._export_three_mf(base_mesh, logos_mesh, output_3mf_path)

        return output_3mf_path, body_stl_path, logos_stl_path
