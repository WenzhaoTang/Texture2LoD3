import os
import math
import shutil
import numpy as np
import pandas as pd
import pyproj
import trimesh
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

import geopandas as gpd
from shapely.geometry import Point


def load_fov_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def project_latlng_to_xy(lat, lng, crs_out="EPSG:25832"):
    wgs84 = pyproj.CRS("EPSG:4326")
    out_crs = pyproj.CRS(crs_out)
    transformer = pyproj.Transformer.from_crs(wgs84, out_crs, always_xy=True)
    x, y = transformer.transform(lng, lat)
    return x, y


def load_building_mesh(obj_path):
    mesh = trimesh.load(obj_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh


def angle_to_direction(angle_deg, pitch_deg=0):
    yaw = math.radians(angle_deg % 360)
    pitch = math.radians(pitch_deg)
    dx = math.cos(pitch) * math.sin(yaw)
    dy = math.cos(pitch) * math.cos(yaw)
    dz = math.sin(pitch)
    vec = np.array([dx, dy, dz], dtype=float)
    return vec / np.linalg.norm(vec)


def do_raycast(mesh, cam_origin, angle_deg, pitch_deg=0):
    direction = angle_to_direction(angle_deg, pitch_deg)
    origins = np.array([cam_origin])
    directions = np.array([direction])
    hits, index_r, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )
    if len(hits) > 0:
        return hits[0], index_tri[0]
    return None, None


def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d(x_middle - max_range / 2, x_middle + max_range / 2)
    ax.set_ylim3d(y_middle - max_range / 2, y_middle + max_range / 2)
    ax.set_zlim3d(z_middle - max_range / 2, z_middle + max_range / 2)


def visualize_3d_scene(mesh, cam_origin, hits_list, out_path=None, title="3D Raycasting Demo"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    verts = mesh.vertices
    faces = mesh.faces
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    try:
        ax.plot_trisurf(
            x, y, faces, Z=z,
            color=(0.7, 0.7, 0.7, 0.3),
            edgecolor='none', shade=True
        )
    except Exception as e:
        print("plot_trisurf error:", e)
        ax.scatter(x, y, z, s=1, c='gray', alpha=0.3)

    hit_tri_indices = set()
    for item in hits_list:
        tri_idx = item.get("tri")
        if tri_idx is not None:
            hit_tri_indices.add(tri_idx)

    if len(hit_tri_indices) > 0:
        hit_polys = []
        for tri_idx in hit_tri_indices:
            face = faces[tri_idx]
            tri_verts = verts[face]
            hit_polys.append(tri_verts)
        hit_collection = Poly3DCollection(
            hit_polys,
            facecolors=np.array([(1, 0, 0, 0.5)]),
            edgecolors='r',
            linewidths=1
        )
        ax.add_collection3d(hit_collection)

    ax.scatter(cam_origin[0], cam_origin[1], cam_origin[2],
               color='red', marker='^', s=60, label='Camera')

    max_range = 200.0
    hit_points = []
    for item in hits_list:
        a_deg = item['angle']
        p_deg = item['pitch']
        hit = item['hit']
        direction = angle_to_direction(a_deg, p_deg)
        if hit is not None:
            hx, hy, hz = hit
            xs = [cam_origin[0], hx]
            ys = [cam_origin[1], hy]
            zs = [cam_origin[2], hz]
            hit_points.append([hx, hy, hz])
        else:
            fallback = cam_origin + direction * max_range
            xs = [cam_origin[0], fallback[0]]
            ys = [cam_origin[1], fallback[1]]
            zs = [cam_origin[2], fallback[2]]
        ax.plot(xs, ys, zs, color='green', alpha=0.8)

    if len(hit_points) > 0:
        hit_points = np.array(hit_points)
        ax.scatter(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2],
                   color='blue', s=30, marker='o', label='Ray Hits')

    arr_x = np.concatenate([x, [cam_origin[0]]])
    arr_y = np.concatenate([y, [cam_origin[1]]])
    arr_z = np.concatenate([z, [cam_origin[2]]])
    x_min, x_max = arr_x.min(), arr_x.max()
    y_min, y_max = arr_y.min(), arr_y.max()
    z_min, z_max = arr_z.min(), arr_z.max()

    def expand(a, b, ratio=0.1):
        dist = (b - a) * ratio
        return a - dist, b + dist

    x_min, x_max = expand(x_min, x_max, 0.1)
    y_min, y_max = expand(y_min, y_max, 0.1)
    z_min, z_max = expand(z_min, z_max, 0.1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    handles, labels = ax.get_legend_handles_labels()
    hit_patch = mpatches.Patch(color='red', alpha=0.5, label='Hit Triangles')
    if hit_patch not in handles:
        handles.append(hit_patch)
        labels.append('Hit Triangles')
    ax.legend(handles, labels)

    set_axes_equal_3d(ax)

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved 3D image: {out_path}")

    plt.show()


def compute_uv_for_triangle(mesh, face_idx, building_center):
    original_face = mesh.faces[face_idx]
    verts = mesh.vertices[original_face]

    face_center = np.mean(verts, axis=0)
    normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    direction_to_center = face_center - building_center
    if np.dot(normal, direction_to_center) < 0:
        flipped_face = original_face[::-1]
        verts = verts[::-1]
        normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        face = flipped_face
    else:
        face = original_face

    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-12:
        normal = np.array([0, 0, 1], dtype=float)
    else:
        normal /= norm_val

    world_up = np.array([0, 0, 1], dtype=float)
    if abs(np.dot(world_up, normal)) > 0.99:
        world_up = np.array([1, 0, 0], dtype=float)

    tangent = np.cross(world_up, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    projected = []
    for v in verts:
        u_val = np.dot(v, tangent)
        v_val = np.dot(v, bitangent)
        projected.append([u_val, v_val])
    projected = np.array(projected)

    min_uv = projected.min(axis=0)
    max_uv = projected.max(axis=0)
    size_uv = max_uv - min_uv
    size_uv[size_uv == 0] = 1e-12
    normed = (projected - min_uv) / size_uv

    return face, normed


def export_textured_hits(mesh, hit_indices, output_folder, texture_path, building_center):
    if not hit_indices:
        print("No faces hit; skipping export.")
        return False

    face_indices = np.arange(len(mesh.faces))
    mask = np.ones(len(mesh.faces), dtype=bool)
    for hi in hit_indices:
        mask[hi] = False
    default_faces = mesh.faces[mask]

    full_vertices = mesh.vertices
    lines = []
    lines.append("mtllib textured_mesh.mtl\n")

    for v in full_vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    lines.append("")

    lines.append("o default")
    for face in default_faces:
        face_indices_str = " ".join(str(idx + 1) for idx in face)
        lines.append("f " + face_indices_str)
    lines.append("")

    lines.append("o textured_hits")
    lines.append("usemtl textured_material")

    vt_counter = 1
    for face_idx in hit_indices:
        face, uv_coords = compute_uv_for_triangle(mesh, face_idx, building_center)
        vt_index_for_face = []
        for i in range(3):
            u = uv_coords[i, 0]
            v = uv_coords[i, 1]
            lines.append(f"vt {u:.6f} {v:.6f}")
            vt_index_for_face.append(vt_counter)
            vt_counter += 1

        f_str = "f"
        for i in range(3):
            vi = face[i] + 1
            ti = vt_index_for_face[i]
            f_str += f" {vi}/{ti}"
        lines.append(f_str)

    lines.append("")

    obj_data = "\n".join(lines)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_obj = os.path.join(output_folder, "textured_mesh.obj")
    with open(output_obj, "w") as f:
        f.write(obj_data)
    print(f"Exported OBJ to {output_obj}")

    output_mtl = os.path.join(output_folder, "textured_mesh.mtl")
    texture_filename = os.path.basename(texture_path)
    mtl_content = "newmtl textured_material\n"
    mtl_content += "Ka 1.000 1.000 1.000\n"
    mtl_content += "Kd 1.000 1.000 1.000\n"
    mtl_content += "Ks 0.000 0.000 0.000\n"
    mtl_content += f"map_Kd {texture_filename}\n"
    with open(output_mtl, "w") as f:
        f.write(mtl_content)
    print(f"Exported MTL to {output_mtl}")

    output_texture = os.path.join(output_folder, texture_filename)
    try:
        shutil.copy(texture_path, output_texture)
        print(f"Copied texture to {output_texture}")
    except Exception as e:
        print("Failed to copy texture:", e)

    return True


def get_building_bounds(gml_path):
    tree = ET.parse(gml_path)
    root = tree.getroot()
    ns = {
        'bldg': 'http://www.opengis.net/citygml/building/2.0',
        'gml': 'http://www.opengis.net/gml'
    }
    lower_list = []
    upper_list = []
    for wall in root.findall('.//bldg:WallSurface', ns):
        posList_elem = wall.find('.//gml:posList', ns)
        if posList_elem is not None and posList_elem.text:
            coords = list(map(float, posList_elem.text.split()))
            z_values = [coords[i] for i in range(2, len(coords), 3)]
            lower_list.append(min(z_values))
            upper_list.append(max(z_values))
    if lower_list and upper_list:
        return min(lower_list), max(upper_list)
    else:
        return None, None


def get_wall_surfaces_info(gml_path):
    tree = ET.parse(gml_path)
    root = tree.getroot()
    ns = {
        'bldg': 'http://www.opengis.net/citygml/building/2.0',
        'gml': 'http://www.opengis.net/gml'
    }
    surfaces = []
    for wall in root.findall('.//bldg:WallSurface', ns):
        all_x, all_y, all_z = [], [], []
        posList_elems = wall.findall('.//gml:posList', ns)
        for posList_elem in posList_elems:
            if posList_elem.text:
                coords = list(map(float, posList_elem.text.split()))
                for i in range(0, len(coords), 3):
                    all_x.append(coords[i])
                    all_y.append(coords[i+1])
                    all_z.append(coords[i+2])
        if all_x:
            cx = sum(all_x) / len(all_x)
            cy = sum(all_y) / len(all_y)
            z_min = min(all_z)
            z_max = max(all_z)
            z_mid = 0.5 * (z_min + z_max)
            surfaces.append({
                "cx": cx,
                "cy": cy,
                "z_min": z_min,
                "z_max": z_max,
                "z_mid": z_mid
            })
    return surfaces


def compute_2d_distance(ax, ay, bx, by):
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def interpolate_angles(a1, a2, n=5):
    a1_mod = a1 % 360
    a2_mod = a2 % 360
    diff = (a2_mod - a1_mod) % 360
    if diff > 180:
        diff -= 360
    step = diff / (n - 1) if n > 1 else 0.0
    return [(a1_mod + i * step) % 360 for i in range(n)]


def interpolate_pitch(p1, p2, n=5):
    step = (p2 - p1) / (n - 1) if n > 1 else 0.0
    return [p1 + i * step for i in range(n)]
