import numpy as np
import math

from texturing_utils import (
    load_fov_csv,
    project_latlng_to_xy,
    load_building_mesh,
    get_building_bounds,
    get_wall_surfaces_info,
    compute_2d_distance,
    interpolate_pitch,
    interpolate_angles,
    angle_to_direction,
    do_raycast,
    visualize_3d_scene,
    export_textured_hits
)


def main():
    csv_fov = "data/demo_data/geofov.csv" # update
    df_fov = load_fov_csv(csv_fov)
    df_fov = df_fov[df_fov["building_id"] == 4959322] # update

    mesh_path = "data/demo_data/WallSurface_4959322/modified_4959322.obj"
    mesh = load_building_mesh(mesh_path)

    gml_path = "data/demo_data/citygml/DEBY_LOD2_4959322.gml" # update
    lower_bound, upper_bound = get_building_bounds(gml_path)
    if lower_bound is None or upper_bound is None:
        print("Failed to extract building bounds.")
        return
    print(f"Building bounds: {lower_bound}, {upper_bound}")

    surfaces = get_wall_surfaces_info(gml_path)
    print(f"Found {len(surfaces)} wall surfaces.")

    camera_z = lower_bound + 1.7
    offset_distance = 0.01

    texture_path = "data/demo_data/texture_4959322/4959322.png" # update
    building_center = mesh.vertices.mean(axis=0)

    grouped = df_fov.groupby("pid")
    exported = False

    for pid, group in grouped:
        lat = group["lat"].iloc[0]
        lng = group["lng"].iloc[0]
        cam_x, cam_y = project_latlng_to_xy(lat, lng, "EPSG:25832")

        if "degree" in group.columns:
            central_angle = group["degree"].iloc[0]
            direction = angle_to_direction(central_angle, 0)
        else:
            direction = np.array([0, 1], dtype=float)

        cam_x -= direction[0] * offset_distance
        cam_y -= direction[1] * offset_distance
        cam_origin = np.array([cam_x, cam_y, camera_z], dtype=float)

        target_height = 0.5 * (lower_bound + upper_bound)
        pitch_rad = math.atan2((target_height - camera_z), offset_distance)
        overall_pitch = math.degrees(pitch_rad)
        print(f"\nCamera pid={pid}, origin=({cam_x:.2f}, {cam_y:.2f}, {camera_z:.2f}), pitch={overall_pitch:.2f}°")

        best_dist = float('inf')
        best_pitch = overall_pitch
        for surf in surfaces:
            dist_2d = compute_2d_distance(cam_x, cam_y, surf["cx"], surf["cy"])
            if dist_2d < best_dist:
                best_dist = dist_2d
                vertical_diff = surf["z_mid"] - camera_z
                pitch_rad_wall = math.atan2(vertical_diff, dist_2d)
                best_pitch = math.degrees(pitch_rad_wall)

        pitch_samples = 5 # to adjust
        pitch_lower = best_pitch - 5.0 # update
        pitch_upper = best_pitch + 5.0 # update
        pitch_list = interpolate_pitch(pitch_lower, pitch_upper, pitch_samples)

        angle_samples = 10
        hits_list = []
        for _, row in group.iterrows():
            left_a = row["left_angle_geo"]
            right_a = row["right_angle_geo"]
            offset_angle = 3.0
            left_a_adj = left_a + offset_angle
            right_a_adj = right_a - offset_angle
            angles = interpolate_angles(left_a_adj, right_a_adj, angle_samples)
            for a in angles:
                for p in pitch_list:
                    hit_pt, tri_idx = do_raycast(mesh, cam_origin, a, p)
                    hits_list.append({
                        "angle": a,
                        "pitch": p,
                        "hit": hit_pt,
                        "tri": tri_idx
                    })

        out_png = f"camera_pid_{pid}_raycasting.png"
        title = f"Camera pid={pid} (angles={angle_samples}, pitches={pitch_samples})"
        visualize_3d_scene(mesh, cam_origin, hits_list, out_path=out_png, title=title)

        hit_indices = {h["tri"] for h in hits_list if h["tri"] is not None}
        if hit_indices:
            print(f"Camera pid={pid} hit {len(hit_indices)} triangles")
            output_folder = f"output_textured_mesh_4959322_{pid}"
            success = export_textured_hits(mesh, hit_indices, output_folder, texture_path, building_center)
            if success and not exported:
                exported = True
        else:
            print(f"Camera pid={pid} had no hits; skipping texture export.")

    if not exported:
        print("No cameras produced textured OBJ.")


if __name__ == "__main__":
    main()
