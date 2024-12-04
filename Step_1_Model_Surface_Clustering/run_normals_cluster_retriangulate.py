import os
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import open3d as o3d
import argparse

def parse_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(coord) for coord in parts[1:]])
            elif parts[0] == 'f':
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])
    return np.array(vertices), np.array(faces)

def compute_normals(vertices, faces):
    normals = []
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal = normal / norm
        normals.append(normal)
    return np.array(normals)

def cluster_normals(normals, eps=0.1, min_samples=1):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(normals)
    clusters = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(i)
    return clusters

def fit_plane_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    centroid = pca.mean_
    normal = pca.components_[2]
    in_plane_x = pca.components_[0]
    in_plane_y = pca.components_[1]
    return centroid, normal, in_plane_x, in_plane_y

def project_points_to_plane(points, centroid, in_plane_x, in_plane_y):
    translated_points = points - centroid
    x_coords = np.dot(translated_points, in_plane_x)
    y_coords = np.dot(translated_points, in_plane_y)
    return np.vstack((x_coords, y_coords)).T

def compute_min_area_rectangle(projected_2d):
    if len(projected_2d) < 3:
        raise ValueError("Not enough points to compute a bounding rectangle.")
    
    polygon = Polygon(projected_2d).convex_hull
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    min_rect = polygon.minimum_rotated_rectangle
    rect_coords = np.array(min_rect.exterior.coords)[:4]
    return rect_coords

def map_corners_to_3d(corners_2d, centroid, in_plane_x, in_plane_y):
    corners_3d = centroid + np.outer(corners_2d[:,0], in_plane_x) + np.outer(corners_2d[:,1], in_plane_y)
    return corners_3d

def create_two_triangles(corners):
    triangle1 = [0, 1, 2]
    triangle2 = [0, 2, 3]
    return [triangle1, triangle2]

def select_quadrilateral_corners(vertices, faces, cluster_faces):
    cluster_face_indices = cluster_faces
    cluster_vertex_indices = np.unique(faces[cluster_face_indices].flatten())
    cluster_vertices = vertices[cluster_vertex_indices]
    
    if len(cluster_vertices) < 4:
        raise ValueError("Not enough vertices to form a quadrilateral.")
    
    centroid, normal, in_plane_x, in_plane_y = fit_plane_pca(cluster_vertices)
    projected_2d = project_points_to_plane(cluster_vertices, centroid, in_plane_x, in_plane_y)
    
    try:
        rect_2d = compute_min_area_rectangle(projected_2d)
    except ValueError as e:
        print(f"Error computing bounding rectangle: {e}")
        hull = ConvexHull(projected_2d)
        rect_2d = projected_2d[hull.vertices][:4]
    
    rect_3d = map_corners_to_3d(rect_2d, centroid, in_plane_x, in_plane_y)
    two_triangles = create_two_triangles(rect_3d)
    
    return rect_3d, two_triangles

def save_clusters_as_obj(vertices, faces, clusters, output_dir, normals):
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster_id, face_indices in clusters.items():
        cluster_file = os.path.join(output_dir, f"cluster_{cluster_id}.obj")
        with open(cluster_file, 'w') as obj_file:
            new_vertices = []
            new_faces = []
            
            try:
                corners_3d, two_triangles = select_quadrilateral_corners(vertices, faces, face_indices)
                starting_index = 1
                for corner in corners_3d:
                    obj_file.write(f"v {corner[0]} {corner[1]} {corner[2]}\n")
                
                obj_file.write(f"f {starting_index} {starting_index + 1} {starting_index + 2}\n")
                obj_file.write(f"f {starting_index} {starting_index + 2} {starting_index + 3}\n")
                
                print(f"Cluster {cluster_id} saved to {cluster_file}")
                print(f"Cluster {cluster_id} Corners:\n{corners_3d}\n")
                
            except Exception as e:
                print(f"Failed to process cluster {cluster_id}: {e}\n")

def visualize_clusters_with_quadrilaterals(vertices, faces, clusters, output_dir):
    geometries = []
    color_map = {}
    for cluster_id in clusters.keys():
        color_map[cluster_id] = np.random.rand(3)
    
    for cluster_id, face_indices in clusters.items():
        try:
            corners_3d, two_triangles = select_quadrilateral_corners(vertices, faces, face_indices)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(corners_3d)
            pcd.paint_uniform_color(color_map[cluster_id])
            geometries.append(pcd)
            
            lines = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0]
            ]
            colors = [[1, 0, 0] for _ in lines]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(corners_3d),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
        
        except Exception as e:
            print(f"Failed to visualize cluster {cluster_id}: {e}")
    
    o3d.visualization.draw_geometries(geometries)

def main(file_path, output_dir):
    vertices, faces = parse_obj(file_path)
    print(f"Parsed {len(vertices)} vertices and {len(faces)} faces.")
    
    normals = compute_normals(vertices, faces)
    print(f"Computed normals for {len(normals)} faces.")
    
    clusters = cluster_normals(normals)
    print("Clustered Normals:")
    for cluster_id, face_indices in clusters.items():
        print(f"  Cluster {cluster_id}: {len(face_indices)} faces")
    
    save_clusters_as_obj(vertices, faces, clusters, output_dir, normals)
    visualize_clusters_with_quadrilaterals(vertices, faces, clusters, output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and cluster OBJ file.")
    parser.add_argument("obj_file_path", type=str, help="Path to the input OBJ file.")
    parser.add_argument("output_directory", type=str, help="Directory to save the output clusters.")
    
    args = parser.parse_args()
    
    main(args.obj_file_path, args.output_directory)
