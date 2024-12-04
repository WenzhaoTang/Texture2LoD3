import trimesh
import numpy as np

# Load the model
# mesh = trimesh.load('/home/tang/code/ReLoD3_nus/raycasting/simplified_mesh/simplified_2nd.obj')
mesh = trimesh.load('/home/tang/code/ReLoD3_nus/raycasting/simplified_mesh/cluster_0.obj')

# Set the ray source & direction
ray_origins = np.array([[40, 0, 10]])  # Update: Ray source position
ray_directions = np.array([[-1, 2, 0]])  # Update: Ray direction, pointing towards the building

# Perform ray casting to find the intersecting faces
locations, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins=ray_origins,
    ray_directions=ray_directions
)

visible_faces = index_tri

# Find adjacent triangles and check normal vector similarity
quad_faces = []
for face_id in visible_faces:
    face_normal = mesh.face_normals[face_id]
    neighbors = mesh.face_adjacency[mesh.face_adjacency[:, 0] == face_id][:, 1]
    
    for neighbor_id in neighbors:
        neighbor_normal = mesh.face_normals[neighbor_id]
        # Check similarity
        if np.dot(face_normal, neighbor_normal) > 0.995:  # Angle less than about 8 degrees
            quad_faces.append((face_id, neighbor_id))

# Output the quadrilateral face verts
for face1, face2 in quad_faces:
    vertices_face1 = mesh.vertices[mesh.faces[face1]]
    vertices_face2 = mesh.vertices[mesh.faces[face2]]
    # Merge verts to form a quadrilateral
    quad_vertices = np.vstack((vertices_face1, vertices_face2))
    quad_vertices = np.unique(quad_vertices, axis=0)  # Remove duplicate vertices
    if len(quad_vertices) == 4:
        print("Quadrilateral vertices:", quad_vertices)
