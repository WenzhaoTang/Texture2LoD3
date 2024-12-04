import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the building model
# mesh = trimesh.load('/home/tang/code/ReLoD3_nus/raycasting/simplified_mesh/simplified_2nd.obj')
mesh = trimesh.load('/home/tang/code/ReLoD3_nus/raycasting/simplified_mesh/cluster_0.obj')

# Given vertices of the detected quadrilateral

'''/raycasting/step_1_test_find_face.py
Quadrilateral vertices: [[-1.07024712e-03  5.51371276e-04 -2.31853647e-10]
 [-1.06750753e-03  5.50293071e-04  1.89510000e+01]
 [ 2.39801696e+01  6.09369333e+01 -6.84448054e-11]
 [ 2.39801723e+01  6.09369322e+01  1.89510000e+01]]'''
quad_vertices = np.array([
    [-1.07024712e-03, 5.51371276e-04, -2.31853647e-10],
    [-1.06750753e-03, 5.50293071e-04, 1.89510000e+01],
    [2.39801696e+01, 6.09369333e+01, -6.84448054e-11],
    [2.39801723e+01, 6.09369322e+01, 1.89510000e+01]
])

# Find the faces containing these vertices in the mesh
quad_faces = []
for face_id, face in enumerate(mesh.faces):
    face_vertices = mesh.vertices[face]
    # Check if each face contains at least two of the quad vertices
    matching_vertices = [np.any(np.all(np.isclose(face_vertices[i], quad_vertices, atol=1e-4), axis=1)) for i in range(3)]
    if sum(matching_vertices) >= 2:  # Require at least two vertices to match
        quad_faces.append(face_id)

# Plot the entire mesh with the quadrilateral faces highlighted
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the entire building mesh as a surface
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                triangles=mesh.faces, color='gray', alpha=0.3, edgecolor='none')

# Highlight the quadrilateral faces using Poly3DCollection
for face_id in quad_faces:
    face = mesh.faces[face_id]
    vertices = mesh.vertices[face]
    # Create the polygon collection
    poly = Poly3DCollection([vertices], facecolors='blue', edgecolors='black', alpha=0.8)
    ax.add_collection3d(poly)

# Mark the four vertices
ax.scatter(quad_vertices[:, 0], quad_vertices[:, 1], quad_vertices[:, 2], color='red', s=50)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=135)

custom_lines = [
    plt.Line2D([0], [0], color="gray", lw=4, alpha=0.3, label='Building'),
    plt.Line2D([0], [0], color="blue", lw=4, alpha=0.8, label='Highlighted Faces'),
    plt.Line2D([0], [0], color="red", marker='o', markersize=8, lw=0, label='Quad Vertices')
]
ax.legend(handles=custom_lines)

plt.savefig('/home/tang/code/ReLoD3_nus/raycasting/building_quad_face_highlight.png')
plt.show()
