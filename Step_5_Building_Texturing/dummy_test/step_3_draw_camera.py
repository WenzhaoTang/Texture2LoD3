import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the building model
# mesh = trimesh.load('/home/tang/code/ReLoD3_nus/raycasting/simplified_mesh/simplified_2nd.obj')
mesh = trimesh.load('/home/tang/code/ReLoD3_nus/raycasting/simplified_mesh/cluster_0.obj')

# Define camera position and ray direction
camera_position = np.array([[40, 0, 10]])
ray_directions = np.array([[-1, 2, 0]])  # Ray directed towards the building

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                triangles=mesh.faces, color='blue', alpha=0.5, edgecolor='none')

# Plot the camera pos
ax.scatter(camera_position[:, 0], camera_position[:, 1], camera_position[:, 2],
           color='red', s=50)

# Draw ray lines
ray_end = camera_position + ray_directions * 20  # Extend the ray for visibility in the plot
for origin, end in zip(camera_position, ray_end):
    ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], 'r--', linewidth=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=135)

custom_lines = [plt.Line2D([0], [0], color="blue", lw=4, alpha=0.5),
                plt.Line2D([0], [0], color="red", marker='o', markersize=8, lw=0)]
ax.legend(custom_lines, ['Building', 'Camera Position'])

plt.savefig('/home/tang/code/ReLoD3_nus/raycasting/building_camera_position.png')
plt.show()
