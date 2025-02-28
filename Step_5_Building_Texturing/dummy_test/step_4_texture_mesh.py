import cv2
import numpy as np
import os
import sys

def main():
    # ---------------------------- Configuration ---------------------------- #

    # Paths to input files
    # image_path = '/home/tang/code/ReLoD3_nus/test_3.png'
    image_path = '/home/tang/code/Semantic-SAM/examples_2/panorama.jpg'
    mesh_path = '/home/tang/code/ReLoD3_nus/raycasting/simplified_mesh/cluster_0.obj'

    # Output directory
    output_dir = '/home/tang/code/ReLoD3_nus/raycasting/textured_mesh'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Path to save the extracted texture
    texture_output_path = os.path.join(output_dir, 'extracted_texture.png')

    # Path to save the updated OBJ file
    textured_obj_path = os.path.join(output_dir, 'cluster_0_with_texture.obj')

    # Material file name and path
    mtl_filename = 'cluster_0_with_texture.mtl'
    mtl_path = os.path.join(output_dir, mtl_filename)

    # Texture file name (used in the .mtl file)
    texture_filename = 'extracted_texture.png'

    # ------------------------ Define Corner Points ------------------------- #

    # Define the four corner points of the building in the image (source points)
    # src_points = np.array([
    #     [961.9998024636279, 1643.6720976322333],  # Top-Left
    #     [321.5075332813236, 2753.499842215027],   # Bottom-Left
    #     [6980.165327507773, 3046.2112531993603],  # Bottom-Right
    #     [5224.299700774733, 705.457904603511]      # Top-Right
    # ], dtype=np.float32)
    '''Corner 1: [   5.697 2381.346]
        Corner 2: [3703.05  2347.164]
        Corner 3: [3634.686 3366.927]
        Corner 4: [  85.455 3395.412]'''
    src_points = np.array([
        [3703.05, 2347.164], 
        [3634.686, 3366.927],
        [85.455, 3395.412],  # Top-Left
        [5.697, 2381.346]  # Top-Left
    ], dtype=np.float32)

    # Define destination points for the perspective transform (destination points)
    # Corrected: width corresponds to x-axis, height to y-axis
    width, height = 2048, 1024  # Swapped to correct width and height
    dst_points = np.array([
        [0, 0],          # Top-Left
        [0, height],     # Bottom-Left
        [width, height], # Bottom-Right
        [width, 0]       # Top-Right
    ], dtype=np.float32)

    # ---------------------- Extract the Texture Quad ------------------------ #

    # Load the building image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    else:
        print(f"Loaded image from {image_path}")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    print("Computed perspective transform matrix.")

    # Apply the perspective transform to extract the texture
    texture = cv2.warpPerspective(image, M, (width, height))
    print("Applied perspective transform to extract texture.")

    # Save the extracted texture image
    cv2.imwrite(texture_output_path, texture)
    print(f"Extracted texture saved to {texture_output_path}")

    # --------------------------- Load the Mesh ------------------------------ #

    # Load the simplified mesh
    try:
        with open(mesh_path, 'r') as mesh_file:
            mesh_lines = mesh_file.readlines()
        print(f"Loaded mesh from {mesh_path}")
    except FileNotFoundError:
        print(f"Error: Mesh file not found at {mesh_path}")
        sys.exit(1)

    # Parse vertices and faces
    vertices = []
    faces = []
    for line in mesh_lines:
        if line.startswith('v '):
            parts = line.strip().split()
            vertex = list(map(float, parts[1:4]))
            vertices.append(vertex)
        elif line.startswith('f '):
            parts = line.strip().split()
            # OBJ indices are 1-based
            face = [int(part.split('/')[0]) for part in parts[1:4]]
            faces.append(face)

    # Verify that the mesh has exactly 4 vertices and 2 faces
    if len(vertices) != 4:
        print(f"Error: Mesh does not have 4 vertices. It has {len(vertices)} vertices.")
        sys.exit(1)
    if len(faces) != 2:
        print(f"Error: Mesh does not have 2 faces. It has {len(faces)} faces.")
        sys.exit(1)
    else:
        print("Mesh has correct number of vertices and faces.")

    # ---------------------- Assign UV Coordinates ---------------------------- #

    # Define UV coordinates corresponding to the texture corners
    # Assuming the mesh vertices are ordered to match the src_points order
    # UV coordinates range from (0,0) to (1,1)
    uv_coords = np.array([
        [0.0, 0.0],  # Corresponds to first vertex (Top-Left)
        [0.0, 1.0],  # Corresponds to second vertex (Bottom-Left)
        [1.0, 1.0],  # Corresponds to third vertex (Bottom-Right)
        [1.0, 0.0]   # Corresponds to fourth vertex (Top-Right)
    ], dtype=np.float32)

    # -------------------------- Create MTL File ------------------------------ #

    # Define material name
    material_name = 'material_0'

    # Create the content of the .mtl file
    mtl_content = f"""newmtl {material_name}
    Ka 1.000 1.000 1.000
    Kd 1.000 1.000 1.000
    Ks 0.000 0.000 0.000
    d 1.0
    illum 2
    map_Kd {texture_filename}
    """

    # Write the .mtl file to the output directory
    with open(mtl_path, 'w') as mtl_file:
        mtl_file.write(mtl_content)
    print(f"MTL file saved to {mtl_path}")

    # ------------------------ Write OBJ File Manually ------------------------- #

    # Prepare the OBJ content
    obj_lines = []

    # Reference the MTL file
    obj_lines.append(f"mtllib {mtl_filename}\n")

    # Write vertex positions
    for vertex in vertices:
        obj_lines.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

    # Write texture coordinates
    for uv in uv_coords:
        obj_lines.append(f"vt {uv[0]} {uv[1]}\n")

    # Use the defined material
    obj_lines.append(f"usemtl {material_name}\n")

    # Write faces with vertex and texture indices
    for face in faces:
        v1, v2, v3 = face
        obj_lines.append(f"f {v3}/{v3} {v2}/{v2} {v1}/{v1}\n")

    with open(textured_obj_path, 'w') as obj_file:
        obj_file.writelines(obj_lines)
    print(f"Textured OBJ file saved to {textured_obj_path}")

    print("\nProcess completed successfully.")

if __name__ == "__main__":
    main()
