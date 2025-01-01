import open3d as o3d
import os
import numpy as np

def convert_stl_directory_to_point_clouds(stl_dir, output_dir, num_points=10000, file_format="pcd", scale_to_meters=True, display=False):
    """
    Converts all STL files in a directory to point clouds and saves them to the specified output directory.
    Optionally scales the point clouds from mm to m and displays them one by one.

    Args:
        stl_dir (str): Path to the directory containing STL files.
        output_dir (str): Directory where the output point cloud files will be saved.
        num_points (int): Number of points to sample from each mesh surface.
        file_format (str): Desired output file format ('pcd', 'ply', etc.).
        scale_to_meters (bool): If True, scales the point cloud from mm to meters.
        display (bool): If True, display the point clouds after conversion.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Transformation matrix to switch coordinate system
    transformation_matrix = np.array([[0, 1, 0],
                                      [1, 0, 0],
                                      [0, 0, 1]])

    # Iterate over all files in the STL directory
    for filename in os.listdir(stl_dir):
        if filename.lower().endswith(".stl"):
            # Construct the full path to the STL file
            stl_file_path = os.path.join(stl_dir, filename)

            # Load the mesh from the STL file
            mesh = o3d.io.read_triangle_mesh(stl_file_path)

            # Check if the mesh is empty
            if not mesh.has_triangles():
                print(f"Warning: The file {filename} does not contain any triangles.")
                continue

            # Sample points uniformly from the mesh
            point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

            # Apply transformation to switch coordinate system
            point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) @ transformation_matrix.T)

            # Scale the point cloud if needed
            if scale_to_meters:
                point_cloud.scale(0.001, center=(0, 0, 0))  # Scale from mm to meters

            # Generate the output file name based on the input STL file name
            base_name = os.path.splitext(filename)[0]
            output_file_path = os.path.join(output_dir, f"{base_name}.{file_format}")

            # Save the point cloud to the specified file
            o3d.io.write_point_cloud(output_file_path, point_cloud)
            print(f"Point cloud saved to {output_file_path}")

            # Optionally display the point cloud
            if display:
                print(f"Displaying point cloud for {filename}")
                o3d.visualization.draw_geometries([point_cloud])

# Example usage
stl_dir = "Path to Project directory/models/STL"  # Replace with the directory containing STL files
output_dir = "Path to Project directory/models/PC"  # Replace with the desired output directory
file_format = "ply"  # Choose between 'pcd', 'ply', etc.
display_results = True  # Set to True if you want to display the point clouds

# Convert all STL files in the directory to point clouds, scale them to meters, save them, and optionally display
convert_stl_directory_to_point_clouds(stl_dir, output_dir, num_points=500, file_format=file_format,
                                      scale_to_meters=True, display=display_results)
