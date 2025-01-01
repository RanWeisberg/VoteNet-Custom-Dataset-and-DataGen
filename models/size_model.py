import os
import numpy as np
from plyfile import PlyData

def load_and_print_sizes(objects_dir):
    """
    Load point cloud files from the specified directory and print the size of each object.

    Args:
        objects_dir (str): Directory containing the objects (PC folder).
    """
    # Get object files from the PC folder
    pc_dir = os.path.join(objects_dir, 'PC')
    object_files = [f for f in os.listdir(pc_dir) if f.endswith('.ply')]

    if not object_files:
        raise FileNotFoundError(f"No .ply files found in the directory: {pc_dir}")

    print(f"Object files found: {object_files}")

    for obj_file in object_files:
        obj_name = os.path.splitext(obj_file)[0]

        # File path for the PC file
        ply_path = os.path.join(pc_dir, obj_file)

        # Load object points
        ply_data = PlyData.read(ply_path)
        object_points = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T

        # Calculate the size of the object
        min_coords = np.min(object_points, axis=0)
        max_coords = np.max(object_points, axis=0)
        size = max_coords - min_coords

        print(f"Object: {obj_name}, Size (LxWxH): {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")

# Example usage
if __name__ == "__main__":
    objects_dir = "Path to Project directory/models"
    load_and_print_sizes(objects_dir)
