import numpy as np
import os


def split_point_cloud_with_vertical_crop(file_path, output_dir, fov_horizontal_deg=69.0, fov_vertical_deg=42.0):
    """
    Splits a 360° point cloud into multiple "shots" horizontally and crops vertically
    as if taken by a camera with specified horizontal and vertical fields of view.

    Args:
        file_path (str): Path to the .bin file containing the 360° point cloud data.
        output_dir (str): Directory to save the split and cropped point cloud files.
        fov_horizontal_deg (float): Horizontal field of view of the camera in degrees.
        fov_vertical_deg (float): Vertical field of view of the camera in degrees.
    """
    # Load the point cloud from the .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    # Convert FOVs to radians
    fov_horizontal_rad = np.deg2rad(fov_horizontal_deg)
    fov_vertical_rad = np.deg2rad(fov_vertical_deg)

    # Calculate the number of horizontal segments
    num_horizontal_segments = int(2 * np.pi / fov_horizontal_rad)

    # Calculate the vertical angle limits
    min_elevation = -fov_vertical_rad / 2
    max_elevation = fov_vertical_rad / 2

    # Extract the relevant parts for naming from the file path
    # Get the `2011_09_26_drive_0011_sync` part by traversing the directory structure
    parent_dir = os.path.dirname(file_path)  # Gets the `data` directory
    drive_name = os.path.basename(os.path.dirname(os.path.dirname(parent_dir)))  # Gets `2011_09_26_drive_0011_sync`

    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # e.g., "0000000130"

    # Iterate over the horizontal segments
    for h in range(num_horizontal_segments):
        # Calculate the angular range for the current segment
        min_azimuth = -np.pi + h * fov_horizontal_rad
        max_azimuth = min_azimuth + fov_horizontal_rad

        # Calculate the center of the current segment
        center_azimuth = (min_azimuth + max_azimuth) / 2

        # Filter the points within the current segment's FOV
        filtered_points = []
        for point in point_cloud:
            x, y, z, intensity = point
            azimuth = np.arctan2(y, x)
            distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            elevation = np.arcsin(z / distance)

            if min_azimuth <= azimuth < max_azimuth and min_elevation <= elevation < max_elevation:
                # Transform to new coordinate system centered at the camera
                # Rotate around the z-axis to align the camera's view direction with the segment center
                rotation_angle = -center_azimuth
                rotation_matrix = np.array([
                    [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                    [0, 0, 1]
                ])

                # Apply the rotation
                original_point = np.array([x, y, z])
                rotated_point = rotation_matrix @ original_point

                # Swap and reorient axes to match the desired output coordinate system
                # The desired system: z-axis is up, y-axis is forward, x-axis is right
                transformed_x = rotated_point[1]  # original y becomes x
                transformed_y = rotated_point[0]  # original x becomes y
                transformed_z = rotated_point[2]  # original z remains z

                filtered_points.append([transformed_x, transformed_y, transformed_z, intensity])

        # Save the filtered points to a new .bin file with a unique name
        if filtered_points:
            filtered_points = np.array(filtered_points, dtype=np.float32)
            output_file = os.path.join(output_dir, f"{drive_name}_{file_name}_shot_{h}.bin")
            filtered_points.tofile(output_file)


# Example usage
if __name__ == "__main__":
    file_path = "Path to Project directory/Kitti Files/2011_09_26_drive_0011_sync/velodyne_points/data/0000000130.bin"
    output_dir = "Path to Project directory/PointCloud_Processing/test/results"
    fov_horizontal_deg = 69.0
    fov_vertical_deg = 42.0

    split_point_cloud_with_vertical_crop(file_path, output_dir, fov_horizontal_deg, fov_vertical_deg)
