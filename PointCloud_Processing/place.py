import os
import numpy as np
from plyfile import PlyData
from render_to_point_cloud import render_to_depth_buffer

TARGET_NUM_POINTS = 20000
MAX_PLACEMENT_ATTEMPTS = 10

def define_scene_area(scene_points):
    """Calculate the bounding box of the scene."""
    min_x = np.min(scene_points[:, 0])
    max_x = np.max(scene_points[:, 0])
    min_y = np.min(scene_points[:, 1])
    max_y = np.max(scene_points[:, 1])
    return min_x, max_x, min_y, max_y

def find_floor_and_ceiling(scene_points, bin_width=0.1, height_threshold=0.1):
    """Find the floor and ceiling of the scene."""
    z_coordinates = scene_points[:, 2]
    hist, bin_edges = np.histogram(z_coordinates, bins=int((z_coordinates.max() - z_coordinates.min()) / bin_width))
    floor_height = bin_edges[np.argmax(hist)]
    ceiling_height = z_coordinates.max()
    if ceiling_height - floor_height < height_threshold:
        raise ValueError("Detected floor and ceiling are too close. Check point cloud data.")
    return floor_height, ceiling_height

def random_heading_angle():
    """Generate a random heading angle."""
    return np.random.uniform(-np.pi, np.pi)

def rotation_matrix_from_heading(heading_angle):
    """Generate a rotation matrix from a heading angle."""
    return np.array([[np.cos(heading_angle), -np.sin(heading_angle), 0],
                     [np.sin(heading_angle), np.cos(heading_angle), 0],
                     [0, 0, 1]])

def random_translation(object_points, min_x, max_x, min_y, max_y, floor_height, ceiling_height):
    """Generate a random translation vector within the scene boundaries."""
    object_height = np.max(object_points[:, 2]) - np.min(object_points[:, 2])
    x = np.random.uniform(min_x, max_x)
    y = np.random.uniform(min_y, max_y)
    z = floor_height + 0.1  # Place the object slightly above the floor
    return np.array([x, y, z])

def check_no_overlap(scene, obj, tolerance=0.1):
    """Check if the object overlaps with the scene points."""
    for point in obj:
        if np.min(np.linalg.norm(scene - point, axis=1)) < tolerance:
            return False
    return True

def get_bounding_box_parameters(object_points, rotation_matrix, translation_vector, heading_angle):
    """Calculate the bounding box parameters for an object."""
    min_coords = np.min(object_points, axis=0)
    max_coords = np.max(object_points, axis=0)
    size = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    transformed_center = center @ rotation_matrix.T + translation_vector
    return transformed_center, size, heading_angle

def adjust_point_cloud_size(point_cloud, target_num_points):
    """Adjust the point cloud to the target number of points."""
    current_num_points = point_cloud.shape[0]
    if current_num_points > target_num_points:
        choice = np.random.choice(current_num_points, target_num_points, replace=False)
        adjusted_point_cloud = point_cloud[choice, :]
    elif current_num_points < target_num_points:
        choice = np.random.choice(current_num_points, target_num_points, replace=True)
        adjusted_point_cloud = point_cloud[choice, :]
    else:
        adjusted_point_cloud = point_cloud
    return adjusted_point_cloud

def get_object_classes_from_filenames(pc_dir):
    """Generate a list of object classes from the filenames in the PC directory."""
    object_files = [os.path.splitext(f)[0] for f in os.listdir(pc_dir) if f.endswith('.ply')]
    return sorted(object_files)

def calculate_combined_heading_angle(object_center, intrinsic_rotation_angle):
    """Calculate the heading angle by combining intrinsic rotation and angle between object center and scene Y-axis."""
    angle_to_y_axis = np.arctan2(object_center[0], object_center[1])
    combined_angle = intrinsic_rotation_angle + angle_to_y_axis
    return combined_angle

def compute_box_corners(center, size, heading_angle):
    """Compute the 8 corners of the bounding box."""
    l, w, h = size
    # Create a bounding box centered at the origin
    x_corners = l / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners)).T  # (8, 3)
    # Rotate and translate the corners
    R = rotation_matrix_from_heading(heading_angle)
    corners = corners @ R.T + center
    return corners

def extract_points_in_box(pc, box_corners):
    """Extract indices of points inside the given bounding box."""
    # Compute the axis-aligned bounding box for quick exclusion
    xmin = np.min(box_corners[:, 0])
    xmax = np.max(box_corners[:, 0])
    ymin = np.min(box_corners[:, 1])
    ymax = np.max(box_corners[:, 1])
    zmin = np.min(box_corners[:, 2])
    zmax = np.max(box_corners[:, 2])
    in_box_mask = (
        (pc[:, 0] >= xmin) & (pc[:, 0] <= xmax) &
        (pc[:, 1] >= ymin) & (pc[:, 1] <= ymax) &
        (pc[:, 2] >= zmin) & (pc[:, 2] <= zmax)
    )
    indices = np.where(in_box_mask)[0]
    return indices

def compute_votes(points, bounding_boxes):
    """
    Compute ground truth votes for each point in the point cloud.

    Parameters:
        points (numpy.ndarray): The point cloud (N, 3), where N is the number of points.
        bounding_boxes (list of numpy.ndarray): List of bounding boxes, each of shape (8,)
            where the elements are [cx, cy, cz, l, w, h, heading_angle, label_index].

    Returns:
        point_votes (numpy.ndarray): The ground truth votes for each point (N, 10).
    """
    num_points = points.shape[0]
    point_votes = np.zeros((num_points, 10), dtype=np.float32)
    point_vote_idx = np.zeros(num_points, dtype=int)  # Keeps track of the number of votes per point

    for bbox in bounding_boxes:
        bbox_center = bbox[:3]  # cx, cy, cz
        bbox_size = bbox[3:6]   # l, w, h
        heading_angle = bbox[6] # heading_angle

        # Compute the bounding box corners
        box_corners = compute_box_corners(bbox_center, bbox_size, heading_angle)

        # Determine which points are within the bounding box
        inds = extract_points_in_box(points, box_corners)

        # For each point inside the bounding box, calculate the vote offsets
        votes = bbox_center - points[inds, :3]

        for i, idx in enumerate(inds):
            vote_count = point_vote_idx[idx]
            if vote_count < 3:
                start_idx = 1 + 3 * vote_count
                point_votes[idx, 0] = 1  # Set vote mask to 1
                point_votes[idx, start_idx:start_idx+3] = votes[i]
                point_vote_idx[idx] += 1

    return point_votes

def place_objects_in_scene(scene_path, objects_dir, output_scene_path, labels_output_path, votes_output_path):
    """
    Place objects in the scene and save the scene, labels, and votes.
    """
    scene_cloud = np.fromfile(scene_path, dtype=np.float32).reshape(-1, 4)
    scene_points = scene_cloud[:, :3]

    min_x, max_x, min_y, max_y = define_scene_area(scene_points)
    floor_height, ceiling_height = find_floor_and_ceiling(scene_points)

    combined_points = scene_points.copy()
    object_list = []

    pc_dir = os.path.join(objects_dir, 'PC')
    object_files = [f for f in os.listdir(pc_dir) if f.endswith('.ply')]
    object_classes = get_object_classes_from_filenames(pc_dir)
    objects_to_place = object_files.copy()

    while objects_to_place:
        obj_file = objects_to_place[0]
        obj_name = os.path.splitext(obj_file)[0]
        ply_path = os.path.join(pc_dir, obj_file)
        stl_path = os.path.join(objects_dir, 'STL', f'{obj_name}.stl')

        ply_data = PlyData.read(ply_path)
        object_points = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T

        placed = False
        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            intrinsic_rotation_angle = random_heading_angle()
            rotation_matrix = rotation_matrix_from_heading(intrinsic_rotation_angle)
            translation_vector = random_translation(object_points, min_x, max_x, min_y, max_y, floor_height,
                                                    ceiling_height)

            transformed_points = object_points @ rotation_matrix.T + translation_vector
            if check_no_overlap(combined_points, transformed_points):
                object_center = translation_vector
                combined_heading_angle = calculate_combined_heading_angle(object_center, intrinsic_rotation_angle)
                partial_scan = render_to_depth_buffer(stl_path, combined_heading_angle)
                transformed_partial_scan = partial_scan @ rotation_matrix.T + translation_vector

                combined_points = np.vstack([combined_points, transformed_partial_scan])

                center, size, heading_angle = get_bounding_box_parameters(object_points, rotation_matrix,
                                                                          translation_vector, intrinsic_rotation_angle)

                label_index = object_classes.index(obj_name)
                object_list.append(np.array([*center, *size, heading_angle, label_index]))

                placed = True
                break

        if placed:
            objects_to_place.pop(0)
        else:
            print(f"Failed to place object {obj_name} after {MAX_PLACEMENT_ATTEMPTS} attempts.")
            objects_to_place.pop(0)

    # Adjust point cloud size before computing votes
    combined_points = adjust_point_cloud_size(combined_points, TARGET_NUM_POINTS)

    # Compute Ground Truth Votes after resizing the point cloud
    point_votes = compute_votes(combined_points, object_list)

    np.savez_compressed(output_scene_path, pc=combined_points)
    np.save(labels_output_path, np.array(object_list))

    # Save Ground Truth Votes
    np.savez_compressed(votes_output_path, point_votes=point_votes)

    print(
        f"Scene saved to {output_scene_path}, labels saved to {labels_output_path}, votes saved to {votes_output_path}")

# Example usage
if __name__ == "__main__":
    scene_path = "Path to Project directory/PointCloud_Processing/test/velodyne_points_0000000022_shot_1.bin"
    objects_dir = "Path to Project directory/models"
    output_scene_path = "Path to Project directory/PointCloud_Processing/test/results/example_scene.npz"
    labels_output_path = "Path to Project directory/PointCloud_Processing/test/results/example_labels.npy"
    votes_output_path = "Path to Project directory/PointCloud_Processing/test/results/example_votes.npz"

    place_objects_in_scene(scene_path, objects_dir, output_scene_path, labels_output_path, votes_output_path)
