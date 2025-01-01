import os
import numpy as np
import open3d as o3d


def load_point_cloud(pc_path):
    """
    Load point cloud from an npz file.

    Args:
        pc_path (str): Path to the .npz file containing the point cloud.

    Returns:
        np.ndarray: Point cloud data of shape (N, 3) or (N, 6).
    """
    if not os.path.isfile(pc_path):
        raise FileNotFoundError(f"Point cloud file not found: {pc_path}")

    data = np.load(pc_path)
    if 'pc' not in data:
        raise KeyError(f"'pc' key not found in the npz file: {pc_path}")

    point_cloud = data['pc']
    print(f"Loaded point cloud with shape: {point_cloud.shape}")
    return point_cloud


def load_bounding_boxes(bb_path):
    """
    Load bounding boxes from an npy file.

    Args:
        bb_path (str): Path to the .npy file containing bounding boxes.

    Returns:
        np.ndarray: Bounding boxes of shape (num_boxes, 8).
    """
    if not os.path.isfile(bb_path):
        raise FileNotFoundError(f"Bounding boxes file not found: {bb_path}")

    bboxes = np.load(bb_path)
    if bboxes.ndim != 2 or bboxes.shape[1] != 8:
        raise ValueError(f"Bounding boxes should have shape (num_boxes, 8). Found: {bboxes.shape}")

    print(f"Loaded {bboxes.shape[0]} bounding boxes.")
    return bboxes


def load_votes(votes_path):
    """
    Load ground truth votes from an npz file.

    Args:
        votes_path (str): Path to the .npz file containing votes.

    Returns:
        np.ndarray: Votes of shape (num_points, 10).
    """
    if not os.path.isfile(votes_path):
        raise FileNotFoundError(f"Votes file not found: {votes_path}")

    data = np.load(votes_path)
    if 'point_votes' not in data:
        raise KeyError(f"'point_votes' key not found in the npz file: {votes_path}")

    votes = data['point_votes']
    if votes.ndim != 2 or votes.shape[1] != 10:
        raise ValueError(f"Votes should have shape (num_points, 10). Found: {votes.shape}")

    print(f"Loaded votes with shape: {votes.shape}")
    return votes


def create_oriented_bounding_box(center, size, heading_angle, color=(1, 0, 0)):
    """
    Create an Open3D Oriented Bounding Box.

    Args:
        center (array-like): Center of the bounding box (3,).
        size (array-like): Size of the bounding box (length, width, height).
        heading_angle (float): Heading angle in radians.
        color (tuple): RGB color for the bounding box.

    Returns:
        open3d.geometry.OrientedBoundingBox: The oriented bounding box.
    """
    # Create a rotation matrix around the Z-axis
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, heading_angle])

    # Create the bounding box
    obb = o3d.geometry.OrientedBoundingBox(center, R, size)
    obb.color = color
    return obb


def create_arrow(start, end, color=(0, 0, 1), cylinder_radius=0.005, cone_radius=0.01, cone_height=0.02):
    """
    Create an arrow from start to end points.

    Args:
        start (array-like): Starting point of the arrow (3,).
        end (array-like): Ending point of the arrow (3,).
        color (tuple): RGB color of the arrow.
        cylinder_radius (float): Radius of the arrow's shaft.
        cone_radius (float): Radius of the arrow's head.
        cone_height (float): Height of the arrow's head.

    Returns:
        open3d.geometry.TriangleMesh: The arrow mesh.
    """
    # Create a cylinder (arrow shaft)
    shaft_length = np.linalg.norm(end - start) - cone_height
    if shaft_length < 0:
        shaft_length = 0
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=shaft_length)
    cylinder.paint_uniform_color(color)

    # Create a cone (arrow head)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    cone.paint_uniform_color(color)

    # Align the cylinder with the arrow direction
    direction = end - start
    if np.linalg.norm(direction) == 0:
        rot_matrix = np.identity(3)
    else:
        direction_norm = direction / np.linalg.norm(direction)
        rot_matrix = get_rotation_matrix_from_vectors([0, 0, 1], direction_norm)
    cylinder.rotate(rot_matrix, center=(0, 0, 0))
    cone.rotate(rot_matrix, center=(0, 0, 0))

    # Translate the cylinder and cone to the start point
    cylinder.translate(start + direction_norm * (shaft_length / 2) if np.linalg.norm(direction) != 0 else start)
    cone.translate(start + direction_norm * shaft_length if np.linalg.norm(direction) != 0 else start)

    # Combine cylinder and cone
    arrow = cylinder + cone
    return arrow


def get_rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return rot_matrix: A transform matrix (3x3) which aligns vec1 with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    if np.allclose(a, b):
        return np.identity(3)
    if np.allclose(a, -b):
        # 180 degrees rotation around any orthogonal vector
        ortho = np.array([1, 0, 0])
        if np.allclose(a, ortho) or np.allclose(a, -ortho):
            ortho = np.array([0, 1, 0])
        v = np.cross(a, ortho)
        v /= np.linalg.norm(v)
        H = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        rot_matrix = -np.identity(3) + 2 * np.outer(v, v)
        return rot_matrix
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    rot_matrix = np.identity(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))
    return rot_matrix


def visualize_scene(pc, bboxes, votes):
    """
    Visualize the point cloud, bounding boxes, and votes using Open3D.

    Args:
        pc (np.ndarray): Point cloud data.
        bboxes (np.ndarray): Bounding boxes data.
        votes (np.ndarray): Votes data.
    """
    # Create Open3D point cloud
    if pc.shape[1] >= 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

        # Optionally, color the point cloud if it has color information
        if pc.shape[1] >= 6:
            colors = pc[:, 3:6]
            # Normalize colors to [0,1] if they are not already
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Assign a default color
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
    else:
        raise ValueError(f"Point cloud has insufficient dimensions: {pc.shape}")

    geometries = [pcd]

    # Define a color map for semantic classes (up to 10 classes)
    color_map = [
        [1, 0, 0],  # Class 0: Red
        [0, 1, 0],  # Class 1: Green
        [0, 0, 1],  # Class 2: Blue
        [1, 1, 0],  # Class 3: Yellow
        [0.5, 0.5, 0.5],  # Class 4: Gray
        [1, 0, 1],  # Class 5: Magenta
        [0, 1, 1],  # Class 6: Cyan
        [1, 0.5, 0],  # Class 7: Orange
        [0.5, 0, 0.5],  # Class 8: Purple
        [0, 0.5, 0.5]  # Class 9: Teal
    ]

    # Visualize Bounding Boxes
    for bbox in bboxes:
        center = bbox[0:3]
        size = bbox[3:6]
        heading_angle = bbox[6]
        sem_cls = int(bbox[7])

        # Assign color based on semantic class
        if sem_cls < len(color_map):
            color = color_map[sem_cls]
        else:
            color = [0, 0, 0]  # Default to black if class index is out of range

        obb = create_oriented_bounding_box(center, size, heading_angle, color=color)
        geometries.append(obb)

    # Visualize Votes as Arrows
    # Each vote consists of 3 offsets per point, total 9 values
    # We'll visualize all votes for each valid point
    vote_label_mask = votes[:, 0]
    valid_vote_indices = np.where(vote_label_mask == 1)[0]

    for idx in valid_vote_indices:
        point = pc[idx, :3]
        vote1 = votes[idx, 1:4]
        vote2 = votes[idx, 4:7]
        vote3 = votes[idx, 7:10]

        # Collect all non-zero votes
        votes_to_visualize = []
        if np.linalg.norm(vote1) > 0:
            votes_to_visualize.append(vote1)
        if np.linalg.norm(vote2) > 0:
            votes_to_visualize.append(vote2)
        if np.linalg.norm(vote3) > 0:
            votes_to_visualize.append(vote3)

        for vote in votes_to_visualize:
            end_point = point + vote
            arrow = create_arrow(point, end_point, color=(0, 0, 1))
            geometries.append(arrow)

    # Create Open3D visualizer and add all geometries
    o3d.visualization.draw_geometries(
        geometries,
        zoom=0.8,
        front=[-0.4999, -0.1659, -0.8499],
        lookat=[0, 0, 0],
        up=[0.1204, -0.9852, 0.1215]
    )


def main():
    """
    Main function to load data and visualize the scene.
    """
    # ---------------------- User-Defined Paths ----------------------
    # Set the paths to your files here
    point_cloud_path = 'Path to Project directory/ProcessedDataset/test/point_clouds/2011_09_26_drive_0001_sync_0000000007_shot_1.npz'  # Replace with your point cloud path
    bounding_boxes_path = 'Path to Project directory/ProcessedDataset/test/bounding_boxes/2011_09_26_drive_0001_sync_0000000007_shot_1.npy'  # Replace with your bounding boxes path
    votes_path = 'Path to Project directory/ProcessedDataset/test/ground_truth_votes/2011_09_26_drive_0001_sync_0000000007_shot_1.npz'  # Replace with your votes path
    # -----------------------------------------------------------------

    # Load data
    point_cloud = load_point_cloud(point_cloud_path)
    bounding_boxes = load_bounding_boxes(bounding_boxes_path)
    votes = load_votes(votes_path)

    # Validate data lengths
    if point_cloud.shape[0] != votes.shape[0]:
        raise ValueError(
            f"Number of points in point cloud ({point_cloud.shape[0]}) does not match number of votes ({votes.shape[0]}).")

    # Visualize
    visualize_scene(point_cloud, bounding_boxes, votes)


def create_arrow(start, end, color=(0, 0, 1), cylinder_radius=0.005, cone_radius=0.01, cone_height=0.02):
    """
    Create an arrow from start to end points.

    Args:
        start (array-like): Starting point of the arrow (3,).
        end (array-like): Ending point of the arrow (3,).
        color (tuple): RGB color of the arrow.
        cylinder_radius (float): Radius of the arrow's shaft.
        cone_radius (float): Radius of the arrow's head.
        cone_height (float): Height of the arrow's head.

    Returns:
        open3d.geometry.TriangleMesh: The arrow mesh.
    """
    # Create a cylinder (arrow shaft)
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm == 0:
        return o3d.geometry.TriangleMesh()  # Return empty mesh for zero-length arrows
    shaft_length = norm - cone_height
    if shaft_length < 0:
        shaft_length = 0
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=shaft_length)
    cylinder.paint_uniform_color(color)

    # Create a cone (arrow head)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    cone.paint_uniform_color(color)

    # Align the cylinder with the arrow direction
    direction_norm = direction / norm
    rot_matrix = get_rotation_matrix_from_vectors([0, 0, 1], direction_norm)
    cylinder.rotate(rot_matrix, center=(0, 0, 0))
    cone.rotate(rot_matrix, center=(0, 0, 0))

    # Translate the cylinder and cone to the start point
    cylinder.translate(start + direction_norm * (shaft_length / 2))
    cone.translate(start + direction_norm * shaft_length)

    # Combine cylinder and cone
    arrow = cylinder + cone
    return arrow


def get_rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return rot_matrix: A transform matrix (3x3) which aligns vec1 with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    if np.allclose(a, b):
        return np.identity(3)
    if np.allclose(a, -b):
        # 180 degrees rotation around any orthogonal vector
        ortho = np.array([1, 0, 0])
        if np.allclose(a, ortho) or np.allclose(a, -ortho):
            ortho = np.array([0, 1, 0])
        v = np.cross(a, ortho)
        v /= np.linalg.norm(v)
        rot_matrix = -np.identity(3) + 2 * np.outer(v, v)
        return rot_matrix
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    rot_matrix = np.identity(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))
    return rot_matrix


if __name__ == "__main__":
    main()
