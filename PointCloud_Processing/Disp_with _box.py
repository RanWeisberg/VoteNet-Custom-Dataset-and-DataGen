import numpy as np
import open3d as o3d
import json


def load_labels(labels_path):
    """
    Load the labels from a JSON, NPZ, or NPY file.

    Args:
        labels_path (str): Path to the labels file.

    Returns:
        list: A list of dictionaries, or a list of arrays, each containing bounding box parameters and class information.
    """
    if labels_path.endswith('.json'):
        with open(labels_path, 'r') as f:
            label_data = json.load(f)
    elif labels_path.endswith('.npz'):
        label_data = np.load(labels_path, allow_pickle=True)['bboxes'].tolist()
    elif labels_path.endswith('.npy'):
        label_data = np.load(labels_path, allow_pickle=True).tolist()
    else:
        raise ValueError(f"Unsupported label file format: {labels_path}")

    return label_data


def create_bounding_box(center, size, heading_angle):
    """
    Create a 3D oriented bounding box from center, size, and heading angle.

    Args:
        center (tuple): (x, y, z) coordinates of the center of the box.
        size (tuple): (l, w, h) dimensions of the box.
        heading_angle (float): Heading angle in radians.

    Returns:
        o3d.geometry.OrientedBoundingBox: The bounding box object.
    """
    center = np.array(center)
    size = np.array(size)

    rotation_matrix = np.array([[np.cos(heading_angle), -np.sin(heading_angle), 0],
                                [np.sin(heading_angle), np.cos(heading_angle), 0],
                                [0, 0, 1]])

    obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, size)
    return obb


def load_point_cloud(point_cloud_path):
    """
    Load a point cloud from a file, supporting various formats.

    Args:
        point_cloud_path (str): Path to the point cloud file.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud object.
    """
    if point_cloud_path.endswith('.ply'):
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    elif point_cloud_path.endswith('.npz'):
        data = np.load(point_cloud_path)
        points = data['points']
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    elif point_cloud_path.endswith('.bin'):
        points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    elif point_cloud_path.endswith('.npy'):
        points = np.load(point_cloud_path)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    else:
        raise ValueError(f"Unsupported point cloud file format: {point_cloud_path}")

    return point_cloud


def visualize_point_cloud_with_bounding_boxes(point_cloud_path, labels_path=None):
    """
    Visualize the point cloud with the corresponding bounding boxes, if provided.

    Args:
        point_cloud_path (str): Path to the point cloud file.
        labels_path (str, optional): Path to the labels file. If None, only the point cloud is displayed.
    """
    # Load point cloud
    point_cloud = load_point_cloud(point_cloud_path)

    # Prepare bounding boxes for visualization if labels are provided
    bounding_boxes = []
    if labels_path is not None:
        label_data_list = load_labels(labels_path)

        # Check if data is a list of dictionaries or list of lists/arrays
        if isinstance(label_data_list[0], dict):
            # If the data is a list of dictionaries, use keys to access data
            for label_data in label_data_list:
                try:
                    center = label_data['center']
                    size = label_data['size']
                    heading_angle = label_data['heading_angle']
                    print(f"Creating BBox: Center={center}, Size={size}, Heading={heading_angle}")
                except KeyError as e:
                    print(f"Missing key {e} in label data: {label_data}")
                    continue  # Skip this entry if data is missing
                bounding_box = create_bounding_box(center, size, heading_angle)
                bounding_box.color = [1, 0, 0]  # Red color for bounding boxes
                bounding_boxes.append(bounding_box)
        elif isinstance(label_data_list[0], (list, np.ndarray)):
            # If the data is a list of lists/arrays, access data by index
            for label_data in label_data_list:
                if len(label_data) < 7:
                    print(f"Label data has insufficient length: {label_data}")
                    continue  # Skip this entry if data is incomplete
                center = label_data[0:3]
                size = label_data[3:6]
                heading_angle = label_data[6]
                print(f"Creating BBox: Center={center}, Size={size}, Heading={heading_angle}")
                bounding_box = create_bounding_box(center, size, heading_angle)
                bounding_box.color = [1, 0, 0]  # Red color for bounding boxes
                bounding_boxes.append(bounding_box)
        else:
            raise ValueError("Unexpected label data format.")

    if not bounding_boxes:
        print("No bounding boxes were created.")

    # Visualize
    o3d.visualization.draw_geometries([point_cloud, *bounding_boxes])


# Example usage
output_path = "Path to the PC npz file"
labels_output_path = "Path to npy labels file"
visualize_point_cloud_with_bounding_boxes(output_path, labels_output_path)

