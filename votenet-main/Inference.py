import os
import torch
import numpy as np
import open3d as o3d
from utils.pc_util import random_sampling
from models.ap_helper import parse_predictions
from data_config import CustomDatasetConfig
from models.votenet import VoteNet
import pickle

def preprocess_point_cloud(pc_file, num_points, use_height=True):
    """Preprocess the point cloud data for inference."""
    point_cloud = np.load(pc_file)['pc']
    if point_cloud.shape[0] != num_points:
        point_cloud, _ = random_sampling(point_cloud, num_points, return_choices=True)

    if use_height:
        floor_height = np.percentile(point_cloud[:, 2], 0.99)
        height = point_cloud[:, 2] - floor_height
        height = np.expand_dims(height, axis=1)
        point_cloud = np.concatenate([point_cloud, height], axis=1)

    return np.expand_dims(point_cloud.astype(np.float32), axis=0)

def rotate_point_cloud(pc, rotation_matrix):
    """Apply rotation to a point cloud."""
    return np.dot(pc, rotation_matrix.T)

def visualize_results(point_cloud, predictions, config):
    """Visualize point cloud and predictions using Open3D with colored bounding boxes."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Rotation matrices
    rot_90_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    rot_180_x = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])

    # Rotate point cloud
    rotated_pc = rotate_point_cloud(point_cloud[:, :3], rot_90_x)

    # Add rotated point cloud
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(rotated_pc)
    vis.add_geometry(pc_o3d)

    # Predefined colors and names
    predefined_colors = [
        ([1, 0, 0], "Red"),
        ([0, 1, 0], "Green"),
        ([0, 0, 1], "Blue"),
        ([1, 1, 0], "Yellow"),
        ([0.5, 0.5, 0.5], "Gray"),
        ([1, 0, 1], "Magenta"),
        ([0, 1, 1], "Cyan"),
        ([1, 0.5, 0], "Orange"),
        ([0.5, 0, 1], "Purple"),
        ([0, 0.5, 0.5], "Teal")
    ]

    # Map classes to predefined colors dynamically
    class_colors = {}
    for idx, (class_name, _) in enumerate(config.type2class.items()):
        color, name = predefined_colors[idx % len(predefined_colors)]
        class_colors[class_name] = {"color": color, "name": name}

    # Count objects per class
    class_counts = {class_name: 0 for class_name in config.type2class.keys()}

    # Print detected objects
    print("\nDetection Results:")
    for pred in predictions:
        class_idx = pred[0]  # Class index
        bbox_corners = pred[1]  # Bounding box corners
        confidence = pred[2]  # Confidence score

        if confidence < 0.5:  # Optional: Filter by confidence threshold
            continue

        class_name = config.class2type[class_idx]
        color_name = class_colors[class_name]["name"]
        class_counts[class_name] += 1

        print(f"Class: {class_name}, Confidence: {confidence:.2f}, Color: {color_name}")
        print(f"Bounding Box Corners:\n{bbox_corners}")

        # Rotate bounding box corners
        rotated_bbox_corners = rotate_point_cloud(bbox_corners, rot_180_x)

        # Create and color bounding box
        box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(rotated_bbox_corners))
        box.color = class_colors[class_name]["color"]
        vis.add_geometry(box)

    # Print summary of detected objects
    print("\nSummary of Detected Objects:")
    for class_name, count in class_counts.items():
        color_name = class_colors[class_name]["name"]
        print(f"Class: {class_name}, Count: {count}, Color: {color_name}")

    vis.run()
    vis.destroy_window()

def run_inference(weights_dir, pc_file, results_dir, num_points=20000, conf_thresh=0.5):
    """Run inference on a point cloud using the modified VoteNet."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = CustomDatasetConfig()

    # Load model
    model = VoteNet(
        num_class=config.num_class,
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        input_feature_dim=1,
        num_proposal=256,
        vote_factor=1,
        sampling='vote_fps'
    ).to(device)

    checkpoint = torch.load(weights_dir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess point cloud
    point_cloud = preprocess_point_cloud(pc_file, num_points)
    pc_tensor = torch.from_numpy(point_cloud).to(device)

    # Perform inference
    inputs = {'point_clouds': pc_tensor}
    with torch.no_grad():
        end_points = model(inputs)

    # Add 'point_clouds' key if missing
    if 'point_clouds' not in end_points:
        end_points['point_clouds'] = pc_tensor

    # Parse predictions
    eval_config_dict = {
        'remove_empty_box': True,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': False,
        'cls_nms': True,
        'per_class_proposal': True,
        'conf_thresh': conf_thresh,
        'dataset_config': config,
    }
    pred_map_cls = parse_predictions(end_points, eval_config_dict)

    # Rotate and save predictions
    rot_180_x = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])

    rotated_predictions = []
    for pred in pred_map_cls[0]:
        class_idx = pred[0]
        bbox_corners = pred[1]
        confidence = pred[2]
        rotated_bbox_corners = rotate_point_cloud(bbox_corners, rot_180_x)
        rotated_predictions.append((class_idx, rotated_bbox_corners, confidence))

    # Save rotated predictions
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "predictions.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(rotated_predictions, f)
    print(f"Rotated results saved to {results_file}")

    # Visualize predictions
    visualize_results(point_cloud[0], rotated_predictions, config)

# Paths
weights_path = 'Path to Project directory/votenet-main/log_150_epochs/best_checkpoint.tar'
pc_path = 'Path to Project directory/ProcessedDataset/test/point_clouds/2011_09_26_drive_0001_sync_0000000093_shot_3.npz'
result_dir = 'Path to Project directory/votenet-main/Results'

run_inference(weights_path, pc_path, result_dir)
