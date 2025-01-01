import os
import random
import yaml
import numpy as np
from plyfile import PlyData
from place import place_objects_in_scene
from split import split_point_cloud_with_vertical_crop


def collect_class_info(objects_dir):
    pc_dir = os.path.join(objects_dir, 'PC')
    if not os.path.exists(pc_dir):
        raise FileNotFoundError(f"PC directory not found: {pc_dir}")
    class_names = [os.path.splitext(f)[0] for f in os.listdir(pc_dir) if f.endswith('.ply')]
    class_info = {name: calculate_mean_size(os.path.join(pc_dir, f"{name}.ply")) for name in class_names}
    return class_info


def calculate_mean_size(ply_file):
    try:
        plydata = PlyData.read(ply_file)
        vertices = plydata['vertex']
        points = np.array([[v[0], v[1], v[2]] for v in vertices])
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        dimensions = np.sort(dimensions)[::-1]
        return dimensions
    except Exception as e:
        print(f"Error processing {ply_file}: {str(e)}")
        return np.array([1.0, 1.0, 1.0])


def create_data_yaml(output_dir, class_info):
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    data_yaml = {
        'train': 'train/point_clouds',
        'val': 'val/point_clouds',
        'test': 'test/point_clouds',
        'nc': len(class_info),
        'names': list(class_info.keys())
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    print(f"data.yaml created at {data_yaml_path}")


def create_data_config_yaml(output_dir, class_info):
    config_yaml_path = os.path.join(output_dir, 'data_config.yaml')
    config_data = {
        'num_class': len(class_info),
        'num_heading_bin': 12,
        'num_size_cluster': len(class_info),
        'type2class': {name: idx for idx, name in enumerate(class_info.keys())},
        'type_mean_size': {name: [float(f"{dim:.2f}") for dim in mean_size] for name, mean_size in class_info.items()}
    }
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    print(f"data_config.yaml created at {config_yaml_path}")


def select_and_remove_random_scene(scenes_txt_path):
    with open(scenes_txt_path, 'r') as f:
        scenes = f.readlines()

    if not scenes:
        return None

    selected_index = random.randint(0, len(scenes) - 1)
    selected_scene = scenes[selected_index].strip()

    # Remove the selected scene from the list and overwrite the file
    with open(scenes_txt_path, 'w') as f:
        for i, scene in enumerate(scenes):
            if i != selected_index:
                f.write(scene)

    return selected_scene


def create_dataset(kitti_dir, objects_dir, output_dir, num_scenes=None, split_ratio=(0.7, 0.2, 0.1)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all scenes into a text file if not already done
    scenes_txt_path = os.path.join(output_dir, 'scenes.txt')
    if not os.path.exists(scenes_txt_path):
        with open(scenes_txt_path, 'w') as f:
            for root, dirs, files in os.walk(kitti_dir):
                for file in files:
                    if file.endswith('.bin'):
                        f.write(os.path.join(root, file) + '\n')

    # If num_scenes is None, process all scenes
    if num_scenes is None:
        with open(scenes_txt_path, 'r') as f:
            num_scenes = len(f.readlines()) * 5  # Each scene splits into 5 segments

    # Define directories for splits
    pc_dirs = {split: os.path.join(output_dir, split, 'point_clouds') for split in ['train', 'val', 'test']}
    label_dirs = {split: os.path.join(output_dir, split, 'bounding_boxes') for split in ['train', 'val', 'test']}
    gt_vote_dirs = {split: os.path.join(output_dir, split, 'ground_truth_votes') for split in ['train', 'val', 'test']}

    for d in pc_dirs.values():
        os.makedirs(d, exist_ok=True)
    for d in label_dirs.values():
        os.makedirs(d, exist_ok=True)
    for d in gt_vote_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Initialize counters and total segment count
    split_counts = {"train": 0, "val": 0, "test": 0}
    total_segments = 0

    while total_segments < num_scenes:
        selected_scene = select_and_remove_random_scene(scenes_txt_path)
        if not selected_scene:
            break  # No more scenes to process

        split_output_dir = os.path.join(output_dir, "splits", "temp")
        os.makedirs(split_output_dir, exist_ok=True)
        split_point_cloud_with_vertical_crop(selected_scene, split_output_dir)
        split_files = [os.path.join(split_output_dir, f) for f in os.listdir(split_output_dir) if f.endswith('.bin')]

        # Process each split file
        for split_file in split_files:
            total_segments += 1
            if split_counts["train"] < int(num_scenes * split_ratio[0]):
                split_type = 'train'
            elif split_counts["val"] < int(num_scenes * split_ratio[1]):
                split_type = 'val'
            else:
                split_type = 'test'

            split_counts[split_type] += 1

            unique_file_name = os.path.splitext(os.path.basename(split_file))[0]
            output_scene_name = f"{unique_file_name}.npz"
            output_label_name = f"{unique_file_name}.npy"
            output_gt_vote_name = f"{unique_file_name}.npz"

            output_scene_path = os.path.join(pc_dirs[split_type], output_scene_name)
            output_label_path = os.path.join(label_dirs[split_type], output_label_name)
            output_gt_vote_path = os.path.join(gt_vote_dirs[split_type], output_gt_vote_name)

            place_objects_in_scene(split_file, objects_dir, output_scene_path, output_label_path, output_gt_vote_path)

        # Clean up
        for file in split_files:
            os.remove(file)
        os.rmdir(split_output_dir)

    print(f"Dataset created with {split_counts['train']} training, {split_counts['val']} validation, and {split_counts['test']} test samples.")

    # Clean up the scenes.txt file if all scenes were processed
    if os.path.exists(scenes_txt_path):
        with open(scenes_txt_path, 'r') as f:
            remaining_scenes = f.readlines()
        if not remaining_scenes:
            os.remove(scenes_txt_path)

    # Remove the temporary splits directory if it still exists
    temp_splits_dir = os.path.join(output_dir, "splits")
    if os.path.exists(temp_splits_dir) and not os.listdir(temp_splits_dir):
        os.rmdir(temp_splits_dir)

    class_info = collect_class_info(objects_dir)
    create_data_yaml(output_dir, class_info)
    create_data_config_yaml(output_dir, class_info)


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    kitti_dir = os.path.join(base_dir, "../Kitti Files")
    objects_dir = os.path.join(base_dir, "../models")
    output_dir = os.path.join(base_dir, "../ProcessedDataset")

    num_scenes = 10  # Set to None to process all scenes
    split_ratio = (0.80, 0.10, 0.10)

    create_dataset(kitti_dir, objects_dir, output_dir, num_scenes, split_ratio)
