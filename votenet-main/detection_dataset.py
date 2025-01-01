# detection_dataset.py

import os
import numpy as np
from torch.utils.data import Dataset
from data_config import CustomDatasetConfig
import utils.pc_util as pc_util
import torch

DC = CustomDatasetConfig()  # Initialize the dataset configuration

class CustomDetectionDataset(Dataset):
    def __init__(self, split_set='train', dataset_dir='Path to Project directory/ProcessedDataset',
                 num_points=20000, augment=False, use_height=False, max_num_obj=10):
        assert split_set in ['train', 'val', 'test'], "split_set must be one of ['train', 'val', 'test']"
        assert dataset_dir is not None, "dataset_dir must be specified"

        self.split_set = split_set
        self.dataset_dir = dataset_dir
        self.num_points = num_points
        self.augment = augment
        self.use_height = use_height
        self.max_num_obj = max_num_obj  # Maximum number of objects per scene

        # Load file paths for point clouds and labels
        self.data_dir = os.path.join(self.dataset_dir, self.split_set, 'point_clouds')
        self.label_dir = os.path.join(self.dataset_dir, self.split_set, 'bounding_boxes')
        self.gt_votes_dir = os.path.join(self.dataset_dir, self.split_set, 'ground_truth_votes')

        self.scan_names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.data_dir) if f.endswith('.npz')])

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]

        # Load point cloud
        point_cloud = np.load(os.path.join(self.data_dir, f'{scan_name}.npz'))['pc']
        if point_cloud.shape[0] != self.num_points:
            point_cloud, _ = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)

        # Compute height and append it as a feature
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            height = np.expand_dims(height, 1)
            point_cloud = np.concatenate([point_cloud, height], axis=1)

        # Load labels (bounding boxes and classes)
        bboxes = np.load(os.path.join(self.label_dir, f'{scan_name}.npy'))

        # Load ground truth votes (if available)
        gt_votes = np.load(os.path.join(self.gt_votes_dir, f'{scan_name}.npz'))['point_votes']
        vote_label_mask = gt_votes[:, 0]

        # Get the vote offsets
        vote = gt_votes[:, 1:4]  # Shape: [num_points, 3]
        vote_label = np.tile(vote, (1, 3))  # Shape: [num_points, 9]

        # Data augmentation
        if self.augment:
            point_cloud, bboxes, vote_label, vote_label_mask = self._augment_data(
                point_cloud, bboxes, vote_label, vote_label_mask)

        # Prepare the data
        num_boxes = bboxes.shape[0]
        num_boxes = min(num_boxes, self.max_num_obj)  # Truncate if more than max_num_obj
        box3d_centers = bboxes[:, 0:3]
        heading_angles = bboxes[:, 6]
        box_sizes = bboxes[:, 3:6]
        sem_cls_labels = bboxes[:, 7].astype(np.int64)

        # Initialize tensors with padding
        center_label = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        heading_class_labels = np.zeros((self.max_num_obj,), dtype=np.int64)
        heading_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        size_class_labels = np.zeros((self.max_num_obj,), dtype=np.int64)
        size_residuals = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        sem_cls_label = np.zeros((self.max_num_obj,), dtype=np.int64)
        box_label_mask = np.zeros((self.max_num_obj,), dtype=np.float32)

        # Fill in the data
        for i in range(num_boxes):
            # Center
            center_label[i, :] = box3d_centers[i, :]

            # Heading
            heading_angle = heading_angles[i]
            heading_class, heading_residual = DC.angle2class(heading_angle)
            heading_class_labels[i] = heading_class
            heading_residuals[i] = heading_residual

            # Size
            box_size = box_sizes[i]
            size_class, size_residual = DC.size2class(box_size, DC.class2type[sem_cls_labels[i]])
            size_class_labels[i] = size_class
            size_residuals[i, :] = size_residual

            # Semantic class
            sem_cls_label[i] = sem_cls_labels[i]

            # Box label mask
            box_label_mask[i] = 1.0

        # Truncate if necessary
        vote_label_mask = vote_label_mask[:self.num_points]
        vote_label = vote_label[:self.num_points, :]
        point_cloud = point_cloud[:self.num_points, :]

        return {
            'point_clouds': torch.from_numpy(point_cloud.astype(np.float32)),
            'center_label': torch.from_numpy(center_label),
            'heading_class_label': torch.from_numpy(heading_class_labels),
            'heading_residual_label': torch.from_numpy(heading_residuals),
            'size_class_label': torch.from_numpy(size_class_labels),
            'size_residual_label': torch.from_numpy(size_residuals),
            'sem_cls_label': torch.from_numpy(sem_cls_label),
            'box_label_mask': torch.from_numpy(box_label_mask),
            'vote_label': torch.from_numpy(vote_label.astype(np.float32)),
            'vote_label_mask': torch.from_numpy(vote_label_mask.astype(np.int64)),
            'scan_idx': torch.tensor(idx, dtype=torch.int64)
        }

    def _augment_data(self, point_cloud, bboxes, vote_label, vote_label_mask):
        """ Apply data augmentation. """

        # Flipping along YZ plane
        if np.random.random() > 0.5:
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            bboxes[:, 0] = -1 * bboxes[:, 0]
            bboxes[:, 6] = np.pi - bboxes[:, 6]
            vote_label[:, 0::3] = -1 * vote_label[:, 0::3]  # Flip x coordinates in votes
            # vote_label_mask remains unchanged

        # Rotation around Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degrees
        rot_mat = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle), 0],
            [np.sin(rot_angle),  np.cos(rot_angle), 0],
            [0,                 0,                1]
        ])

        # Rotate point cloud
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], rot_mat.T)

        # Rotate bounding boxes
        bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], rot_mat.T)
        bboxes[:, 6] -= rot_angle

        # Rotate vote labels
        vote_label_reshaped = vote_label.reshape(-1, 3, 3)  # Shape: [num_points, 3 votes, 3 coords]
        for i in range(3):  # Apply rotation to each vote
            vote_label_end = point_cloud[:, 0:3] + vote_label_reshaped[:, i, :]
            vote_label_end = np.dot(vote_label_end, rot_mat.T)
            vote_label_reshaped[:, i, :] = vote_label_end - point_cloud[:, 0:3]
        vote_label = vote_label_reshaped.reshape(-1, 9)  # Back to shape [num_points, 9]

        return point_cloud, bboxes, vote_label, vote_label_mask

if __name__ == "__main__":
    dataset = CustomDetectionDataset()
    data = dataset[0]
    print(data.keys())
    for key in data:
        print(f"{key}: {data[key].shape}")
