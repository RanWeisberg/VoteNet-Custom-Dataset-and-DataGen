# votenet.py

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep Hough Voting Network for 3D Object Detection in Point Clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss


class VoteNet(nn.Module):
    """
    A deep neural network for 3D object detection with end-to-end optimizable Hough voting.

    Parameters
    ----------
    num_class : int
        Number of semantic classes to predict over.
    num_heading_bin : int
        Number of heading bins.
    num_size_cluster : int
        Number of size clusters.
    mean_size_arr : np.ndarray
        Mean size array for size clusters.
    input_feature_dim : int, optional (default=0)
        Input feature dimension. If the point cloud has additional features beyond XYZ, set this accordingly.
    num_proposal : int, optional (default=128)
        Number of proposals/detections generated from the network.
    vote_factor : int, optional (default=1)
        Number of votes generated from each seed point.
    sampling : str, optional (default='vote_fps')
        Sampling strategy ('vote_fps', 'seed_fps', 'random').
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super(VoteNet, self).__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                   mean_size_arr, num_proposal, sampling)

    def forward(self, inputs):
        """
        Forward pass of the VoteNet.

        Args:
            inputs : dict
                {
                    'point_clouds' : torch.FloatTensor (B, N, 3 + C)
                }

        Returns:
            end_points : dict
                Contains all intermediate and final predictions.
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        # Ensure point_clouds are float
        point_clouds = inputs['point_clouds'].float()

        # Forward through backbone
        end_points = self.backbone_net(point_clouds, end_points)

        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        # Generate votes
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        # Forward through Proposal Module
        end_points = self.pnet(xyz, features, end_points)

        return end_points


if __name__ == '__main__':
    # Testing the VoteNet architecture
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model with example parameters
    model = VoteNet(num_class=10,
                    num_heading_bin=12,
                    num_size_cluster=10,
                    mean_size_arr=np.random.random((10, 3)),
                    input_feature_dim=0,
                    num_proposal=128,
                    vote_factor=1,
                    sampling='seed_fps').cuda()

    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass with a sample
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda().float()}
    except Exception as e:
        print(f'Dataset has not been prepared. Using a random sample. Error: {e}')
        inputs = {'point_clouds': torch.rand((1, 20000, 3)).cuda().float()}

    end_points = model(inputs)
    for key in end_points:
        print(f'{key}: {end_points[key].shape}')

    try:
        # Compute loss with the sample
        for key in sample:
            if isinstance(sample[key], np.ndarray):
                end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda().float()
            else:
                end_points[key] = sample[key].cuda().float()
        loss, end_points = get_loss(end_points, DC)
        print(f'Loss: {loss.item()}')
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = torch.ones((1, 128)).cuda().float()
        dump_results(end_points, 'tmp', DC)
    except Exception as e:
        print(f'Dataset has not been prepared. Skipping loss computation and dump. Error: {e}')
