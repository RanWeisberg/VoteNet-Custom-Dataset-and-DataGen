# check_point_clouds.py

import os
import sys
import numpy as np

def check_point_clouds(directory, expected_dim=None):
    mismatched_files = []
    first_file = True

    for fname in os.listdir(directory):
        file_path = os.path.join(directory, fname)
        if os.path.isfile(file_path):
            # Load the point cloud file
            try:
                point_cloud = np.load(file_path)
                if first_file and expected_dim is None:
                    expected_dim = point_cloud.shape[0]
                    first_file = False
                elif point_cloud.shape[0] != expected_dim:
                    mismatched_files.append((fname, point_cloud.shape[0]))
            except Exception as e:
                print(f"Error loading file {fname}: {e}")
                mismatched_files.append((fname, 'Error loading file'))

    if mismatched_files:
        print("The following files have mismatched dimensions:")
        for fname, dim in mismatched_files:
            print(f"File: {fname}, Dimension: {dim}")
    else:
        print("All point cloud files have the same dimension.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check point cloud files for consistent dimensions.')
    parser.add_argument('directory', type=str, help='Directory containing point cloud files')
    args = parser.parse_args()

    check_point_clouds(args.directory)
