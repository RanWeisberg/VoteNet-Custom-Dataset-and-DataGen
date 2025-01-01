import numpy as np
import sys

def check_dataset(pc_file, bbox_file, votes_file):
    # Load point cloud data
    try:
        pc_data = np.load(pc_file)
        if 'pc' in pc_data:
            pc = pc_data['pc']  # Shape: (N, 3 + input_feature_dim)
            print(f"Point cloud loaded with shape: {pc.shape}")
        else:
            print("Error: 'pc' key not found in point cloud npz file.")
            return
    except Exception as e:
        print(f"Error loading point cloud file: {e}")
        return

    # Load bounding boxes
    try:
        bboxes = np.load(bbox_file)
        print(f"Bounding boxes loaded with shape: {bboxes.shape}")
    except Exception as e:
        print(f"Error loading bounding box file: {e}")
        return

    # Load ground truth votes
    try:
        votes_data = np.load(votes_file)
        if 'point_votes' in votes_data:
            point_votes = votes_data['point_votes']  # Shape: (N, 10)
            print(f"Ground truth votes loaded with shape: {point_votes.shape}")
        else:
            print("Error: 'point_votes' key not found in votes npz file.")
            return
    except Exception as e:
        print(f"Error loading ground truth votes file: {e}")
        return

    # Check that the number of points matches between PC and votes
    if pc.shape[0] != point_votes.shape[0]:
        print(f"Mismatch in number of points: PC has {pc.shape[0]}, votes have {point_votes.shape[0]}")
        return
    else:
        print(f"Number of points match: {pc.shape[0]}")

    # Check dimensions
    if pc.shape[1] != 3:
        print(f"Unexpected point cloud dimension: expected (N, 3), got {pc.shape}")
    else:
        print(f"Point cloud dimension is correct: {pc.shape}")

    if bboxes.shape[1] != 8:
        print(f"Unexpected bounding box dimension: expected (K, 8), got {bboxes.shape}")
    else:
        print(f"Bounding box dimension is correct: {bboxes.shape}")

    if point_votes.shape[1] != 10:
        print(f"Unexpected ground truth votes dimension: expected (N, 10), got {point_votes.shape}")
    else:
        print(f"Ground truth votes dimension is correct: {point_votes.shape}")

    # Verify that vote masks are 0 or 1
    unique_vote_masks = np.unique(point_votes[:, 0])
    if not np.all(np.isin(unique_vote_masks, [0, 1])):
        print(f"Invalid vote masks found: {unique_vote_masks}")
    else:
        print(f"Vote masks are valid: {unique_vote_masks}")

    # Verify semantic class labels are integers starting from 0
    if not np.issubdtype(bboxes[:, 7].dtype, np.integer):
        print("Converting semantic class labels to integers.")
        bboxes[:, 7] = bboxes[:, 7].astype(np.int32)
    min_class = np.min(bboxes[:, 7])
    if min_class < 0:
        print(f"Semantic class labels should start from 0, but minimum is {min_class}")
        return
    else:
        print(f"Semantic class labels are valid, starting from {min_class}")

    # Optionally, verify vote offsets correspond to bounding box centers
    # For a sample of points, check if the votes point to object centers
    print("\nVerifying ground truth votes for a few sample points...")

    num_samples = 5  # Number of sample points to verify
    vote_mask = point_votes[:, 0] == 1
    indices = np.where(vote_mask)[0]

    if len(indices) == 0:
        print("No points with vote mask == 1 found.")
        return

    sample_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)

    for idx in sample_indices:
        point = pc[idx, :3]
        votes = point_votes[idx, 1:4]  # Using the first vote offset
        vote_xyz = point + votes  # Predicted object center
        # Find the closest bounding box center to the vote
        centers = bboxes[:, :3]
        distances = np.linalg.norm(centers - vote_xyz, axis=1)
        closest_bbox_idx = np.argmin(distances)
        distance = distances[closest_bbox_idx]
        print(f"Point index {idx}:")
        print(f"  Point coordinates: {point}")
        print(f"  Vote offset: {votes}")
        print(f"  Voted center: {vote_xyz}")
        print(f"  Closest bounding box center: {centers[closest_bbox_idx]}")
        print(f"  Distance to closest bbox center: {distance:.4f}")
        if distance < 0.1:
            print("  Vote aligns well with a bounding box center.")
        else:
            print("  Vote does not align closely with any bounding box center.")

    print("\nDataset check completed.")

if __name__ == "__main__":
    # Example usage:
    # python check_dataset.py pc_file.npz bbox_file.npy votes_file.npz
    if len(sys.argv) != 4:
        print("Usage: python check_dataset.py <pc_file.npz> <bbox_file.npy> <votes_file.npz>")
    else:
        pc_file = sys.argv[1]
        bbox_file = sys.argv[2]
        votes_file = sys.argv[3]
        check_dataset(pc_file, bbox_file, votes_file)
