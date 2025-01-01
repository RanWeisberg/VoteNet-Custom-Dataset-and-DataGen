import os
import numpy as np
import argparse

def rotation_matrix_z(theta):
    """Creates a rotation matrix for a rotation around the z-axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    return R

def compute_box_corners(center, size, heading_angle):
    """
    Computes the 8 corners of the bounding box.
    center: (3,) ndarray of box center coordinates
    size: (3,) ndarray of box dimensions [length, width, height]
    heading_angle: scalar, rotation around z-axis
    Returns:
    corners_3d: (8, 3) ndarray of box corner coordinates
    """
    l, w, h = size
    # Create a bounding box centered at the origin
    x_corners = l / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners)).T  # (8, 3)
    # Rotate and translate the corners
    R = rotation_matrix_z(heading_angle)
    corners = np.dot(corners, R.T)
    corners += center
    return corners

def extract_points_in_box(pc, box_corners):
    """
    Extracts indices of points inside the given bounding box.
    pc: (N, 3) ndarray of point cloud coordinates
    box_corners: (8, 3) ndarray of box corner coordinates
    Returns:
    indices: list of point indices inside the bounding box
    """
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

def generate_ground_truth_votes(pc, bboxes):
    """
    Generates the ground truth votes for the given point cloud and bounding boxes.
    pc: (N, 3) ndarray of point cloud coordinates
    bboxes: (K, 8) ndarray of bounding boxes
    Returns:
    point_votes: (N, 10) ndarray of ground truth votes
    """
    N = pc.shape[0]
    point_votes = np.zeros((N, 10), dtype=np.float32)
    point_vote_idx = np.zeros(N, dtype=int)

    for bbox in bboxes:
        center = bbox[:3]
        size = bbox[3:6]
        heading_angle = bbox[6]
        # Compute box corners
        box_corners = compute_box_corners(center, size, heading_angle)
        # Find points inside the bounding box
        inds = extract_points_in_box(pc[:, :3], box_corners)
        # Update vote mask
        point_votes[inds, 0] = 1
        # Compute votes
        votes = center - pc[inds, :3]
        for i, idx in enumerate(inds):
            vote_count = point_vote_idx[idx]
            if vote_count < 3:
                start_idx = 1 + 3 * vote_count
                point_votes[idx, start_idx:start_idx+3] = votes[i]
                point_vote_idx[idx] += 1
    return point_votes

def main(pc_dir, bb_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pc_files = [f for f in os.listdir(pc_dir) if f.endswith('.npz')]
    pc_files.sort()

    for pc_file in pc_files:
        base_name = os.path.splitext(pc_file)[0]
        pc_path = os.path.join(pc_dir, pc_file)
        bb_file = base_name + '.npy'
        bb_path = os.path.join(bb_dir, bb_file)
        output_file = os.path.join(output_dir, base_name + '.npz')

        print(f'Processing: {base_name}')

        # Load point cloud
        try:
            pc_data = np.load(pc_path)
            if 'pc' in pc_data:
                pc = pc_data['pc']
            else:
                print(f'Error: "pc" key not found in {pc_path}')
                continue
        except Exception as e:
            print(f'Error loading point cloud file {pc_path}: {e}')
            continue

        # Load bounding boxes
        if not os.path.exists(bb_path):
            print(f'Warning: Bounding box file not found: {bb_path}')
            continue
        try:
            bboxes = np.load(bb_path)
            if bboxes.shape[1] != 8:
                print(f'Error: Bounding box file has incorrect shape: {bb_path}')
                continue
        except Exception as e:
            print(f'Error loading bounding box file {bb_path}: {e}')
            continue

        # Generate ground truth votes
        point_votes = generate_ground_truth_votes(pc, bboxes)

        # Save ground truth votes
        np.savez_compressed(output_file, point_votes=point_votes)
        print(f'Saved ground truth votes to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ground truth votes for VoteNet.')
    parser.add_argument('--pc_dir', type=str, required=True, help='Directory of point cloud npz files')
    parser.add_argument('--bb_dir', type=str, required=True, help='Directory of bounding box npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for ground truth votes')
    args = parser.parse_args()

    main(args.pc_dir, args.bb_dir, args.output_dir)
