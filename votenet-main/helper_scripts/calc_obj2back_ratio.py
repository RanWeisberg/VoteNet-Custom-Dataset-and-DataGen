import os
import sys
import numpy as np
import glob


def compute_average_ratio(directory, point_votes_key='point_votes'):
    """
    Computes the average ratio of object samples to background samples across all .npz files in the directory.

    Args:
        directory (str): Path to the directory containing .npz files.
        point_votes_key (str): Key name for the point_votes data in the .npz files.

    Returns:
        float: Average object-to-background ratio.
    """
    # Verify that the directory exists
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        sys.exit(1)

    # Find all .npz files in the directory
    npz_files = glob.glob(os.path.join(directory, '*.npz'))

    if not npz_files:
        print(f"No .npz files found in the directory '{directory}'.")
        sys.exit(1)

    ratios = []
    file_ratios = {}

    for file_path in npz_files:
        try:
            # Load the .npz file
            data = np.load(file_path)

            if point_votes_key not in data:
                print(f"Warning: '{point_votes_key}' key not found in '{file_path}'. Skipping this file.")
                continue

            # Extract the point_votes array
            point_votes = data[point_votes_key]

            if not isinstance(point_votes, np.ndarray):
                print(f"Warning: '{point_votes_key}' in '{file_path}' is not a NumPy array. Skipping this file.")
                continue

            # Check the shape of point_votes
            if point_votes.ndim != 2 or point_votes.shape[1] < 1:
                print(
                    f"Warning: '{point_votes_key}' in '{file_path}' has an unexpected shape {point_votes.shape}. Skipping this file.")
                continue

            # Extract the mask from the first column
            mask = point_votes[:, 0]

            # Ensure mask is binary (0 or 1)
            unique_values = np.unique(mask)
            if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values,
                                                                                [0]) and not np.array_equal(
                    unique_values, [1]):
                print(
                    f"Warning: '{point_votes_key}' in '{file_path}' contains non-binary values {unique_values}. Skipping this file.")
                continue

            # Count object and background samples
            num_objects = np.sum(mask == 1)
            num_background = np.sum(mask == 0)
            total = num_objects + num_background

            if total == 0:
                print(f"Warning: '{point_votes_key}' in '{file_path}' contains no samples. Skipping this file.")
                continue

            # Calculate ratio
            ratio = num_objects / total
            ratios.append(ratio)
            file_ratios[os.path.basename(file_path)] = ratio

            print(f"File: {os.path.basename(file_path)} | Object Ratio: {ratio:.4f} ({num_objects}/{total})")

        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            continue

    if not ratios:
        print("No valid files were processed. Cannot compute average ratio.")
        sys.exit(1)

    # Compute average ratio
    average_ratio = np.mean(ratios)
    print("\n--- Summary ---")
    print(f"Processed {len(ratios)} files.")
    print(f"Average Object-to-Background Ratio: {average_ratio:.4f}")
    return average_ratio


if __name__ == "__main__":
    import argparse

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Compute average object-to-background ratio from .npz files.")
    parser.add_argument('directory', type=str, help="Path to the directory containing .npz files.")
    parser.add_argument('--point_votes_key', type=str, default='point_votes',
                        help="Key name for the point_votes data in the .npz files (default: 'point_votes').")

    args = parser.parse_args()

    # Compute and display the average ratio
    compute_average_ratio(args.directory, point_votes_key=args.point_votes_key)
