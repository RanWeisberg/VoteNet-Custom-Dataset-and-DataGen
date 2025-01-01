import numpy as np


def get_npz_dimensions(file_path):
    try:
        # Load the .npz file
        data = np.load(file_path)

        # Retrieve and print dimensions of each array
        dimensions = {key: arr.shape for key, arr in data.items()}

        print("Dimensions of arrays in the .npz file:")
        for key, shape in dimensions.items():
            print(f"{key}: {shape}")

        return dimensions
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage
file_path = "/FinalProject/ProcessedDataset/train/ground_truth_votes/2011_09_26_drive_0001_sync_0000000001_shot_1.npz"  # Replace with your .npz file path
get_npz_dimensions(file_path)
