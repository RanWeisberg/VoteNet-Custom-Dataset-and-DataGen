import numpy as np


def npy_to_txt(npy_file_path, txt_file_path):
    try:
        # Load the .npy file
        data = np.load(npy_file_path)

        # Save the data to a .txt file
        np.savetxt(txt_file_path, data, fmt='%s')  # Adjust `fmt` as needed for your data

        print(f"Successfully saved data from {npy_file_path} to {txt_file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
npy_file_path = "/FinalProject/ProcessedDataset/train/bounding_boxes/2011_09_26_drive_0001_sync_0000000001_shot_4.npy"  # Replace with your .npy file path
txt_file_path = "/FinalProject/votenet-main/output.txt"  # Replace with your desired output .txt file path
npy_to_txt(npy_file_path, txt_file_path)
