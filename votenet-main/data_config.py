import numpy as np
import yaml



class CustomDatasetConfig(object):
    def __init__(self):
        self.config_file_path = 'Path to Project directory/ProcessedDataset/data_config.yaml'
        self.read_config()

    def read_config(self):
        with open(self.config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set basic attributes
        self.num_class = config['num_class']
        self.num_heading_bin = config['num_heading_bin']
        self.num_size_cluster = config['num_size_cluster']

        # Set type2class
        self.type2class = config['type2class']

        # Generate inverse mapping
        self.class2type = {v: k for k, v in self.type2class.items()}

        # Generate one-hot encoding
        self.type2onehotclass = {t: i for i, t in enumerate(self.type2class)}

        # Set type_mean_size
        self.type_mean_size = {k: np.array(v) for k, v in config['type_mean_size'].items()}

        # Precompute the mean size array
        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]

    # The rest of the methods remain the same
    def size2class(self, size, type_name):
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb


# Usage example
if __name__ == "__main__":
    config = CustomDatasetConfig()

    print(f"Number of classes: {config.num_class}")
    print(f"Number of heading bins: {config.num_heading_bin}")
    print(f"Number of size clusters: {config.num_size_cluster}")
    print(f"Type to class mapping: {config.type2class}")
    print(f"Type mean sizes: {config.type_mean_size}")