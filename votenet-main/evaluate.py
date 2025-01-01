import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from models.votenet import VoteNet
from data_config import CustomDatasetConfig
from detection_dataset import CustomDetectionDataset
from loss_helper import get_loss

# Configuration
DATASET_CONFIG = CustomDatasetConfig()
BATCH_SIZE = 8
NUM_POINT = 20000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = 'Path to Project directory/votenet-main/log_150_epochs/best_checkpoint.tar'
DATASET_DIR = 'Path to Project directory/ProcessedDataset'
OUTPUT_DIR = 'Path to Project directory/votenet-main/EvaluationResults'
AP_IOU_THRESH = 0.5

# Dataset Loading
TEST_DATASET = CustomDetectionDataset(
    split_set='test',
    dataset_dir=DATASET_DIR,
    num_points=NUM_POINT,
    augment=False,
    use_height=True
)

TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# Model Initialization
model = VoteNet(
    num_class=DATASET_CONFIG.num_class,
    num_heading_bin=DATASET_CONFIG.num_heading_bin,
    num_size_cluster=DATASET_CONFIG.num_size_cluster,
    mean_size_arr=DATASET_CONFIG.mean_size_arr,
    input_feature_dim=1,
    num_proposal=256,
    vote_factor=1,
    sampling='vote_fps'
).to(DEVICE)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluation Configuration
EVAL_CONFIG_DICT = {
    'remove_empty_box': False,
    'use_3d_nms': True,
    'nms_iou': 0.25,
    'use_old_type_nms': False,
    'cls_nms': True,
    'per_class_proposal': True,
    'conf_thresh': 0.05,
    'dataset_config': DATASET_CONFIG,
}

# Output Directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate():
    print("Starting evaluation...")
    ap_calculator = APCalculator(ap_iou_thresh=AP_IOU_THRESH, class2type_map=DATASET_CONFIG.class2type)
    total_loss = 0.0
    num_batches = 0

    # Confusion matrix variables
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0

    # Additional metrics
    pos_ratio_sum = 0.0
    neg_ratio_sum = 0.0

    # Loss metrics accumulators
    center_loss_sum = 0.0
    objectness_loss_sum = 0.0
    heading_cls_loss_sum = 0.0
    heading_reg_loss_sum = 0.0
    size_cls_loss_sum = 0.0
    size_reg_loss_sum = 0.0
    sem_cls_loss_sum = 0.0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(TEST_DATALOADER):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(TEST_DATALOADER)}")

            # Move data to device
            for key in batch_data:
                batch_data[key] = batch_data[key].to(DEVICE)

            # Forward pass
            inputs = {'point_clouds': batch_data['point_clouds']}
            end_points = model(inputs)

            # Compute loss
            for key in batch_data:
                end_points[key] = batch_data[key]
            loss, end_points = get_loss(end_points, DATASET_CONFIG)
            total_loss += loss.item()
            num_batches += 1

            # Accumulate individual loss components
            center_loss_sum += end_points['center_loss'].item()
            objectness_loss_sum += end_points['objectness_loss'].item()
            heading_cls_loss_sum += end_points['heading_cls_loss'].item()
            heading_reg_loss_sum += end_points['heading_reg_loss'].item()
            size_cls_loss_sum += end_points['size_cls_loss'].item()
            size_reg_loss_sum += end_points['size_reg_loss'].item()
            sem_cls_loss_sum += end_points['sem_cls_loss'].item()

            # Update confusion matrix
            confusion = end_points.get('confusion_matrix', {})
            total_TP += confusion.get('TP', 0)
            total_FP += confusion.get('FP', 0)
            total_TN += confusion.get('TN', 0)
            total_FN += confusion.get('FN', 0)

            # Update ratios
            pos_ratio_sum += end_points['pos_ratio'].item()
            neg_ratio_sum += end_points['neg_ratio'].item()

            # Parse predictions and ground truth
            batch_pred_map_cls = parse_predictions(end_points, EVAL_CONFIG_DICT)
            batch_gt_map_cls = parse_groundtruths(end_points, EVAL_CONFIG_DICT)

            # Calculate Average Precision (AP)
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # Compute final metrics
    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_pos_ratio = pos_ratio_sum / num_batches if num_batches > 0 else 0.0
    avg_neg_ratio = neg_ratio_sum / num_batches if num_batches > 0 else 0.0

    # Compute average loss components
    avg_center_loss = center_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_objectness_loss = objectness_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_heading_cls_loss = heading_cls_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_heading_reg_loss = heading_reg_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_size_cls_loss = size_cls_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_size_reg_loss = size_reg_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_sem_cls_loss = sem_cls_loss_sum / num_batches if num_batches > 0 else 0.0

    # Precision, Recall
    precision = total_TP / (total_TP + total_FP + 1e-6)
    recall = total_TP / (total_TP + total_FN + 1e-6)

    # Log metrics
    metrics = ap_calculator.compute_metrics()
    results_file = os.path.join(OUTPUT_DIR, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Mean Loss: {mean_loss}\n")
        f.write(f"Center Loss: {avg_center_loss}\n")
        f.write(f"Objectness Loss: {avg_objectness_loss}\n")
        f.write(f"Heading Classification Loss: {avg_heading_cls_loss}\n")
        f.write(f"Heading Regression Loss: {avg_heading_reg_loss}\n")
        f.write(f"Size Classification Loss: {avg_size_cls_loss}\n")
        f.write(f"Size Regression Loss: {avg_size_reg_loss}\n")
        f.write(f"Semantic Classification Loss: {avg_sem_cls_loss}\n")
        f.write(f"Average Positive Ratio: {avg_pos_ratio}\n")
        f.write(f"Average Negative Ratio: {avg_neg_ratio}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"TP: {total_TP}, FP: {total_FP}, TN: {total_TN}, FN: {total_FN}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # Print metrics
    print(f"Mean Loss: {mean_loss}")
    print(f"Center Loss: {avg_center_loss}")
    print(f"Objectness Loss: {avg_objectness_loss}")
    print(f"Heading Classification Loss: {avg_heading_cls_loss}")
    print(f"Heading Regression Loss: {avg_heading_reg_loss}")
    print(f"Size Classification Loss: {avg_size_cls_loss}")
    print(f"Size Regression Loss: {avg_size_reg_loss}")
    print(f"Semantic Classification Loss: {avg_sem_cls_loss}")
    print(f"Average Positive Ratio: {avg_pos_ratio}")
    print(f"Average Negative Ratio: {avg_neg_ratio}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix: TP={total_TP}, FP={total_FP}, TN={total_TN}, FN={total_FN}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    evaluate()
