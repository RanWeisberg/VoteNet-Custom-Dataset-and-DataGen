# train.py

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import time
import csv
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Suppress specific warnings if desired
warnings.filterwarnings("ignore", category=UserWarning, module='pyglet')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='custom', help='Dataset name. [default: custom]')
parser.add_argument('--dataset_dir', default='Path to Project directory/ProcessedDataset', help='Dataset directory')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log_150_epochs', help='Dump dir to save model checkpoint [default: log_custom]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 150]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='70,100,140', help='When to decay the learning rate (in epochs)')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else os.path.join(LOG_DIR, 'dump')
if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# Import dataset and model
from detection_dataset import CustomDetectionDataset
from data_config import CustomDatasetConfig

DATASET_CONFIG = CustomDatasetConfig()

# Initialize dataset and dataloader
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

TRAIN_DATASET = CustomDetectionDataset(
    split_set='train',
    dataset_dir=FLAGS.dataset_dir,
    num_points=NUM_POINT,
    augment=True,
    use_height=(not FLAGS.no_height)
)

TEST_DATASET = CustomDetectionDataset(
    split_set='val',
    dataset_dir=FLAGS.dataset_dir,
    num_points=NUM_POINT,
    augment=False,
    use_height=(not FLAGS.no_height)
)

print("Number of training samples:", len(TRAIN_DATASET))
print("Number of validation samples:", len(TEST_DATASET))

if len(TRAIN_DATASET) == 0 or len(TEST_DATASET) == 0:
    print("Error: The dataset is empty. Please check the dataset directory and ensure that the data is correctly prepared.")
    sys.exit(1)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, worker_init_fn=worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, worker_init_fn=worker_init_fn)

print("Number of training batches:", len(TRAIN_DATALOADER))
print("Number of validation batches:", len(TEST_DATALOADER))

MODEL = importlib.import_module(FLAGS.model)  # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(not FLAGS.no_height) * 1  # Adjust according to your input features

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
else:
    Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

if torch.cuda.device_count() > 1:
    log_string(f"Let's use {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)

net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Decay BatchNorm momentum from 0.5 to 0.001
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')

# Used for AP calculation
CONFIG_DICT = {
    'remove_empty_box': False,
    'use_3d_nms': True,
    'nms_iou': 0.25,
    'use_old_type_nms': False,
    'cls_nms': True,
    'per_class_proposal': True,
    'conf_thresh': 0.05,
    'dataset_config': DATASET_CONFIG
}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def log_training_metrics(epoch, metrics):
    csv_file = os.path.join(LOG_DIR, 'train_log.csv')
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'epoch',
            'loss',
            'vote_loss',
            'objectness_loss',
            'box_loss',
            'sem_cls_loss',
            'obj_acc',
            'pos_ratio',
            'neg_ratio',
            'TP',
            'FP',
            'TN',
            'FN',
            'epoch_time_sec',
            'estimated_finish_time_sec'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def log_evaluation_metrics(epoch, metrics):
    csv_file = os.path.join(LOG_DIR, 'eval_log.csv')
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'epoch',
            'loss',
            'center_loss',
            'heading_cls_loss',
            'heading_reg_loss',
            'size_cls_loss',
            'size_reg_loss',
            'sem_cls_loss',
            'pos_ratio',
            'neg_ratio',
            'TP',
            'FP',
            'TN',
            'FN',
            'eval_time_sec',
            'estimated_finish_time_sec'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def train_one_epoch(epoch, total_start_time, total_epochs):
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, epoch)
    bnm_scheduler.step()  # decay BN momentum
    net.train()  # set model to training mode

    epoch_start_time = time.time()

    # Initialize confusion matrix counts
    epoch_TP = 0
    epoch_FP = 0
    epoch_TN = 0
    epoch_FN = 0

    # Initialize loss accumulators
    loss_sum = 0.0
    vote_loss_sum = 0.0
    objectness_loss_sum = 0.0
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    obj_acc_sum = 0.0
    pos_ratio_sum = 0.0
    neg_ratio_sum = 0.0
    num_batches = 0

    print(f"Starting training epoch {epoch+1}")

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        num_batches += 1

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            log_string(f'Epoch: {epoch+1}/{MAX_EPOCH}, Batch: {batch_idx+1}/{len(TRAIN_DATALOADER)}')

        try:
            # Move data to device
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            optimizer.zero_grad()
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            end_points = net(inputs)

            # Compute loss and gradients, update parameters.
            for key in batch_data_label:
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, DATASET_CONFIG)
            loss.backward()
            optimizer.step()

            # Accumulate statistics
            loss_sum += loss.item()
            vote_loss_sum += end_points['vote_loss'].item()
            objectness_loss_sum += end_points['objectness_loss'].item()
            box_loss_sum += end_points['box_loss'].item()
            sem_cls_loss_sum += end_points['sem_cls_loss'].item()
            obj_acc_sum += end_points['obj_acc'].item()
            pos_ratio_sum += end_points['pos_ratio'].item()
            neg_ratio_sum += end_points['neg_ratio'].item()

            # Accumulate confusion matrix
            confusion = end_points.get('confusion_matrix', {})
            epoch_TP += confusion.get('TP', 0)
            epoch_FP += confusion.get('FP', 0)
            epoch_TN += confusion.get('TN', 0)
            epoch_FN += confusion.get('FN', 0)

        except Exception as e:
            print(f"Error during batch processing at batch {batch_idx}: {e}")
            continue  # Skip to the next batch

    if num_batches == 0:
        print("No batches were processed during this epoch.")
        return  # Skip the rest of the epoch

    # Compute average metrics
    avg_loss = loss_sum / num_batches
    avg_vote_loss = vote_loss_sum / num_batches
    avg_objectness_loss = objectness_loss_sum / num_batches
    avg_box_loss = box_loss_sum / num_batches
    avg_sem_cls_loss = sem_cls_loss_sum / num_batches
    avg_obj_acc = obj_acc_sum / num_batches
    avg_pos_ratio = pos_ratio_sum / num_batches
    avg_neg_ratio = neg_ratio_sum / num_batches

    # Compute epoch time
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    elapsed_time = epoch_end_time - total_start_time
    avg_epoch_time = elapsed_time / (epoch + 1)
    remaining_epochs = total_epochs - (epoch + 1)
    estimated_finish_time = remaining_epochs * avg_epoch_time

    # Debugging: Print average metrics
    print(f"Epoch {epoch+1} Average Metrics:")
    print(f"Loss: {avg_loss}")
    print(f"Vote Loss: {avg_vote_loss}")
    print(f"Objectness Loss: {avg_objectness_loss}")
    print(f"Box Loss: {avg_box_loss}")
    print(f"Semantic Classification Loss: {avg_sem_cls_loss}")
    print(f"Objectness Accuracy: {avg_obj_acc}")
    print(f"Positive Ratio: {avg_pos_ratio}")
    print(f"Negative Ratio: {avg_neg_ratio}")
    print(f"Confusion Matrix: TP={epoch_TP}, FP={epoch_FP}, TN={epoch_TN}, FN={epoch_FN}")

    # Prepare metrics for CSV logging
    metrics = {
        'epoch': epoch + 1,
        'loss': avg_loss,
        'vote_loss': avg_vote_loss,
        'objectness_loss': avg_objectness_loss,
        'box_loss': avg_box_loss,
        'sem_cls_loss': avg_sem_cls_loss,
        'obj_acc': avg_obj_acc,
        'pos_ratio': avg_pos_ratio,
        'neg_ratio': avg_neg_ratio,
        'TP': epoch_TP,
        'FP': epoch_FP,
        'TN': epoch_TN,
        'FN': epoch_FN,
        'epoch_time_sec': epoch_time,
        'estimated_finish_time_sec': estimated_finish_time
    }

    # Log to CSV
    log_training_metrics(epoch, metrics)

    # Log to console and visualizer
    log_string(' ---- Epoch: %03d ----' % (epoch + 1))
    TRAIN_VISUALIZER.log_scalars({
        'loss': avg_loss,
        'vote_loss': avg_vote_loss,
        'objectness_loss': avg_objectness_loss,
        'box_loss': avg_box_loss,
        'sem_cls_loss': avg_sem_cls_loss,
        'obj_acc': avg_obj_acc,
        'pos_ratio': avg_pos_ratio,
        'neg_ratio': avg_neg_ratio,
        'TP': epoch_TP,
        'FP': epoch_FP,
        'TN': epoch_TN,
        'FN': epoch_FN
    }, epoch + 1)

    log_string('mean loss: %f' % avg_loss)
    log_string('mean vote_loss: %f' % avg_vote_loss)
    log_string('mean objectness_loss: %f' % avg_objectness_loss)
    log_string('mean box_loss: %f' % avg_box_loss)
    log_string('mean sem_cls_loss: %f' % avg_sem_cls_loss)
    log_string('mean obj_acc: %f' % avg_obj_acc)
    log_string('mean pos_ratio: %f' % avg_pos_ratio)
    log_string('mean neg_ratio: %f' % avg_neg_ratio)
    log_string(f'Confusion Matrix: TP={epoch_TP}, FP={epoch_FP}, TN={epoch_TN}, FN={epoch_FN}')
    log_string(f'Epoch time: {epoch_time:.2f} sec')
    log_string(f'Estimated finish time: {estimated_finish_time/60:.2f} min')


def evaluate_one_epoch(epoch):
    """
    Evaluate the model on the validation dataset for one epoch.

    Args:
        epoch (int): Current epoch number.

    Returns:
        float: The mean validation loss.
    """
    stat_dict = {}  # Dictionary to collect statistics
    ap_calculator = APCalculator(
        ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type
    )
    net.eval()  # Set the model to evaluation mode

    total_loss = 0.0  # Accumulate total loss
    num_batches = 0   # Count number of batches

    # Initialize confusion matrix counts
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    pos_ratio_sum = 0.0
    neg_ratio_sum = 0.0

    epoch_start_time = time.time()

    print(f"Starting evaluation epoch {epoch+1}")

    with torch.no_grad():  # No gradient computation during evaluation
        for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
            if batch_idx % 10 == 0:
                log_string(f"Eval batch: {batch_idx}/{len(TEST_DATALOADER)}")

            try:
                # Move data to device
                for key in batch_data_label:
                    batch_data_label[key] = batch_data_label[key].to(device)

                # Forward pass
                inputs = {'point_clouds': batch_data_label['point_clouds']}
                end_points = net(inputs)

                # Compute loss
                for key in batch_data_label:
                    end_points[key] = batch_data_label[key]  # Ensure ground truth is part of end_points
                loss, end_points = criterion(end_points, DATASET_CONFIG)
                total_loss += loss.item()
                num_batches += 1

                # Accumulate confusion matrix
                confusion = end_points.get('confusion_matrix', {})
                total_TP += confusion.get('TP', 0)
                total_FP += confusion.get('FP', 0)
                total_TN += confusion.get('TN', 0)
                total_FN += confusion.get('FN', 0)

                # Accumulate positive and negative ratios
                pos_ratio_sum += end_points['pos_ratio'].item()
                neg_ratio_sum += end_points['neg_ratio'].item()

                # Calculate Average Precision (AP)
                batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
                batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

                # Optionally save results
                if FLAGS.dump_results and batch_idx == 0 and (epoch + 1) % 10 == 0:
                    MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

            except Exception as e:
                print(f"Error during evaluation batch {batch_idx}: {e}")
                continue  # Skip to the next batch

    if num_batches == 0:
        print("No batches were processed during evaluation.")
        return None  # Skip the rest of the evaluation

    # Compute average metrics
    mean_loss = total_loss / num_batches
    avg_pos_ratio = pos_ratio_sum / num_batches
    avg_neg_ratio = neg_ratio_sum / num_batches

    # Compute epoch time
    epoch_end_time = time.time()
    eval_time = epoch_end_time - epoch_start_time

    # Debugging: Print average metrics
    print(f"Evaluation Epoch {epoch+1} Average Metrics:")
    print(f"Loss: {mean_loss}")
    print(f"Positive Ratio: {avg_pos_ratio}")
    print(f"Negative Ratio: {avg_neg_ratio}")
    print(f"Confusion Matrix: TP={total_TP}, FP={total_FP}, TN={total_TN}, FN={total_FN}")

    # Ensure all tensor values are converted to scalars
    center_loss = end_points.get('center_loss', 0.0)
    if isinstance(center_loss, torch.Tensor):
        center_loss = center_loss.item()

    heading_cls_loss = end_points.get('heading_cls_loss', 0.0)
    if isinstance(heading_cls_loss, torch.Tensor):
        heading_cls_loss = heading_cls_loss.item()

    heading_reg_loss = end_points.get('heading_reg_loss', 0.0)
    if isinstance(heading_reg_loss, torch.Tensor):
        heading_reg_loss = heading_reg_loss.item()

    size_cls_loss = end_points.get('size_cls_loss', 0.0)
    if isinstance(size_cls_loss, torch.Tensor):
        size_cls_loss = size_cls_loss.item()

    size_reg_loss = end_points.get('size_reg_loss', 0.0)
    if isinstance(size_reg_loss, torch.Tensor):
        size_reg_loss = size_reg_loss.item()

    sem_cls_loss = end_points.get('sem_cls_loss', 0.0)
    if isinstance(sem_cls_loss, torch.Tensor):
        sem_cls_loss = sem_cls_loss.item()

    # Prepare metrics for CSV logging
    metrics = {
        'epoch': epoch + 1,
        'loss': mean_loss,
        'center_loss': center_loss,
        'heading_cls_loss': heading_cls_loss,
        'heading_reg_loss': heading_reg_loss,
        'size_cls_loss': size_cls_loss,
        'size_reg_loss': size_reg_loss,
        'sem_cls_loss': sem_cls_loss,
        'pos_ratio': avg_pos_ratio,
        'neg_ratio': avg_neg_ratio,
        'TP': total_TP,
        'FP': total_FP,
        'TN': total_TN,
        'FN': total_FN,
        'eval_time_sec': eval_time,
        'estimated_finish_time_sec': 0  # Placeholder
    }

    # Log to CSV
    log_evaluation_metrics(epoch, metrics)

    # Log mean statistics for the epoch
    TEST_VISUALIZER.log_scalars(
        {
            'loss': mean_loss,
            'center_loss': center_loss,
            'heading_cls_loss': heading_cls_loss,
            'heading_reg_loss': heading_reg_loss,
            'size_cls_loss': size_cls_loss,
            'size_reg_loss': size_reg_loss,
            'sem_cls_loss': sem_cls_loss,
            'pos_ratio': avg_pos_ratio,
            'neg_ratio': avg_neg_ratio,
            'TP': total_TP,
            'FP': total_FP,
            'TN': total_TN,
            'FN': total_FN
        },
        epoch + 1
    )

    # Evaluate and log Average Precision (AP)
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string(f"eval {key}: {metrics_dict[key]}")

    # Log confusion matrix
    log_string(f'Eval Confusion Matrix: TP={total_TP}, FP={total_FP}, TN={total_TN}, FN={total_FN}')

    # Log epoch time
    log_string(f'Eval time: {eval_time:.2f} sec')
    log_string(f'Estimated finish time: N/A')  # Placeholder

    return mean_loss


def train(start_epoch, train_start_time):
    global EPOCH_CNT, best_loss
    best_loss = float('inf')  # Initialize the best loss for checkpointing
    val_loss = float('inf')  # Initialize validation loss for the first iteration

    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch + 1))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(epoch)))
        log_string(str(datetime.now()))

        # Reset numpy seed for randomness
        np.random.seed()

        # Train for one epoch
        train_one_epoch(epoch, train_start_time, MAX_EPOCH)

        # Evaluate after each epoch
        val_loss = evaluate_one_epoch(epoch)  # Collect validation loss

        # Skip saving the best checkpoint if val_loss is invalid
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            save_dict = {
                'epoch': epoch + 1,  # Save next epoch start
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
                'loss': best_loss,
                'lr': get_current_lr(epoch),  # Save current learning rate
            }
            torch.save(save_dict, os.path.join(LOG_DIR, 'best_checkpoint.tar'))
            log_string(f"Best checkpoint saved at epoch {epoch} with loss: {best_loss}")

        # Save checkpoint for resuming training
        save_dict = {
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
            'loss': val_loss if val_loss is not None else float('inf'),  # Save valid or high loss
            'lr': get_current_lr(epoch),  # Save current learning rate
        }
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

        log_string(f"Checkpoint saved at epoch {epoch} with loss: {val_loss}")

    # After all epochs, compute total runtime
    total_end_time = time.time()
    total_runtime = total_end_time - train_start_time
    log_string(f"Total training runtime: {total_runtime/3600:.2f} hours")


if __name__ == '__main__':
    EPOCH_CNT = 0
    best_loss = float('inf')  # Initialize best loss

    # Record the start time
    train_start_time = time.time()

    # Load from checkpoint if provided
    if FLAGS.checkpoint_path is not None and os.path.isfile(FLAGS.checkpoint_path):
        checkpoint = torch.load(FLAGS.checkpoint_path, map_location=device)
        pretrained_state_dict = checkpoint['model_state_dict']
        model_state_dict = net.state_dict()

        # Filter out keys that are not present in the model or have mismatched sizes
        matched_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k in model_state_dict and v.size() == model_state_dict[k].size():
                matched_state_dict[k] = v
            else:
                log_string(f"Skipping loading parameter '{k}' due to size mismatch: "
                           f"checkpoint {v.size()}, model {model_state_dict[k].size()}")

        # Update the model's state dict with the matched parameters
        model_state_dict.update(matched_state_dict)
        net.load_state_dict(model_state_dict)
        log_string("Loaded matched model parameters from checkpoint.")

        # Check if any parameters were skipped
        if len(matched_state_dict) != len(pretrained_state_dict):
            log_string("Some parameters were skipped due to size mismatch. "
                       "Optimizer state will not be loaded to avoid inconsistencies.")
            # Do not load optimizer state
            # Set EPOCH_CNT to 0 as the training cannot be resumed reliably
            if 'epoch' in checkpoint:
                log_string("Set EPOCH_CNT to 0 due to parameter mismatches.")
            EPOCH_CNT = 0
        else:
            # If all parameters matched, safely load optimizer state and epoch count
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log_string("Optimizer state loaded from checkpoint.")

            if 'epoch' in checkpoint:
                EPOCH_CNT = checkpoint['epoch']
                log_string(f"Resumed training from epoch {EPOCH_CNT}.")

            if 'loss' in checkpoint:
                best_loss = checkpoint['loss']
                log_string(f"Restored best loss: {best_loss}.")

            if 'lr' in checkpoint:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = checkpoint['lr']
                log_string(f"Resumed training with learning rate: {checkpoint['lr']}.")

        log_string(f"Loaded checkpoint '{FLAGS.checkpoint_path}' successfully.")

    # Initialize weights for mismatched layers if any
    def weights_init(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    net.apply(weights_init)
    log_string("Initialized mismatched layers with Xavier uniform initialization.")

    # Start training
    train(EPOCH_CNT, train_start_time)
