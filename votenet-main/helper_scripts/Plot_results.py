import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_and_evaluation_results(train_log_path, eval_log_path, output_dir):
    """
    Generate plots for each column in the training and evaluation log files.

    Args:
        train_log_path (str): Path to the training log CSV file.
        eval_log_path (str): Path to the evaluation log CSV file.
        output_dir (str): Directory to save the generated plots.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the CSV files
    train_log = pd.read_csv(train_log_path)
    eval_log = pd.read_csv(eval_log_path)

    # Plot each column in the training log
    for column in train_log.columns:
        if column == "epoch":
            continue  # Skip the 'epoch' column
        plt.figure()
        plt.plot(train_log["epoch"], train_log[column], label=f"Training {column}")
        plt.xlabel("Epoch")
        plt.ylabel(column)
        plt.title(f"Training {column} vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"training_{column}.png"))
        plt.close()

    # Plot each column in the evaluation log
    for column in eval_log.columns:
        if column == "epoch":
            continue  # Skip the 'epoch' column
        plt.figure()
        plt.plot(eval_log["epoch"], eval_log[column], label=f"Evaluation {column}")
        plt.xlabel("Epoch")
        plt.ylabel(column)
        plt.title(f"Evaluation {column} vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"evaluation_{column}.png"))
        plt.close()

    print(f"Plots saved to {output_dir}")


# Paths to the uploaded CSV files
train_log_path = "C:/PyCharmProjects/FinalProject/votenet-main/log_150_epochs/train_log.csv"
eval_log_path = "C:/PyCharmProjects/FinalProject/votenet-main/log_150_epochs/eval_log.csv"
output_dir = "C:/PyCharmProjects/FinalProject/votenet-main/log_150_epochs/log_plots"

plot_training_and_evaluation_results(train_log_path, eval_log_path, output_dir)
