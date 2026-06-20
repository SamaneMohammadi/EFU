import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from mplfonts import use_font

def plot_accuracies(log_fp, save_fp=None):
    with open(log_fp, "r") as f:
        lines = f.readlines()
    rounds, train_accuracies, test_accuracies = [], [], []
    train_accs = []
    dataset, num_rounds = None, None
    for line in lines:
        if "[Server]" in line and "test_acc" in line:
            match = re.search(r"test_acc: ([\d.]+)%", line)
            if match:
                test_accuracies.append(float(match.group(1)))
        elif "[Client" in line and "train_acc" in line:
            match = re.search(r"train_acc: ([\d.]+)%", line)
            if match:
                train_accs.append(float(match.group(1)))
        elif "Federated Round" in line:
            if train_accs:
                train_accuracies.append((np.mean(train_accs), np.std(train_accs)))
                train_accs = []
            rounds.append(len(rounds) + 1)
        elif "dataset:" in line:
            dataset = re.search(r"dataset: (\w+)", line).group(1).upper()
        elif "num_rounds:" in line:
            num_rounds = int(re.search(r"num_rounds: (\d+)", line).group(1))
    if train_accs:
        train_accuracies.append((np.mean(train_accs), np.std(train_accs)))
    train_means, train_stds = zip(*train_accuracies)
    if save_fp is None:
        save_fp = f"{os.path.splitext(log_fp)[0]}.png"
    use_font('SegoeUI')
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(rounds, test_accuracies, label="test_acc", linewidth=2)
    plt.plot(rounds, train_means, label="train_acc", linewidth=2)
    plt.fill_between(rounds, np.array(train_means) - np.array(train_stds), np.array(train_means) + np.array(train_stds), alpha=0.2)
    plt.title(f"{dataset}", fontsize=16, fontweight="bold")
    plt.xlabel("Federated Rounds", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xticks(ticks=list(range(1, num_rounds + 1)))  # Set ticks from 1 to num_rounds
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="upper left", frameon=True, fontsize=10, fancybox=True, title_fontsize=12)
    plt.savefig(save_fp, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and test accuracies from log file.")
    parser.add_argument("log_fp", type=str, help="Path to the log file.")
    parser.add_argument("--save_fp", type=str, default=None, help="Path to save the plot image.")
    args = parser.parse_args()
    plot_accuracies(args.log_fp, args.save_fp)