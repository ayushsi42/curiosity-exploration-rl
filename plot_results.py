import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_results(log_files, labels, title):
    """
    Plots the training curves from one or more log files.
    """
    plt.figure(figsize=(12, 8))

    for log_file, label in zip(log_files, labels):
        df = pd.read_csv(log_file)
        plt.plot(df['episode'], df['reward'].rolling(100).mean(), label=label)

    plt.xlabel('Episode')
    plt.ylabel('Reward (100-episode moving average)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(title.replace(' ', '_') + '.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training results.')
    parser.add_argument('--log-files', nargs='+', required=True, help='List of log files to plot.')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for each log file.')
    parser.add_argument('--title', type=str, default='Training Curves', help='Title of the plot.')
    
    args = parser.parse_args()
    plot_results(args.log_files, args.labels, args.title)

