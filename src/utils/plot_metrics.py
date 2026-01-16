import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"



def plot_metrics(csv_file: str, mapping: dict, save_dir: str = None):
    """
    Reads a CSV with WandB run metrics and plots each metric over epochs for all runs,
    preserving special capitalization (e.g. 'mIoU') and mapping 'pxAccuracy' to 'Pixel Accuracy'.

    Parameters:
    - csv_file: path to CSV (first column 'Step', others '<run> - <metric>').
    - mapping: dict mapping WandB run IDs to display names.
    - save_dir: directory to save figures; defaults to CSV folder.
    """
    # Load data
    df = pd.read_csv(csv_file)
    df = df[df['Step'] < 40] 
    epochs = df['Step']

    # Derive scenario from filename (before first underscore)
    fname = os.path.basename(csv_file)
    scenario = fname.split('_')[0]

    # Prepare output directory
    if save_dir is None:
        save_dir = os.path.dirname(csv_file)
    os.makedirs(save_dir, exist_ok=True)

    # Identify metrics (ignore __MIN/__MAX columns)
    metrics = set()
    for col in df.columns:
        if col == 'Step' or '__MIN' in col or '__MAX' in col:
            continue
        parts = col.split(' - ')
        if len(parts) == 2:
            metrics.add(parts[1])

    # Special-case labels
    special_labels = {
        'train_miou': 'Train mIoU',
        'val_miou': 'Validation mIoU',
        'iou': 'IoU',
        'cpu': 'CPU',
        'fps': 'FPS',
        'train_pixelaccuracy': 'Train Pixel Accuracy',
        'val_pixelaccuracy': 'Validation Pixel Accuracy'
    }

    # Styles
    markers = ['s', 'o', '^', 'd', 'v', 'P', '*']
    lines = ['-', '--', '-.', ':']

    for metric in sorted(metrics):
        key = metric.lower()
        ylabel = special_labels.get(key, metric.replace('_', ' ').title())
        fig, ax = plt.subplots(figsize=(8, 5))

        # Frame spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        # Plot each run
        for idx, col in enumerate(df.columns):
            if f' - {metric}' not in col or '__' in col:
                continue
            run_id, _ = col.split(' - ')
            label = mapping.get(run_id, run_id)
            ax.plot(epochs, df[col], label=label,
                    marker=markers[idx % len(markers)],
                    linestyle=lines[idx % len(lines)],
                    markersize=5)

        # Grid
        ax.grid(True, which='major', linewidth=0.5, color='lightgray')
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Labels & title
        ax.set_xlabel('Epoch', fontsize=25)
        ax.set_title(f'{ylabel}', fontsize=30, pad=10)

        # Legend with border
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) == 4:
            order = [3, 1, 0, 2]
            leg = ax.legend([handles[i] for i in order],
                [labels[i] for i in order],frameon=True, fontsize=25)
        else:
            leg = ax.legend(frameon=True, fontsize=25)
        leg.get_frame().set_edgecolor('black')

        plt.tight_layout(pad=2)

        # Save plot
        out_path = os.path.join(save_dir, f'{scenario}_{metric}.pdf')
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


def batch_plot(csv_dir: str, mapping: dict, save_dir: str = None):
    """
    Iterates over all CSV files in a directory and calls plot_metrics for each.

    Parameters:
    - csv_dir: directory containing CSV files.
    - mapping: dict mapping WandB run IDs to display names.
    - save_dir: directory to save all plots; defaults to csv_dir.
    """
    pattern = os.path.join(csv_dir, '*.csv')
    files = glob.glob(pattern)
    if not files:
        print(f"No CSV files found in {csv_dir}")
        return

    for csv_file in files:
        plot_metrics(csv_file, mapping, save_dir)

if __name__ == '__main__':
    # Example usage:
    # Define mapping from WandB run IDs to model names
    mapping = {
        'lively-mountain-127': 'Frozen Backbone',
        'youthful-brook-67': 'Fine-tuned Backbone',
        'rose-puddle-95': 'Early Fusion',
        'lively-wave-77': "Mid Fusion", 
        'lyric-energy-103': 'LoRA Fusion',
        'trim-frost-110': 'CMA Fusion',
        'solar-spaceship-178': 'Freeze Schedule',
        'lunar-energy-183': 'Deeper Encoder',
        'honest-plasma-175': 'ConvT Decoder'
    }

    # Directory with CSV files
    csv_directory = "/home/francisco.m.ribeiro/PDEEC/VC/Assignment 2/rgbd-align/results/plots/"

    # (Optional) directory to save plots
    save_directory = "/home/francisco.m.ribeiro/PDEEC/VC/Assignment 2/rgbd-align/results/plots/plots_all/"

    batch_plot(csv_directory, mapping, save_directory)
