import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import AutoEncoder
from dataset import FeatureExtractorDataset
from utils import draw_spec
import torch.nn.functional as F
import os
from tqdm import tqdm

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("CHECKPOINT LOADED! **** ")
    return checkpoint['epoch']

def calculate_mse(gt, recon):
    mse = F.mse_loss(gt,recon)
    return mse.item()

# def visualize_comparison(gt_spec, input_spec, recon_spec, wb, index, output_dir, sample_rate=48000, freq_range=[6, 15], vmin=-1.1, vmax=1.6, show_colorbar=True):
#     """
#     Visualizes the Ground Truth (GT), Input (Masked), and Reconstructed spectrograms in a single figure.
#     Saves the figure to the specified output directory with the dataset index.
#     """

#     # Create 1 row, 3 columns figure
#     fig, axes = plt.subplots(3, 1, figsize=(8, 10))

#     freqsize = sample_rate / (2*32)
#     freq_range[1] = freq_range[1] + 1
    
#     # Plot GT spectrogram
#     im = axes[0].imshow(gt_spec.squeeze(), aspect='auto', origin='lower', cmap='inferno',
#                    extent=[0, wb.shape[-1] / sample_rate, freq_range[0] * freqsize, freq_range[1] * freqsize],
#                    vmin=vmin, vmax=vmax)
#     axes[0].set_title("Ground Truth (GT)")
#     axes[0].set_xlabel('Time (s)')
#     axes[0].set_ylabel('Frequency (Hz)')

#     # Plot Input (Masked) spectrogram
#     axes[1].imshow(input_spec.squeeze(), aspect='auto', origin='lower', cmap='inferno',
#                    extent=[0, wb.shape[-1] / sample_rate, freq_range[0] * freqsize, freq_range[1] * freqsize],
#                    vmin=vmin, vmax=vmax)
#     axes[1].set_title("Input (Masked)")
#     axes[1].set_xlabel('Time (s)')
#     axes[1].set_ylabel('Frequency (Hz)')

#     # Plot Reconstructed spectrogram
#     axes[2].imshow(recon_spec.squeeze(), aspect='auto', origin='lower', cmap='inferno',
#                    extent=[0, wb.shape[-1] / sample_rate, freq_range[0] * freqsize, freq_range[1] * freqsize],
#                    vmin=vmin, vmax=vmax)
#     axes[2].set_title("Reconstructed")
#     axes[2].set_xlabel('Time (s)')
#     axes[2].set_ylabel('Frequency (Hz)')

#     if show_colorbar:
#         fig.subplots_adjust(right=0.85)  # Adjust space on the right for the colorbar
#         cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  # Colorbar position [left, bottom, width, height]
#         fig.colorbar(im, cax=cbar_ax)

#     # Adjust layout to ensure nothing is cut off, especially the colorbar
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to account for the colorbar

#     # Save figure to outputs directory
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{index}.png")
#     plt.savefig(output_path, bbox_inches='tight')  # Use bbox_inches='tight' to ensure everything is saved
#     plt.close(fig)  # Close the figure to avoid display in interactive environments
#     print(f"Figure saved for index {index}: {output_path}", end="\r")

def visualize_comparison(gt_spec, input_spec, recon_spec, wb, index, output_dir, sample_rate=48000, freq_range=[6, 15], vmin=-1.1, vmax=1.6, 
                         show_colorbar=True, save_separate=False):
    """
    Visualizes the Ground Truth (GT), Input (Masked), and Reconstructed spectrograms.
    Saves the figure(s) to the specified output directory with the dataset index.
    If save_separate is True, each spectrogram is saved as a separate figure.
    Otherwise, all three are saved in a single figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    freqsize = sample_rate / (2*32)
    freq_range[1] = freq_range[1] + 1
    
    if save_separate:
        # Save each spectrogram separately
        specs = [(gt_spec, "Ground Truth (GT)"), (input_spec, "Input (Masked)"), (recon_spec, "Reconstructed")]
        for i, (spec, title) in enumerate(specs):
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(spec.squeeze(), aspect='auto', origin='lower', cmap='inferno',
                           extent=[0, wb.shape[-1] / sample_rate, freq_range[0] * freqsize, freq_range[1] * freqsize],
                           vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            
            if show_colorbar:
                fig.colorbar(im, ax=ax)
            
            # Save each figure
            output_path = os.path.join(output_dir, f"{index}_{i}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Separate figure saved: {output_path}", end="\r")
    else:
        # Create 1 row, 3 columns figure as before
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        
        # Plot GT spectrogram
        im = axes[0].imshow(gt_spec.squeeze(), aspect='auto', origin='lower', cmap='inferno',
                            extent=[0, wb.shape[-1] / sample_rate, freq_range[0] * freqsize, freq_range[1] * freqsize],
                            vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth (GT)")
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')

        # Plot Input (Masked) spectrogram
        axes[1].imshow(input_spec.squeeze(), aspect='auto', origin='lower', cmap='inferno',
                       extent=[0, wb.shape[-1] / sample_rate, freq_range[0] * freqsize, freq_range[1] * freqsize],
                       vmin=vmin, vmax=vmax)
        axes[1].set_title("Input (Masked)")
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')

        # Plot Reconstructed spectrogram
        axes[2].imshow(recon_spec.squeeze(), aspect='auto', origin='lower', cmap='inferno',
                       extent=[0, wb.shape[-1] / sample_rate, freq_range[0] * freqsize, freq_range[1] * freqsize],
                       vmin=vmin, vmax=vmax)
        axes[2].set_title("Reconstructed")
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Frequency (Hz)')

        if show_colorbar:
            fig.subplots_adjust(right=0.85)  # Adjust space on the right for the colorbar
            cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  # Colorbar position [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)

        # Adjust layout to ensure nothing is cut off, especially the colorbar
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to account for the colorbar

        # Save figure to outputs directory
        output_path = os.path.join(output_dir, f"{index}.png")
        plt.savefig(output_path, bbox_inches='tight')  # Use bbox_inches='tight' to ensure everything is saved
        plt.close(fig)  # Close the figure to avoid display in interactive environments
        print(f"Combined figure saved for index {index}: {output_path}", end="\r")



def main():
    # SEED
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device (GPU or CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and checkpoint
    weight_path = "./weights/epoch_170_mse_0.008.pth"
    model = AutoEncoder().to(DEVICE)
    load_checkpoint(model, weight_path)

    # Load dataset
    # data_path = ["/mnt/hdd/Dataset/MUSDB18_HQ_mono_48kHz/test"]
    data_path = ["/mnt/hdd/Dataset/VCTK/wav48/p225"]
    # data_path = ["/mnt/hdd/Dataset/FSD50K_48kHz/FSD50K.eval_audio"]
    dataset = FeatureExtractorDataset(data_path)

    # Output directory for saving figures
    output_dir = "output_VCTK_170"
    # output_dir = "output_MUSDB3"
    # output_dir = "output_FSD50K_170"

    mse_list = []
    # Process and visualize random samples in the dataset
    NUMSAMPLE = 5

    pbar = tqdm(range(len(dataset)))
    # for index in pbar:
    for index in (torch.randperm(200)[:NUMSAMPLE] + 1):  # Randomly select 5 samples
        index = index.item()  # Convert tensor to Python integer
        wb, spec, mask, spece, maske, name = dataset[index]
        print(f"Processing: {name}", end="\r")

        with torch.no_grad():
            # Reconstruct the masked input
            recon = model(maske.to(DEVICE)).detach()

        # Calculate MSE between ground truth and reconstructed spectrogram
        mse = calculate_mse(spece.to(DEVICE), recon)
        print(f"MSE for index {index} ({name}): {mse:.4f}", end="\r")

        # Append MSE to list
        mse_list.append(mse)

        spece, maske, recon = spece.to('cpu'), maske.to('cpu'), recon.to('cpu')
        # Visualize GT, Masked input, and Reconstructed spectrograms, save to outputs/
        visualize_comparison(gt_spec=spece, input_spec=maske, recon_spec=recon, wb=wb, index=index, 
                             freq_range=[6, 15], output_dir=output_dir, show_colorbar=False,
                             save_separate=True)

    # Calculate and print the average MSE for the entire dataset
    average_mse = sum(mse_list) / len(mse_list)
    print(f"\nAverage MSE for the dataset: {average_mse:.4f}")  

if __name__ == "__main__":
    main()