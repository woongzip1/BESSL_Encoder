import os
import shutil
import argparse
import torch
import numpy as np
from scipy.signal import stft
import scipy.signal as sig
import torch.nn.functional as F

from matplotlib import pyplot as plt
import librosa

"""
*** Utils ***
- get_audio_paths()
- get_filename()
- mask_random_subbands()
- draw_spec()
- plot_hist()
"""

""" Return all audio paths in sub-directories """
def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
        
    for path in paths:
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1].lower() in file_extensions]
                        
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def get_filename(path):
    return os.path.splitext(os.path.basename(path))  

def mask_random_subbands(spec, num_subbands=32, mask_fraction=0.5):
    """
    Mask the random subbands in the spectrogram, independently for each frame.
    
    Args:
    spec (torch.Tensor): The input spectrogram with shape (B, 1, 1025, T).
    num_subbands (int): Total number of subbands to divide the frequency bins into.
    mask_fraction (float): Fraction of subbands to mask (e.g., 0.5 for 50%).

    Returns:
    torch.Tensor: The masked spectrogram.
    """

    # if spec.dim() == 3: # C x F x T
        # spec = spec.unsqueeze(0)
        
    C, F, T = spec.shape
    
    # The first subband includes indices 0-32 (33 frequency bins)
    subband0_size = 33
    
    # The remaining subbands each contain 32 frequency bins
    remaining_size = F - subband0_size
    subband_size = remaining_size // (num_subbands - 1)
    
    # First subband (indices 0-32)
    subband0 = spec[:, :subband0_size, :]
    
    # Remaining subbands (split into chunks of 32 frequency bins)
    subbands = [subband0]
    start_idx = subband0_size
    for _ in range(1, num_subbands):
        end_idx = min(start_idx + subband_size, F)
        subbands.append(spec[ :, start_idx:end_idx, :])
        start_idx = end_idx
    
    # Initialize the masked spectrogram as a list of tensors
    masked_spec = torch.zeros_like(spec)

    MASKINGVALUE = -1.112640558355952 # -100 dB

    # For each frame, mask different subbands
    for t in range(T):
        # Select ratio of the subbands randomly for the current frame
        num_to_mask = int(mask_fraction * num_subbands)
        subband_indices = np.random.choice(num_subbands, num_to_mask, replace=False)
        
        # Apply masking to the selected subbands for the current frame
        masked_subbands = []
        for i, subband in enumerate(subbands):
            if i in subband_indices:
                # Mask this subband for the current frame
                masked_frame = torch.ones_like(subband[:, :, t:t+ 1]) * MASKINGVALUE
            else:
                # Keep the subband as is for the current frame
                masked_frame = subband[ :, :, t:t+1]
            
            masked_subbands.append(masked_frame)
        
        # Concatenate the masked subbands along the frequency dimension
        masked_spec[:, :, t:t+1] = torch.cat(masked_subbands, dim=1)
    
    return masked_spec

# 2-D input spectrogram
# 2-D input spectrogram
def draw_spec(spectrogram, sample_rate, seglen=2, title="Spectrogram", 
                figsize=(10,4), vmin=-1.1, vmax=1.6, freq_range=None):
    n_fft = 2048

    # frequency axis
    freq_bins = np.fft.rfftfreq(n_fft, d=1/sample_rate)
    if freq_range:
        s,e = freq_range
        start = 24000 / 32 * s
        end = 24000 / 32 * (e+1)

    else:
        start = freq_bins.min()
        end = freq_bins.max()
    fig=plt.figure(figsize=figsize)
    # imshow
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno',
                extent=[0, seglen, start, end],
                vmin=vmin, vmax=vmax,
                )

    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    return fig

def plot_hist(spec):
    spec_flat = spec.flatten().numpy()
    
    # Get min and max values
    min_val = spec_flat.min()
    max_val = spec_flat.max()
    
    print("Min value:", min_val)
    print("Max value:", max_val)
    
    # Plot hist
    plt.figure(figsize=(4, 4))
    plt.hist(spec_flat, bins=50, color='blue', alpha=0.7)
    
    # Set x-axis limits to the min and max values of the data
    plt.xlim(min_val, max_val)
    plt.title('Histogram of Spectrogram Values')
    plt.xlabel('Spectrogram Value (dB)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("CHECKPOINT LOADED! **** ")
    return checkpoint['epoch']