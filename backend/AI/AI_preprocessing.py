import cv2
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Helpers

def create_mel_spectrogram(audio_path):
    # Load the audio file
    waveform, sampling_rate = librosa.load(audio_path, sr=None)

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)

    # Convert to decibels (log scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db,sampling_rate

def save_mel_spectrogram_image(audio_path, output_image_path, target_size=(256, 256)):
    # Load the audio file
    mel_spectrogram, sampling_rate = create_mel_spectrogram(audio_path)

    # Create a figure
    fig = plt.figure(frameon=False)
    fig.set_size_inches(target_size[0] / 100, target_size[1] / 100)  # Convert from pixels to inches

    # Add an axis to the figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Draw the Mel spectrogram on the axis
    ax.imshow(mel_spectrogram, aspect='auto', cmap='viridis', origin='lower', extent=(0, mel_spectrogram.shape[1], 0, mel_spectrogram.shape[0]))

    # Save the figure as an image file
    fig.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
# save_mel_spectrogram_image(audio_file_path, "mel_spectrogram.png")
