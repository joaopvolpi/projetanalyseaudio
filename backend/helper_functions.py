import os
import io
import base64
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import *
from datetime import datetime
from pydub import AudioSegment
from scipy.io.wavfile import read
from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def plot_fft_from_wav(file_path, file_name):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Compute FFT
    fft = np.fft.fft(y)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fft)
    
    # Create frequency variable
    frequency = np.linspace(0, sr, len(magnitude))

    # Plotting the magnitude spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(frequency[:int(len(frequency)/2)], magnitude[:int(len(magnitude)/2)]) # Plotting only the first half to avoid mirroring
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # img_path = os.path.join(UPLOAD_FOLDER, f'{file_name}_fft.png')
    # plt.savefig(img_path)
    # plt.close()
    
    # Convert plot to image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode image to base64
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_data

def plot_audio_time(file_path, file_name):
    # Read audio samples
    input_data = read(file_path)
    audio = input_data[1]

    plt.plot(audio)

    # Label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time")

    # Set the title
    plt.title("Sample Wav")

    # Display the plot
    # img_path = os.path.join(UPLOAD_FOLDER, f'{file_name}_time.png')
    # plt.savefig(img_path)
    # plt.close()
    

    # Convert plot to image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return image_data

def convert_3gp_to_wav(input_file, output_file):
    # Load the .3gp file
    audio = AudioSegment.from_file(input_file, format="3gp")
    
    # Export the audio to .wav format
    audio.export(output_file, format="wav")

def send_to_azure(file_path):
    connect_str = SECRET_CONNECTION_KEY

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    container_name = 'projet-cs-sound'

    blob_name = file_path.split('\\')[-1]

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)