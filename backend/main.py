import os
import io
import base64
import librosa
import numpy as np
import matplotlib.pyplot as plt

from config import *
from datetime import datetime
from pydub import AudioSegment
from scipy.io.wavfile import read
from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

app = Flask(__name__)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
   # Check if the request contains 'audio' as a file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    audio = request.files['audio']

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")    
    file_name = f"Rec_{timestamp}"
    audio.filename = f"{file_name}.3gp"
    
    received_audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    
    # Save original audio file
    audio.save(received_audio_path)
    

    converted_file_path = os.path.join(UPLOAD_FOLDER, f'{file_name}.wav')
    
    convert_3gp_to_wav(received_audio_path, converted_file_path)

    fft_img = plot_fft_from_wav(converted_file_path, file_name)
    time_img = plot_audio_time(converted_file_path, file_name)

    response = jsonify({
        'message': 'File uploaded and processed successfullyy',
        'fft_img': fft_img,
        'time_img': time_img
    })
    return response, 200


@app.route('/test', methods=['POST'])
def test():
    return jsonify({
        'message': UPLOAD_FOLDER
    }), 200

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


if __name__ == '__main__':
    app.run(debug=True)
