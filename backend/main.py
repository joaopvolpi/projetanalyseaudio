from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from pydub import AudioSegment
import wave
import base64
import io

app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\joaop\Documents\CentraleSupélec\3A\Projet Sopra Steria\AnalyseAudioExpo\backend'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
   # Check if the request contains 'audio' as a file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    audio = request.files['audio']
    print("AAAAAAAAA", audio)
    # Define the path where the file will be saved
    filepath = os.path.join(UPLOAD_FOLDER, audio.filename)
    
    save_path = os.path.join(r'C:\Users\joaop\Documents\CentraleSupélec\3A\Projet Sopra Steria\AnalyseAudioExpo\backend\output.wav')
    
    # Save the file to the specified path
    audio.save(filepath)

    print("Received audio file:", audio.filename)

    save_audio_blob_to_wav(audio.read(), save_path)

    # audio_vector, fft_result = process_audio(filepath)

    response = jsonify({
        'message': 'File uploaded and processed successfully'
        # 'filename': audio.filename,
        # 'audio_vector': audio_vector.tolist(),  # Convert numpy array to list for JSON serialization
        # 'fft': fft_result.tolist()
    })
    print(response)
    return response, 200


@app.route('/test', methods=['POST'])
def test():
    return jsonify({
        'message': 'bonjour'
    }), 200

    
def process_audio(file_path):
    print(file_path)
    # Read the audio file
    audio = AudioSegment.from_wav(file_path)

    # Compute the FFT
    fft_result = fft(audio)
    
    # Return only the magnitude of the FFT and the audio vector
    fft_magnitude = np.abs(fft_result)
    
    return audio, fft_magnitude

def save_audio_blob_to_wav(audio_blob, output_file_path):
    # Decode the base64 encoded audio data
    audio_data = base64.b64decode(audio_blob)

    # Write the audio data to a WAV file
    with wave.open(output_file_path, 'wb') as wav_file:
        # Set the WAV file parameters (1 channel, 16-bit, 44100 Hz sample rate)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 2 bytes for 16-bit audio
        wav_file.setframerate(44100)

        # Write the audio data to the WAV file
        wav_file.writeframes(audio_data)

# Example usage:
audio_blob = b'base64_encoded_audio_data_here'
output_file_path = 'output.wav'


if __name__ == '__main__':
    app.run(debug=True)
