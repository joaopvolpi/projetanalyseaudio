import os
import io
import base64
import librosa
import numpy as np
import matplotlib.pyplot as plt

from config import *
from helper_functions import *
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyse-audio', methods=['POST'])
def analyse_audio():
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

    fft_array = get_fft_from_wav(converted_file_path)
    
    time_array = get_audio_from_wav(converted_file_path)

    os.remove(received_audio_path)
    os.remove(converted_file_path)

    response = jsonify({
        'message': 'File uploaded and processed successfullyy',
        'fft_img': fft_img,
        'time_img': time_img,
        'fft_array': fft_array,
        'time_array': time_array
    })
    return response, 200


@app.route('/test', methods=['POST'])
def test():
    return jsonify({
        'message': 'Hello World!'
    }), 200


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    audio = request.files['audio']

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")    
    file_name = f"Rec_{timestamp}"
    audio.filename = f"{file_name}.3gp"
    received_audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    
    audio.save(received_audio_path)
    # send_to_azure(received_audio_path) # Cloud Upload
    os.remove(received_audio_path)

    return jsonify({
        'message': 'File Successfully Uploaded!'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
