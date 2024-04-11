# Electrical Equipment Failure Detection Tool

## Overview

This project aims to develop a portable tool for detecting failures in electrical equipment using sound analysis.

## Features

- **Audio Data Capture**: Capture audio from electrical equipment using a mobile device.
- **Audio Analysis**: Analyze audio signals using Fast Fourier Transforms (FFT) and spectrograms to detect anomalies.
- **Real-Time AI Analysis**: Utilize an AI model to classify the equipment status as either 'potential failure' or 'regular functioning'. The model is not yet implemented, since the dataset is still being collected.
- **Mobile App**: A user-friendly mobile application that allows technicians to record, analyze, and upload audio data to a centralized server.
- **Data Collection**: Facilitate data gathering for continuous improvement of the AI model.

## Project Structure

- `/projectenedis`: Contains the source code for the mobile application developed using React Native.
- `/backend`: Flask API for handling audio data processing and machine learning model interaction.
- `/backend/AI`: Preprocessing files and notebook used for model training.

## Getting Started

### Prerequisites

- Node.js
- Python 3.x
- Flask
- React Native environment set up (see React Native documentation)

### Installation

1. Clone the repository:

https://github.com/joaopvolpi/projetanalyseaudio.git

2. Install dependencies for the backend:

cd backend
pip install -r requirements.txt

3. Install dependencies for the app:

cd ../projectenedis
npm install

4. Create config.py file:

Use config_example.py as an example
Create file in folder /backend

### Running the Application

1. Start the Flask backend:

cd backend
flask --app .\main.py run

2. Run the React Native app:

cd ../app
npm start

## Usage

- Open the app on your mobile device.
- Follow the on-screen instructions to record audio from the equipment.
- Analyze the recording to detect potential failures.

