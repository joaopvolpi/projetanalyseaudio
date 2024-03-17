import React from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';

const MainScreen = ({navigation}) => {
  const [recording, setRecording] = React.useState();
  const [recordings, setRecordings] = React.useState([]);

  async function startRecording() {
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (perm.status === "granted") {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true
        });
        const { recording } = await Audio.Recording.createAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
        setRecording(recording);
      }
    } catch (err) {}
  }

  async function stopRecording() {
    setRecording(undefined);

    await recording.stopAndUnloadAsync();
    let allRecordings = [...recordings];
    const { sound, status } = await recording.createNewLoadedSoundAsync();
    allRecordings.push({
      sound: sound,
      duration: getDurationFormatted(status.durationMillis),
      file: recording.getURI()
    });

    setRecordings(allRecordings);
  }

  function getDurationFormatted(milliseconds) {
    const minutes = milliseconds / 1000 / 60;
    const seconds = Math.round((minutes - Math.floor(minutes)) * 60);
    return seconds < 10 ? `${Math.floor(minutes)}:0${seconds}` : `${Math.floor(minutes)}:${seconds}`
  }

  function getRecordingLines() {
    return recordings.map((recordingLine, index) => {
      return (
        <View key={index} style={styles.row}>
          <Text style={styles.fill}>
            Recording #{index + 1} | {recordingLine.duration}
          </Text>
          <Button onPress={() => recordingLine.sound.replayAsync()} title="Play"></Button>
          <Button onPress={() => uploadAudio(recordingLine.file)} title="Send"></Button>
          <Button onPress={() => navigation.navigate(('Analysis'))} title="Analyse"></Button>
        </View>
      );
    });
  }

  function clearRecordings() {
    setRecordings([])
  }

  async function uploadAudio(uri) {
    const apiUrl = 'http://10.0.2.2:5000/upload-audio';
    const fileType = 'audio/wav'; // Assuming the audio is in WAV format
    const fileName = 'filename.3gp'; // Or dynamically generate this based on your requirements
  
    // Create a new FormData instance
    const formData = new FormData();
    
    // Append the file data to the FormData object
    // React Native's fetch API supports direct file uploads using file URIs
    // No need to read the file as a base64-encoded string
    formData.append('audio', {
        name: fileName,
        type: fileType,
        uri: uri,
    });
    console.log(uri)
    // Options for the fetch request
    const options = {
        method: 'POST',
        body: formData,
        // Do NOT explicitly set the Content-Type header
        // The browser/fetch API will automatically set it with the correct boundary
        headers: {
            // 'Content-Type': 'multipart/form-data', // Remove this line
        },
    };

    try {
        const response = await fetch(apiUrl, options);
        const result = await response.json();
        console.log(result);
    } catch (error) {
        console.error(error);
    }
}

  return (
    <View style={styles.container}>
      <Button title={recording ? 'Stop Recording' : 'Start Recording'} onPress={recording ? stopRecording : startRecording} />
      {getRecordingLines()}
      <Button title='Clear Recordings' onPress={clearRecordings} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 10,
    marginRight: 40
  },
  fill: {
    flex: 1,
    margin: 15
  }
});

export default MainScreen;