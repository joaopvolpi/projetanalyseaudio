import React from 'react';
import { StyleSheet, Text, View, ScrollView, Button, Image } from 'react-native';
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
        const { recording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
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
          Audio #{index + 1} | {recordingLine.duration}
        </Text>
        <View style={styles.actionButtonContainer}>
          <Button onPress={() => recordingLine.sound.replayAsync()} title="Play" color="#41a57d" />
        <View style={{ width: 2 }} /> 
          <Button onPress={() => analyseAudio(recordingLine.file)} title="Analyse" color="#1423dc" />
        </View>
      </View>
      );
    });
  }

  function clearRecordings() {
    setRecordings([])
  }

  async function analyseAudio(uri) {
    const apiUrl = 'http://10.0.2.2:5000/analyse-audio';
    const fileType = 'audio/3gp'; 
    const fileName = 'filename.3gp';
    const formData = new FormData();
    
    formData.append('audio', {
        name: fileName,
        type: fileType,
        uri: uri,
    });

    const options = {
        method: 'POST',
        body: formData,

        headers: {
        },
    };

    try {
      const response = await fetch(apiUrl, options);
      const result = await response.json();
      
      if (response.ok) {
          navigation.navigate('Analysis', { fftImage: result.fft_img, timeImage: result.time_img, fft_array: result.fft_array, 
          wav_array:result.time_array,audio: uri });
      } else {
          console.error(result.message);
      }
  } catch (error) {
      console.error(error);
  }
}

  return (
    <ScrollView style={styles.scrollView}>
      <View style={styles.container}>
      <Image source={require('../assets/Logo_enedis.png')} style={styles.logo} />
        <View style={styles.buttonContainer}>
          <Button 
            title={recording ? 'Arrêter enregistrement' : 'Enregistrer Audio'} 
            onPress={recording ? stopRecording : startRecording} 
            color="#41a57d"
          />
        </View>
        {getRecordingLines()}
        <View style={styles.buttonContainer}>
          <Button 
            title='Supprimer Audio'
            onPress={clearRecordings} 
            color="#a52a2a" // Reddish color for Clear Recordings button
          />
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scrollView: {
    backgroundColor: '#e9f3fb', // light_blue_enedis
  },
  container: {
    flex: 1,
    backgroundColor: '#e9f3fb', // light_blue_enedis
    alignItems: 'center',
    justifyContent: 'flex-start', // Changed to flex-start to align items from the top
    paddingTop: 20, // Add some padding at the top
  },
  row: {
    justifyContent: 'space-between', // Adjusted for spacing between Play and Analyse buttons
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between', // Adjust spacing
    width: '90%', // Adjust width to ensure consistency
    marginBottom: 10, // Add margin for separation between rows
    backgroundColor: '#fff', // secondary color for row background
    borderRadius: 10, // Rounded corners for rows
    padding: 10, // Padding inside rows
  },
  fill: {
    flex: 1,
    marginHorizontal: 10, // Adjusted margin for consistency
    color: '#1423dc', // Text color for recording info
  },
  logo: {
    width: 200,
    height: 200,
    resizeMode: 'contain',
    marginTop: 10,
  },
  buttonContainer: {
    marginTop: 20,
    marginBottom: 20, // Add spacing between buttons and the rest
    width: '80%', // Specify the width to make the buttons the same size
    alignSelf: 'center', // Center the buttons in the container
  },
  actionButtonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between', // Ensures space between Play and Analyse buttons
    width: '35%', // Adjust the width as necessary to fit your design
  },
});


export default MainScreen;