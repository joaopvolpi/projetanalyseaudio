import React from 'react';
import { StyleSheet, Text, ScrollView, View, Image, Button, Alert } from 'react-native';

const AnalyseScreen = ({ route }) => {
  // Extracting FFT and time images from route params
  const { fftImage, timeImage, audio } = route.params;

  // Function to handle button press
  const handlePress = () => Alert.alert("Not implemented");

  async function uploadAudio(audio) {
    const apiUrl = 'http://10.0.2.2:5000/upload-audio';
    const fileType = 'audio/3gp'; 
    const fileName = 'filename.3gp';
    const formData = new FormData();
    
    formData.append('audio', {
        name: fileName,
        type: fileType,
        uri: audio,
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
        Alert.alert("Success!");
      } else {
          console.error(result.message);
      }
  } catch (error) {
      console.error(error);
  }
}

  return (
    <ScrollView style={styles.scrollView} contentContainerStyle={styles.container}>
      <View style={styles.contentContainer}>
        <Image source={require('../assets/Logo_enedis.png')} style={styles.logo} />
        <Text style={styles.imageLabel}>Frequency Analysis:</Text>
        <Image source={{ uri: `data:image/png;base64,${fftImage}` }} style={styles.image} />
        <Text style={styles.imageLabel}>Audio in time:</Text>
        <Image source={{ uri: `data:image/png;base64,${timeImage}` }} style={styles.image} />
        <View style={styles.buttonContainer}>
          <Button title="Upload Audio" onPress={() => uploadAudio(audio)} color="#1423dc" />
        </View>
        <View style={styles.buttonContainer}>
          <Button title="AI Analyse" onPress={handlePress} color={styles.button.color} />
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
    backgroundColor: '#e9f3fb', // light_blue_enedis background for the entire scroll view
  },
  contentContainer: {
    alignItems: 'center',
    paddingVertical: 10, // Reduced padding at the top and bottom
  },
  logo: {
    width: 200,
    height: 200,
    resizeMode: 'contain',
    marginTop: 10,
    marginBottom: 10, // Reduced margin below the logo for tighter spacing
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1423dc', // blue_enedis
  },
  imageLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#41a57d', // green_enedis
  },
  image: {
    width: 300, // Image size remains the same
    height: 300,
    resizeMode: 'contain',

  },
  buttonContainer: {
    marginTop: 10, // Reduced top margin
    marginBottom: 20, // Slightly reduced bottom margin
    width: '60%', // Width remains the same
  },
  button: {
    color: '#41a57d', // Color remains the same
  },
});

export default AnalyseScreen;
