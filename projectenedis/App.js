import React from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import { Audio } from 'expo-av';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import { LineChart } from "react-native-gifted-charts";

import MainScreen from './components/MainScreen';
import AnalyseScreen from './components/AnalyseScreen';


const Stack = createNativeStackNavigator();

const App = () => {
  const data = [{value: 15}, {value: 30}, {value: 26}, {value: 40}];

  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen
          name="Main"
          component={MainScreen}
          LineChart data={data}
        />
        <Stack.Screen name="Analysis" component={AnalyseScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;