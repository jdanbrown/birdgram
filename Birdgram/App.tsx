import React from 'react'
import {Component} from 'react';
import {Platform, StyleSheet, Text, View} from 'react-native';

import {Hello} from './components/Hello';
import {Recorder} from './components/Recorder';

const instructions = Platform.select({
  ios:     'Press Cmd+R to reload,\nCmd+D or shake for dev menu',
  android: 'Double tap R on your keyboard to reload,\nShake or press menu button for dev menu',
});

type Props = {};

export default class App extends Component<Props> {
  render() {
    return (
      <View style={styles.container}>
        <Text style={styles.welcome}>Birdgram</Text>
        <Text style={styles.instructions}>{instructions}</Text>
        {/* <Hello name={'person'} enthusiasmLevel={3} /> */}
        <Recorder/>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  welcome: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  instructions: {
    textAlign: 'center',
    color: '#333333',
    marginBottom: 5,
  },
});
