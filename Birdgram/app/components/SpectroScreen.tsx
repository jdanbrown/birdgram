import React from 'React';
import { Component } from 'react';
import { Dimensions, Image, Platform, StyleSheet, Text, View } from 'react-native';
import KeepAwake from 'react-native-keep-awake';

import { SpectroRecorder } from './SpectroRecorder';
import { global } from '../utils';

type Props = {};

export class SpectroScreen extends Component<Props> {

  render() {
    return (
      <View style={styles.container}>

        {__DEV__ && <KeepAwake/>}

        <SpectroRecorder
          sampleRate={22050}
          refreshRate={4}
          spectroHeight={400}
        />

      </View>
    );
  }

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  banner: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});
