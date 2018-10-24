import React from 'React';
import { Component } from 'react';
import { Dimensions, Image, Platform, Text, View } from 'react-native';
import KeepAwake from 'react-native-keep-awake';

import { Recorder } from './Recorder';
import { StyleSheet } from '../stylesheet';
import { global } from '../utils';

type Props = {};

export class RecordScreen extends Component<Props> {

  render = () => {
    return (
      <View style={styles.container}>

        {__DEV__ && <KeepAwake/>}

        <Recorder
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
