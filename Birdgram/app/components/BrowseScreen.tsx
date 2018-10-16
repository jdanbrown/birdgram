import React from 'React';
import { Component } from 'react';
import { Dimensions, Image, Platform, StyleSheet, Text, View, WebView } from 'react-native';
import KeepAwake from 'react-native-keep-awake';

import { global } from '../utils';

type Props = {};

export class BrowseScreen extends Component<Props> {

  render() {
    return (
      <View style={styles.container}>

        {__DEV__ && <KeepAwake/>}

        <Text style={styles.banner}>
          Browse
        </Text>

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
