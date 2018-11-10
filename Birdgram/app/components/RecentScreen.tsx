import React from 'React';
import { Component } from 'react';
import { Dimensions, Image, Platform, Text, View, WebView } from 'react-native';

import { log } from '../log';
import { StyleSheet } from '../stylesheet';
import { global } from '../utils';

type Props = {};

export class RecentScreen extends Component<Props> {

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);
  }

  componentWillUnmount = async () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
  }

  render = () => {
    return (
      <View style={styles.container}>

        <Text style={styles.banner}>
          Recent
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
