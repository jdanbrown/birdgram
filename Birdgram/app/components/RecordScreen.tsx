import React, { PureComponent } from 'react';
import { Dimensions, Image, Platform, Text, View } from 'react-native';

import { log } from '../log';
import { Recorder } from './Recorder';
import { StyleSheet } from '../stylesheet';
import { shallowDiffPropsState } from '../utils';

interface Props {}
interface State {}

export class RecordScreen extends PureComponent<Props, State> {

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);
  }

  componentWillUnmount = async () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));
  }

  render = () => {
    return (
      <View style={styles.container}>

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
