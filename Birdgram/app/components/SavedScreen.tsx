import React, { PureComponent } from 'react';
import { Dimensions, Image, Platform, Text, View, WebView } from 'react-native';

import { log } from '../log';
import { StyleSheet } from '../stylesheet';
import { global, shallowDiffPropsState } from '../utils';

interface Props {}
interface State {}

export class SavedScreen extends PureComponent<Props, State> {

  constructor(props: Props) {
    super(props);
    log.info(`${this.constructor.name}.constructor`);
  }

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

        <Text style={styles.banner}>
          Saved
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
