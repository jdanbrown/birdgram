import React, { PureComponent } from 'react';
import { Dimensions, Image, Platform, Text, View, WebView } from 'react-native';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { Log, rich } from '../log';
import { StyleSheet } from '../stylesheet';
import { global, json, shallowDiffPropsState, yaml } from '../utils';

const log = new Log('SavedScreen');

interface Props {
}
interface State {
}

export class SavedScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state = {
  };

  componentDidMount = async () => {
    log.info('componentDidMount');
  }

  componentWillUnmount = async () => {
    log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  render = () => {
    log.info('render');
    return (
      <View style={{
        flex: 1,
      }}>

        <View style={{
          borderBottomWidth: 1,
          borderColor: iOSColors.midGray,
        }}>
          <Text style={{
            alignSelf: 'center',
            marginTop: 30 - getStatusBarHeight(), // No status bar
            marginBottom: 10,
            ...material.titleObject,
          }}>
            Saved Lists
          </Text>
        </View>

      </View>
    );
  }
}

const styles = StyleSheet.create({
});
