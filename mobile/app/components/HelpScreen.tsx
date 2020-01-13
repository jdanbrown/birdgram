import React, { PureComponent } from 'react';
import { Dimensions, Image, Platform, ScrollView, Text, View } from 'react-native';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { TitleBar } from 'app/components/TitleBar';
import { Log, logErrors, logErrorsAsync, rich } from 'app/log';
import { Go, Histories, History, Location } from 'app/router';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import { global, json, shallowDiffPropsState, yaml } from 'app/utils';

const log = new Log('HelpScreen');

interface Props {
  // App globals
  location:   Location;
  history:    History;
  histories:  Histories;
  go:         Go;
}

interface State {
}

export class HelpScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state = {
  };

  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
    log.info('componentDidMount');
  });

  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    log.info('componentWillUnmount');
  });

  componentDidUpdate = async (prevProps: Props, prevState: State) => logErrorsAsync('componentDidUpdate', async () => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  });

  render = () => {
    log.info('render');
    return (
      <View style={{
        flex: 1,
      }}>

        <TitleBar
          title='Help'
        />

        <ScrollView style={{
          flex: 1,
          // backgroundColor: iOSColors.customGray,
        }}>

          <View style={{
            borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
          }}>
            <Text style={material.title}>
              Overview
            </Text>
            <Text>
              ...
            </Text>
          </View>

          <View style={{
            borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
          }}>
            <Text style={material.title}>
              Record tab
            </Text>
            <Text>
              ...
            </Text>
          </View>

          <View style={{
            borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
          }}>
            <Text style={material.title}>
              Search tab
            </Text>
            <Text>
              ...
            </Text>
          </View>

          <View style={{
            borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
          }}>
            <Text style={material.title}>
              Recents tab
            </Text>
            <Text>
              ...
            </Text>
          </View>

          <View style={{
            borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
          }}>
            <Text style={material.title}>
              Saved tab
            </Text>
            <Text>
              ...
            </Text>
          </View>

        </ScrollView>

      </View>
    );
  }
}

const styles = StyleSheet.create({
});
