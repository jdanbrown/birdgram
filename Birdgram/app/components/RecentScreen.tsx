import { Location, MemoryHistory } from 'history';
import React, { PureComponent } from 'react';
import { Dimensions, Image, Platform, Text, View, WebView } from 'react-native';

import { log } from '../log';
import { Settings } from '../settings';
import { StyleSheet } from '../stylesheet';
import { global, json, pretty, shallowDiffPropsState } from '../utils';

interface Props {
  settings: Settings;
  location: Location;
  history:  MemoryHistory;
}

interface State {
}

export class RecentScreen extends PureComponent<Props, State> {

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
          Recent
        </Text>

        <Text style={styles.banner}>
          {/* TODO(nav_router) */}
          {/* {pretty(this.props.history)} */}
          ({this.props.history.entries.length} entries)
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
