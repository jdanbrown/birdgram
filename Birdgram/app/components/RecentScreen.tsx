import { Location, MemoryHistory } from 'history';
import React, { PureComponent } from 'react';
import { Dimensions, FlatList, Image, Platform, SectionList, Text, View, WebView } from 'react-native';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';

import { log } from '../log';
import { Go, Histories } from '../router';
import { Settings } from '../settings';
import { StyleSheet } from '../stylesheet';
import { global, json, pretty, shallowDiffPropsState, Styles } from '../utils';

interface Props {
  settings:   Settings;
  location:   Location;
  history:    MemoryHistory;
  histories:  Histories;
  go:         Go;
  maxRecents: number;
}

interface State {
  recents: Array<Recent>;
}

interface Recent {
  location: Location;
  timestamp: Date;
}

export class RecentScreen extends PureComponent<Props, State> {

  static defaultProps = {
    maxRecents: 1000,
  };

  state = {
    recents: [],
  };

  addLocations = (locations: Array<Location>) => {
    this.addRecents(locations.map(location => ({
      location,
      timestamp: new Date(),
    })));
  }

  addRecents = (recents: Array<Recent>) => {
    this.setState((state, props) => ({
      recents: [...recents, ...state.recents].slice(0, this.props.maxRecents),
    }));
  }

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);
    global.RecentScreen = this; // XXX Debug

    // Capture all locations from histories.search
    //  - Listen for location changes (future)
    //  - Capture existing history (past)
    //  - TODO How to avoid races?
    this.props.histories.search.listen((location, action) => {
      this.addLocations([location]);
    });
    this.addLocations(this.props.histories.search.entries
      .slice().reverse() // Reverse without mutating
    );

  }

  componentWillUnmount = async () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));
  }

  render = () => (
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
          Recent searches
        </Text>
      </View>

      <View style={{
        flex: 1,
        // backgroundColor: iOSColors.customGray,
      }}>

        {/* TODO SectionList with dates as section headers */}
        <FlatList <Recent>
          style={{
            ...Styles.fill,
          }}
          data={this.state.recents}
          keyExtractor={(recent, index) => `${index}`}
          ListHeaderComponent={(
            // Simulate top border on first item
            <View style={{
              height: 0,
              borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
            }}/>
          )}
          renderItem={({item: recent, index}) => (
            <RectButton
              onPress={() => this.props.go('search', recent.location.pathname)}
            >
              <View style={{
                flex: 1,
                flexDirection: 'column',
                backgroundColor: iOSColors.white,
                padding: 5,
                borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
              }}>
                <Text style={material.body1}>
                  {recent.location.pathname}
                </Text>
                <Text style={material.caption}>
                  {recent.timestamp.toDateString()}, {recent.timestamp.toLocaleTimeString()}
                </Text>
              </View>
            </RectButton>
          )}
        />

      </View>

    </View>
  );

}

const styles = StyleSheet.create({
});
