import { Location, MemoryHistory } from 'history';
import React, { PureComponent } from 'react';
import { Dimensions, FlatList, Image, Platform, SectionList, Text, View, WebView } from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { matchSearchPathParams, searchPathParamsFromLocation, showSourceId } from '../datatypes';
import { Log, rich } from '../log';
import { Go, Histories } from '../router';
import { Settings } from '../settings';
import { Styles } from '../styles';
import { StyleSheet } from '../stylesheet';
import { global, json, pretty, shallowDiffPropsState } from '../utils';

const log = new Log('RecentScreen');

interface Props {
  // App globals
  location:   Location;
  history:    MemoryHistory;
  histories:  Histories;
  go:         Go;
  // Settings
  showDebug:  boolean;
  maxHistory: number;
}

interface State {
  recents: Array<Recent>;
  locationIndex: number;
}

interface Recent {
  location: Location;
  timestamp: Date;
}

export class RecentScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state = {
    recents: [],
    locationIndex: this.props.histories.search.length - 1 - this.props.histories.search.index,
  };

  componentDidMount = async () => {
    log.info('componentDidMount');
    global.RecentScreen = this; // XXX Debug

    // Capture all locations from histories.search
    //  - Listen for location changes (future)
    //  - Capture existing history (past)
    //  - TODO How to avoid races?
    this.addLocations(this.props.histories.search.entries
      .slice().reverse() // Reverse without mutating
    );
    this.props.histories.search.listen((location, action) => {
      // Add to history unless POP, which happens on history.go*, which happens from RecentScreen
      //  - Actions reference
      //    - PUSH:    history.push
      //    - REPLACE: history.replace
      //    - POP:     history.go*
      if (action !== 'POP') {
        this.addLocations([location]);
      }
      this.setState({
        locationIndex: this.props.histories.search.length - 1 - this.props.histories.search.index,
      });
    });

  }

  addLocations = (locations: Array<Location>) => {
    this.addRecents(locations.map(location => ({
      location,
      timestamp: new Date(),
    })));
  }

  addRecents = (recents: Array<Recent>) => {
    this.setState((state, props) => ({
      recents: [...recents, ...state.recents].slice(0, this.props.maxHistory), // Most recent first (reverse of history.entries)
    }));
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
            Recent Searches
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
            contentInset={{
              top:    -1, // Hide top elem border under bottom border of title bar
              bottom: -1, // Hide bottom elem border under top border of tab bar
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
                onPress={() => {
                  this.props.go('search', {index})
                }}
              >
                <View style={{
                  flex: 1,
                  flexDirection: 'column',
                  backgroundColor: iOSColors.white,
                  padding: 5,
                  borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                }}>
                  <Text style={{
                    ...material.body1Object,
                    // Highlight active location
                    ...(index !== this.state.locationIndex ? {} : {
                      color: iOSColors.blue,
                    }),
                  }}>
                    {matchSearchPathParams(searchPathParamsFromLocation(recent.location), {
                      root:    ()                    => 'Home (/)',
                      random:  ({filters, seed})     => `Random (${seed})`,
                      species: ({filters, species})  => `Species: ${species}`,
                      rec:     ({filters, sourceId}) => `Rec: ${showSourceId(sourceId)}`,
                    })}
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

}

const styles = StyleSheet.create({
});
