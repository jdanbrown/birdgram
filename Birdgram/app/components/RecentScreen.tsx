import { Location, MemoryHistory } from 'history';
import React, { PureComponent } from 'react';
import { Dimensions, FlatList, Image, Platform, SectionList, Text, View, WebView } from 'react-native';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';

import { log } from '../log';
import { Go } from '../router';
import { Settings } from '../settings';
import { StyleSheet } from '../stylesheet';
import { global, json, pretty, shallowDiffPropsState, Styles } from '../utils';

interface Props {
  settings:   Settings;
  location:   Location;
  history:    MemoryHistory;
  histories:  {[key: string]: MemoryHistory};
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
    this.addLocations(this.props.histories.search.entries.reverse());

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

        <View style={{
          width: '100%',
          backgroundColor: '#f7f7f7', // Default background color in iOS 10
          borderBottomWidth: StyleSheet.hairlineWidth,
          borderBottomColor: 'rgba(0,0,0,.3)',
          paddingVertical: 5, paddingHorizontal: 10,
        }}>
          <Text style={{
            ...material.headlineObject,
          }}>
            Recent searches
          </Text>
        </View>

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
                padding: 5,
                borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
              }}>
                <Text>{recent.timestamp.toLocaleString()}</Text>
                <Text>{recent.location.pathname}</Text>
              </View>
            </RectButton>
          )}
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
    textAlign: 'left',
    margin: 10,
  },
});
