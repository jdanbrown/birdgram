import _ from 'lodash';
import React, { RefObject, PureComponent } from 'react';
import {
  Dimensions, FlatList, Image, Platform, SectionList, Text, TouchableWithoutFeedback, View, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';

import {
  matchRecordPathParams, matchSearchPathParams, recordPathParamsFromLocation, searchPathParamsFromLocation, SourceId,
} from '../datatypes';
import { Ebird } from '../ebird';
import { Log, rich } from '../log';
import { Go, Histories, History, Location, tabHistoriesKeys, TabName } from '../router';
import { Settings } from '../settings';
import { Styles } from '../styles';
import { StyleSheet } from '../stylesheet';
import {
  enumerate, global, into, json, local, mapNil, mapNull, mapUndefined, match, matchUndefined, mergeArraysWith,
  objectKeysTyped, pretty, shallowDiffPropsState, showDate, throw_, yaml,
} from '../utils';
import { XC } from '../xc';

const log = new Log('RecentScreen');

const capturedTabProps = {
  record: {
    color: iOSColors.pink,
  },
  search: {
    color: iOSColors.blue,
  },
};
type CapturedTabName = keyof typeof capturedTabProps;
const capturedTabs: Array<CapturedTabName> = [
  'record',
  'search',
];

interface Props {
  // App globals
  location:   Location;
  history:    History;
  histories:  Histories;
  go:         Go;
  xc:         XC;
  ebird:      Ebird;
  // Settings
  showDebug:  boolean;
  maxHistory: number;
  // RecentScreen
  iconForTab: {[key in CapturedTabName]: string};
}

interface State {
  recents:      Array<Recent>;
  tabLocations: {[key in CapturedTabName]: Location}, // Track as state to avoid having to .forceUpdate()
}

interface Recent {
  tab:      CapturedTabName;
  location: Location;
}

export class RecentScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state = {
    recents:      [],
    tabLocations: _.mapValues(this.props.histories, x => x.location),
  };

  // Refs
  flatListRef: RefObject<FlatList<Recent>> = React.createRef();

  componentDidMount = async () => {
    log.info('componentDidMount');
    global.RecentScreen = this; // XXX Debug

    // Capture all locations from histories.{record,search}
    //  - Listen for location changes (future)
    //  - Capture existing history (past)
    //  - TODO How to avoid races?
    this.addRecents(
      mergeArraysWith( // [TODO Why did I mergeArraysWith i/o just _.sortBy?]
        recent => recent.location.state.timestamp,
        ...capturedTabs.map(tab =>
          this.props.histories[tab].entries
          .map(location => ({tab, location}))
          .slice().reverse() // Reverse without mutating
        ),
      ),
    );
    capturedTabs.forEach(tab => {
      this.props.histories[tab].listen((location, action) => {
        // Add to history unless POP, which happens on history.go*, which happens from RecentScreen
        //  - Actions reference
        //    - PUSH:    history.push
        //    - REPLACE: history.replace
        //    - POP:     history.go*
        if (action !== 'POP') {
          this.addRecents([{
            tab,
            location,
          }]);
        }
        // Update tabLocations on each route change
        this.setTabLocations();
      });
    });

    // Update tabLocations once after adding listeners in case any changed in the interim
    this.setTabLocations();

  }

  addRecents = (recents: Array<Recent>) => {
    this.setState((state, props) => ({
      recents: [...recents, ...state.recents].slice(0, this.props.maxHistory), // Most recent first (reverse of history.entries)
    }));
  }

  setTabLocations = () => {
    this.setState((state, props) => ({
      tabLocations: _.mapValues(props.histories, x => x.location),
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

        <TouchableWithoutFeedback onPress={() => mapNull(this.flatListRef.current, x => x.scrollToOffset({offset: 0}))}>
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
              History
            </Text>
          </View>
        </TouchableWithoutFeedback>

        <View style={{
          flex: 1,
          // backgroundColor: iOSColors.customGray,
        }}>

          {/* TODO SectionList with dates (recent.timestamp) as section headers */}
          <FlatList <Recent>
            ref={this.flatListRef}
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
            renderItem={({item: recent}) => (
              <RectButton
                onPress={() => {
                  this.props.go(recent.tab, {
                    index: _.findIndex(
                      this.props.histories[recent.tab].entries,
                      x => x.key === recent.location.key,
                    )!,
                  });
                }}
              >
                <View style={{
                  flex: 1,
                  flexDirection: 'row',
                  // backgroundColor: iOSColors.white,
                  // Highlight active location per tab
                  backgroundColor: (recent.location.key === this.state.tabLocations[recent.tab].key
                    ? `${capturedTabProps[recent.tab].color}22`
                    : undefined
                  ),
                  padding: 5,
                  borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                  ...Styles.center,
                }}>
                  <Feather style={{
                    ...material.titleObject,
                    // Highlight active location per tab
                    color: (recent.location.key === this.state.tabLocations[recent.tab].key
                      ? capturedTabProps[recent.tab].color
                      : iOSColors.gray
                    )
                  }}
                    name={this.props.iconForTab[recent.tab]}
                  />
                  <View style={{
                    flex: 1,
                    flexDirection: 'column',
                    paddingLeft: 5,
                  }}>
                    <Text style={{
                      ...material.body1Object,
                    }}>

                      {/* TODO Dedupe with SavedScreen.render */}
                      {match<CapturedTabName, string>(recent.tab,
                        ['record', () => matchRecordPathParams(recordPathParamsFromLocation(recent.location), {
                          root: () => (
                            '[ROOT]' // TODO When does this show? How should it display?
                          ),
                          edit: ({sourceId}) => (
                            SourceId.show(sourceId, {
                              species: this.props.xc,
                              long:    true, // e.g. 'User recording: ...' / 'XC recording: ...'
                            })
                          ),
                        })],
                        ['search', () => matchSearchPathParams(searchPathParamsFromLocation(recent.location), {
                          root: () => (
                            '[ROOT]' // Shouldn't ever show b/c redirect
                          ),
                          random: ({filters, seed}) => (
                            `Random`
                          ),
                          species: ({filters, species}) => (
                            species === '_BLANK' ? '[BLANK]' :
                            matchUndefined(this.props.ebird.speciesMetadataFromSpecies.get(species), {
                              undefined: () => `${species} (?)`,
                              x:         x  => `${species} (${x.com_name})`,
                            })
                          ),
                          rec: ({filters, sourceId}) => (
                            SourceId.show(sourceId, {
                              species: this.props.xc,
                              long:    true, // e.g. 'User recording: ...' / 'XC recording: ...'
                            })
                          ),
                        })],
                      )}

                    </Text>
                    <Text style={material.caption}>
                      {showDate(recent.location.state.timestamp)}
                    </Text>
                  </View>
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
