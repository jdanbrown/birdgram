import _ from 'lodash';
import React, { RefObject, PureComponent } from 'react';
import {
  ActivityIndicator, Dimensions, FlatList, Image, Platform, SectionList, Text, TouchableWithoutFeedback, View, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';

import { matchQuery, Query } from 'app/components/SearchScreen';
import {
  matchRecordPathParams, matchSearchPathParams, recordPathParamsFromLocation, searchPathParamsFromLocation, Source,
} from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { debug_print, Log, rich, puts, tap } from 'app/log';
import {
  Go, Histories, History, Location, locationKeyIsEqual, locationPathIsEqual, locationStateOrEmpty, tabHistoriesKeys,
  TabName,
} from 'app/router';
import { Settings } from 'app/settings';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  enumerate, global, into, json, local, mapNil, mapNull, mapUndefined, match, matchNull, matchUndefined,
  mergeArraysWith, objectKeysTyped, pretty, shallowDiffPropsState, showDate, throw_, typed, yaml,
} from 'app/utils';
import { XC } from 'app/xc';

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
  status:       'ready' | 'loading';
  recents:      Array<Recent>;
  tabLocations: TabLocations; // Track as state to avoid having to .forceUpdate()
}

type TabLocations = {[key in CapturedTabName]: Location};

type Recent = RecordRecent | SearchRecent;
interface RecordRecent {
  tab:        'record';
  location:   Location;
  recordPath: RecordPath;
}
interface SearchRecent {
  tab:      'search';
  location: Location;
  query:    Query;
}

export function matchRecent<X>(recent: Recent, cases: {
  record: (recent: RecordRecent) => X,
  search: (recent: SearchRecent) => X,
}): X {
  switch(recent.tab) {
    case 'record': return cases.record(recent);
    case 'search': return cases.search(recent);
  }
}

const Recent = {

  isOpenInTab: (recent: Recent, tabLocations: TabLocations): boolean => {
    return Recent.locationIsEqual(recent.location, tabLocations[recent.tab]);
  },

  // Compare by key i/o path
  //  - Want to show the user _where_ in history.entries they are, e.g. to reflect forward/back operations
  locationIsEqual: (x?: Location, y?: Location): boolean => {
    return locationKeyIsEqual(x, y);
  },

  // null means source not found
  fromLocation: async (tab: CapturedTabName, location: Location): Promise<null | Recent> => {
    switch(tab) {
      case 'record': return await RecordRecent.fromLocation(location);
      case 'search': return await SearchRecent.fromLocation(location);
    }
  },

};

const RecordRecent = {
  // null means source not found
  fromLocation: async (location: Location): Promise<null | RecordRecent> => {
    return await matchRecordPathParams<Promise<null | RecordRecent>>(recordPathParamsFromLocation(location), {
      root: async () => {
        return {
          tab: 'record',
          location,
          recordPath: {kind: 'root'},
        };
      },
      edit: async ({sourceId}) => {
        const source = await Source.load(sourceId);
        if (source === null) return null; // Not found: source
        return {
          tab: 'record',
          location,
          recordPath: {kind: 'edit', source},
        };
      },
    });
  },
};

const SearchRecent = {
  // null means source (for query) not found
  fromLocation: async (location: Location): Promise<null | SearchRecent> => {
    const query = await Query.loadFromLocation(location);
    if (query === null) return null; // Not found: source (for query)
    return {
      tab: 'search',
      location,
      query,
    };
  },
};

// TODO Integrate into RecordScreen (like Query for SearchScreen)
//  - Should also subsume `editRecording: null | EditRecording` in RecordScreen.State
//  - But EditRecording is very rich and will need to remain separate from RecordPath
export type RecordPath = RecordPathRoot | RecordPathEdit;
export interface RecordPathRoot {kind: 'root'};
export interface RecordPathEdit {kind: 'edit', source: Source};
export function matchRecordPath<X>(recordPath: RecordPath, cases: {
  root: (recordPath: RecordPathRoot) => X,
  edit: (recordPath: RecordPathEdit) => X,
}): X {
  switch(recordPath.kind) {
    case 'root': return cases.root(recordPath);
    case 'edit': return cases.edit(recordPath);
  }
}

export class RecentScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state: State = {
    status:       'loading',
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
        recent => locationStateOrEmpty(recent.location.state).timestamp || new Date(),
        ...(await Promise.all(capturedTabs.map(async tab => (
          _.flatten(await Promise.all(
            this.props.histories[tab].entries
            .map(async location => matchNull(await Recent.fromLocation(tab, location), {
              x:    recent => [recent],
              null: ()     => {
                log.info(`histories[${tab}].entries: Source not found, ignoring`, rich(location));
                return [];
              },
            }))
          ))
          .slice().reverse() // Reverse without mutating
        )))),
      ),
    );
    capturedTabs.forEach(tab => {
      this.props.histories[tab].listen(async (location, action) => {
        // Add to history unless POP, which happens on history.go*, which happens from RecentScreen
        //  - Actions reference
        //    - PUSH:    history.push
        //    - REPLACE: history.replace
        //    - POP:     history.go*
        if (action !== 'POP') {
          matchNull(await Recent.fromLocation(tab, location), {
            null: ()     => log.info(`histories[${tab}].listen: Source not found, ignoring`, rich(location)),
            x:    recent => this.addRecents([recent]),
          })
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
      status:       'ready',
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

        {/* NOTE BaseButton b/c TouchableWithoutFeedback wouldn't trigger onPress during FlatList scroll animation */}
        <BaseButton onPress={() => mapNull(this.flatListRef.current, x => x.scrollToOffset({offset: 0}))}>
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
        </BaseButton>

        <View style={{
          flex: 1,
          // backgroundColor: iOSColors.customGray,
        }}>

          {/* Loading spinner */}
          {this.state.status === 'loading' && (
            <View style={{
              flex: 1,
              justifyContent: 'center',
            }}>
              <ActivityIndicator size='large' />
            </View>
          )}

          {/* TODO SectionList with dates (recent.timestamp) as section headers */}
          {this.state.status === 'ready' && (
            <FlatList <Recent>
              ref={this.flatListRef}
              style={{
                ...Styles.fill,
              }}
              contentInset={{
                top:    -1, // Hide top elem border under bottom border of title bar
                bottom: -1, // Hide bottom elem border under top border of tab bar
              }}
              initialNumToRender={20} // Enough to fill one screen (and not much more)
              data={this.state.recents}
              keyExtractor={(recent, index) => `${index}`}
              ListEmptyComponent={(
                <View style={[Styles.center, {padding: 30}]}>
                  <Text style={material.subheading}>
                    No history yet
                  </Text>
                </View>
              )}
              renderItem={({item: recent, index}) => (
                <RectButton
                  onPress={() => {
                    this.props.go(recent.tab, {
                      index: _.findIndex(
                        this.props.histories[recent.tab].entries,
                        x => Recent.locationIsEqual(x, recent.location),
                      )!,
                    });
                  }}
                >
                  <View style={{
                    flex: 1,
                    flexDirection: 'row',
                    // backgroundColor: iOSColors.white,
                    // Highlight active location per tab
                    backgroundColor: (Recent.isOpenInTab(recent, this.state.tabLocations)
                      ? `${capturedTabProps[recent.tab].color}22`
                      : undefined
                    ),
                    padding: 5,
                    // Bottom border on all items, top border on first item
                    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                    ...(index != 0 ? {} : {
                      borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: 'black',
                    }),
                    ...Styles.center,
                  }}>
                    <Feather style={{
                      ...material.titleObject,
                      // Highlight active location per tab
                      color: (Recent.isOpenInTab(recent, this.state.tabLocations)
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
                        {matchRecent(recent, {
                          record: recent => matchRecordPath(recent.recordPath, {
                            root: ()         => '[Root]',
                            edit: ({source}) => Source.show(source, {
                              species: this.props.xc,
                              long:    true, // e.g. 'User recording: ...' / 'XC recording: ...'
                            }),
                          }),
                          search: recent => matchQuery(recent.query, {
                            none: () => (
                              '[None]' // [Does this ever happen?]
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
                            rec: ({filters, source}) => (
                              Source.show(source, {
                                species: this.props.xc,
                                long:    true, // e.g. 'User recording: ...' / 'XC recording: ...'
                              })
                            ),
                          }),
                        })}

                      </Text>
                      <Text style={material.caption}>
                        {mapNull(locationStateOrEmpty(recent.location.state).timestamp, showDate)}
                      </Text>
                    </View>
                  </View>
                </RectButton>
              )}
            />
          )}

        </View>

      </View>
    );
  }

}

const styles = StyleSheet.create({
});
