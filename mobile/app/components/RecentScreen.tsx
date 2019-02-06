import _ from 'lodash';
import React, { RefObject, PureComponent } from 'react';
import {
  ActivityIndicator, Dimensions, FlatList, Image, LayoutChangeEvent, Platform, SectionList, SectionListData, Text,
  TouchableWithoutFeedback, View, WebView,
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
  TabLocations, TabName,
} from 'app/router';
import { Settings } from 'app/settings';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  enumerate, global, into, json, local, mapNil, mapNull, mapUndefined, match, matchNull, matchUndefined,
  mergeArraysWith, objectKeysTyped, pretty, shallowDiffPropsState, showDate, showDateNoTime, showTime, throw_, typed,
  yaml,
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
  location:     Location;
  history:      History;
  histories:    Histories;
  go:           Go;
  tabLocations: TabLocations<CapturedTabName>;
  xc:           XC;
  ebird:        Ebird;
  // Settings
  showDebug:    boolean;
  maxHistory:   number;
  // RecentScreen
  iconForTab:   {[key in CapturedTabName]: string};
}

interface State {
  status:  'ready' | 'loading';
  recents: Array<Recent>;
}

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

  isOpenInTab: (recent: Recent, tabLocations: TabLocations<CapturedTabName>): boolean => {
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
    status:  'loading',
    recents: [],
  };

  // Refs
  sectionListRef: RefObject<SectionList<Recent>> = React.createRef();

  // State
  _firstSectionHeaderHeight: number = 0; // For SectionList.scrollToLocation({viewOffset})

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
      });
    });

    this.setState((state, props) => ({
      status: 'ready',
    }));

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

    // Precompute sections so we can figure out various indexes
    type Section = SectionListData<Recent>;
    const sections: Array<Section> = (
      _(this.state.recents)
      .groupBy(recent => showDateNoTime(_.get(recent.location.state, 'timestamp',
        new Date('3000'), // Put weird/unexpected stuff at the top so it's visible
      )))
      .entries().map(([title, data]) => ({title, data}))
      .value()
    );
    const firstSection   = _.head(sections);
    const lastSection    = _.last(sections);
    const isFirstSection = (section: Section) => firstSection && section.title === firstSection.title;
    const isLastSection  = (section: Section) => lastSection  && section.title === lastSection.title;
    const isLastItem     = (section: Section, index: number) => isLastSection(section) && index === section.data.length - 1;

    return (
      <View style={{
        flex: 1,
      }}>

        {/* NOTE BaseButton b/c TouchableWithoutFeedback wouldn't trigger onPress during FlatList scroll animation */}
        <BaseButton onPress={() => {
          mapNull(this.sectionListRef.current, sectionList => { // Avoid transient nulls [why do they happen?]
            if (sectionList.scrollToLocation) { // (Why typed as undefined? I think only for old versions of react-native?)
              sectionList.scrollToLocation({
                sectionIndex: 0, itemIndex: 0,              // First section, first item
                viewOffset: this._firstSectionHeaderHeight, // Else first item covered by first section header
              });
            }
          });
        }}>
          <View style={{
            backgroundColor:   Styles.tabBar.backgroundColor,
            borderBottomWidth: Styles.tabBar.borderTopWidth,
            borderBottomColor: Styles.tabBar.borderTopColor,
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
            <SectionList
              ref={this.sectionListRef as any} // HACK Is typing for SectionList busted? Can't make it work
              style={{
                ...Styles.fill,
              }}
              contentInset={{
                top:    -1, // Hide top elem border under bottom border of title bar
                bottom: -1, // Hide bottom elem border under top border of tab bar
              }}
              initialNumToRender={20} // Enough to fill one screen (and not much more)
              sections={sections}
              keyExtractor={(recent, index) => `${index}`}
              ListEmptyComponent={(
                <View style={[Styles.center, {padding: 30}]}>
                  <Text style={material.subheading}>
                    No history yet
                  </Text>
                </View>
              )}
              renderSectionHeader={({section}) => (
                <View
                  style={[Styles.fill, {
                    backgroundColor:   iOSColors.lightGray,
                    paddingHorizontal: 5,
                    paddingTop:        2,
                    paddingBottom:     2,
                  }]}
                  // For SectionList.scrollToLocation({viewOffset})
                  onLayout={!isFirstSection(section) ? undefined : this.onFirstSectionHeaderLayout}
                >
                  <Text style={{
                    fontWeight: 'bold',
                    color:      '#444444',
                  }}>{section.title}</Text>
                </View>
              )}
              renderItem={({item: recent, index, section}: {item: Recent, index: number, section: Section}) => (
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
                    ...Styles.center,
                    padding: 5,
                    // Highlight active location per tab
                    backgroundColor: (Recent.isOpenInTab(recent, this.props.tabLocations)
                      ? `${capturedTabProps[recent.tab].color}22`
                      : undefined
                    ),
                    // Vertical borders
                    //  - Internal borders: top border on non-first items per section
                    //  - Plus bottom border on last item of last section
                    borderTopWidth: 1,
                    borderTopColor: iOSColors.lightGray,
                    ...(!isLastItem(section, index) ? {} : {
                      borderBottomWidth: 1,
                      borderBottomColor: iOSColors.lightGray,
                    }),
                  }}>
                    <Feather style={{
                      ...material.titleObject,
                      // Highlight active location per tab
                      color: (Recent.isOpenInTab(recent, this.props.tabLocations)
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
                              species:  this.props.xc,
                              long:     true, // e.g. 'User recording: ...' / 'XC recording: ...'
                              showDate: x => showDate(x).replace(`${section.title} `, ''), // HACK showTime if date = section date
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
                                species:  this.props.xc,
                                long:     true, // e.g. 'User recording: ...' / 'XC recording: ...'
                                showDate: x => showDate(x).replace(`${section.title} `, ''), // HACK showTime if date = section date
                              })
                            ),
                          }),
                        })}

                      </Text>
                      <Text style={material.caption}>
                        {mapNull(locationStateOrEmpty(recent.location.state).timestamp, showTime)}
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

  onFirstSectionHeaderLayout = async (event: LayoutChangeEvent) => {
    const {nativeEvent: {layout: {x, y, width, height}}} = event; // Unpack SyntheticEvent (before async)
    this._firstSectionHeaderHeight = height;
  }

}

const styles = StyleSheet.create({
});
