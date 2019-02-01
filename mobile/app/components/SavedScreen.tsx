import _ from 'lodash';
import React, { RefObject, PureComponent } from 'react';
import {
  ActivityIndicator, Dimensions, FlatList, Image, Platform, Text, TouchableWithoutFeedback, View, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';

import { matchQuery, Query } from 'app/components/SearchScreen';
import {
  matchRecordPathParams, matchSearchPathParams, matchSource, recordPathParamsFromLocation, Rec,
  searchPathParamsFromLocation, Source, SourceId, UserRec, XCRec,
} from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { debug_print, Log, puts, rich } from 'app/log';
import { Go, Location, locationKeyIsEqual, locationPathIsEqual, TabLocations, TabName } from 'app/router';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  global, into, json, local, mapNil, mapNull, mapUndefined, match, matchNil, matchNull, matchUndefined, mergeArraysWith,
  pretty, shallowDiffPropsState, showDate, throw_, typed, yaml,
} from 'app/utils';
import { XC } from 'app/xc';

const log = new Log('SavedScreen');

const tabProps = {
  record: {
    color: iOSColors.pink,
  },
  search: {
    color: iOSColors.blue,
  },
};

interface Props {
  // App globals
  go:           Go;
  tabLocations: TabLocations;
  xc:           XC;
  ebird:        Ebird;
  // RecentScreen
  iconForTab:   {[key in TabName]: string};
}

interface State {
  status: 'loading' | 'ready';
  saveds: Array<Saved>;
}

type Saved = RecordSaved | SearchSaved;
interface RecordSaved {
  tab:      'record';
  location: Location;
  source:   Source;
}
interface SearchSaved { // TODO Unused; ready and waiting for saves from SearchScreen
  tab:      'search';
  location: Location;
  query:    Query;
}

export function matchSaved<X>(saved: Saved, cases: {
  record: (saved: RecordSaved) => X,
  search: (saved: SearchSaved) => X,
}): X {
  switch(saved.tab) {
    case 'record': return cases.record(saved);
    case 'search': return cases.search(saved);
  }
}

const Saved = {

  isOpenInTab: (saved: Saved, tabLocations: TabLocations): boolean => {
    return Saved.locationIsEqual(saved.location, tabLocations[saved.tab]);
  },

  // Compare by path i/o key
  //  - Saved.location doesn't always have .key, and unlike RecentScreen we only have each path once
  locationIsEqual: (x?: Location, y?: Location): boolean => {
    return locationPathIsEqual(x, y);
  },

};

export class SavedScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state: State = {
    status: 'loading',
    saveds: [],
  };

  // Refs
  flatListRef: RefObject<FlatList<Saved>> = React.createRef();

  componentDidMount = async () => {
    log.info('componentDidMount');
    global.SavedScreen = this; // XXX Debug

    // Reload saveds when a new user rec is created
    ['user'].forEach(k => Rec.emitter.addListener(k, async (source: Source) => {
      log.info('Rec.emitter.listener', {source});
      this.addSaved(this.recordSavedFromSource(source));
    }));

    // Load initial saveds from user recs on fs
    //  - TODO Load SearchSaved's from somewhere too
    await this.loadSavedsFromFs();

  }

  loadSavedsFromFs = async () => {
    log.info('loadSavedsFromFs');

    // Load user recs
    //  - Current representation of "saved" is all user recs in the fs
    //  - TODO Add a delete/unsave button so user can clean up unwanted recs
    //  - TODO(cache_user_metadata): Perf: limit num results to avoid unbounded readMetadata operations
    const userRecSources = await UserRec.listAudioSources();

    // Order saveds
    const saveds = (
      _<Source[][]>([
        userRecSources,
      ])
      .flatten()
      .sortBy(x => matchSource(x, {
        xc:   source => throw_(`Impossible xc case: ${x}`),
        user: source => source.metadata.created,
      }))
      .value()
      .slice().reverse() // (Copy b/c reverse mutates)
      .map<RecordSaved>(source => this.recordSavedFromSource(source))
    );

    this.setState({
      status: 'ready',
      saveds,
    });

  }

  addSaved = (saved: Saved) => {
    this.setState((state, props) => ({
      saveds: [saved, ...state.saveds], // Most recent first
    }));
  }

  recordSavedFromSource = (source: Source): RecordSaved => {
    return {
      tab: 'record',
      location: {
        pathname: `/edit/${encodeURIComponent(Source.stringify(source))}`,
        // HACK Dummy fields, we only ever use location.pathname (in onPress -> this.props.go)
        search:   '',
        hash:     '',
        key:      undefined,
        state:    {timestamp: new Date(0)},
      },
      source,
    };
  }

  // TODO
  // searchSavedFromSource = (source: Source): SearchSaved => {
  //   ...
  // }

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
              Saved
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

          {/* TODO SectionList with dates [of what?] as section headers */}
          {this.state.status === 'ready' && (
            <FlatList
              ref={this.flatListRef}
              style={{
                ...Styles.fill,
              }}
              contentInset={{
                top:    -1, // Hide top elem border under bottom border of title bar
                bottom: -1, // Hide bottom elem border under top border of tab bar
              }}
              initialNumToRender={20} // Enough to fill one screen (and not much more)
              data={typed<Saved[]>(this.state.saveds)}
              keyExtractor={(saved, index) => `${index}`}
              ListEmptyComponent={(
                <View style={[Styles.center, {padding: 30}]}>
                  <Text style={material.subheading}>
                    No saved items
                  </Text>
                </View>
              )}
              renderItem={({item: saved, index}) => (
                <RectButton
                  onPress={() => {
                    this.props.go(saved.tab, {path: saved.location.pathname});
                  }}
                >
                  <View style={{
                    flex: 1,
                    flexDirection: 'row',
                    padding: 5,
                    // Highlight active location per tab
                    backgroundColor: (Saved.isOpenInTab(saved, this.props.tabLocations)
                      ? `${tabProps[saved.tab].color}22`
                      : undefined
                    ),
                    // Bottom border on all items, top border on first item
                    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                    ...(index != 0 ? {} : {
                      borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: 'black',
                    }),
                    ...Styles.center,
                  }}>
                    <Feather style={{
                      ...material.titleObject,
                      color: iOSColors.gray,
                    }}
                      name={this.props.iconForTab[saved.tab]}
                    />
                    <View style={{
                      flex: 1,
                      flexDirection: 'column',
                      paddingLeft: 5,
                    }}>
                      <Text style={{
                        ...material.body1Object,
                      }}>

                        {/* TODO Dedupe with RecentScreen.render */}
                        {matchSaved(saved, {
                          record: saved => Source.show(saved.source, {
                            species: this.props.xc,
                            long:    true, // e.g. 'User recording: ...' / 'XC recording: ...'
                          }),
                          search: saved => matchQuery(saved.query, {
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
