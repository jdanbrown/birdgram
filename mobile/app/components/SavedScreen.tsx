import _ from 'lodash';
import moment from 'moment';
import React, { Component, RefObject, PureComponent } from 'react';
import {
  ActivityIndicator, Alert, AlertIOS, Dimensions, FlatList, Image, LayoutChangeEvent, Platform, ScrollView, SectionList,
  SectionListData, Text, TouchableWithoutFeedback, View, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { matchQuery, Query } from 'app/components/SearchScreen';
import {
  matchRecordPathParams, matchSearchPathParams, matchSource, recordPathParamsFromLocation, Rec,
  searchPathParamsFromLocation, Source, SourceId, UserRec, UserSource, XCRec,
} from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { debug_print, Log, puts, rich } from 'app/log';
import { Go, Location, locationKeyIsEqual, locationPathIsEqual, TabLocations, TabName } from 'app/router';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  global, ifEmpty, ifNull, into, json, local, mapNil, mapNull, mapUndefined, match, matchNil,
  matchNull, matchUndefined, mergeArraysWith, pretty, shallowDiffPropsState, showDate,
  showDateNoTime, showTime, throw_, typed, yaml,
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
  sectionListRef: RefObject<SectionList<Saved>> = React.createRef();

  // State
  _firstSectionHeaderHeight: number = 0; // For SectionList.scrollToLocation({viewOffset})

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
        state:    {
          timestamp: matchSource(source, {
            xc:   source => new Date('3000'), // Put xc at the top [TODO Add Saved.created:timestamp]
            user: source => source.metadata.created,
          }),
        },
      },
      source,
    };
  }

  // TODO
  // searchSavedFromSource = (source: Source): SearchSaved => {
  //   ...
  // }

  // Edit user rec
  //  - Reload saveds when done
  editUserRec = async (source: UserSource): Promise<void> => {
    const audioPath = UserRec.audioPath(source);
    const metadata  = await UserRec.readMetadata(audioPath);
    const title     = await new Promise<null | string>((resolve, reject) => {
      // Docs: https://facebook.github.io/react-native/docs/alertios#prompt
      //  - TODO(android): Won't work on android
      AlertIOS.prompt(
        'Title',      // title
        undefined,    // message
        [             // callbackOrButtons
          {text: 'Cancel', style: 'cancel',  onPress: () => resolve(null)},
          {text: 'Save',   style: 'default', onPress: s  => resolve(s)},
        ],
        'plain-text', // type ('plain-text' | 'secure-text' | 'login-password')
        ifNull(metadata.title, () => undefined), // defaultValue
        'default',    // keyboardType (see https://facebook.github.io/react-native/docs/alertios#prompt)
      );
    });
    if (title !== null) {
      UserRec.writeMetadata(audioPath, {...metadata,
        title,
      });
      await this.loadSavedsFromFs();
    }
  }

  // Delete user rec
  //  - Reload saveds when done
  deleteUserRec = async (source: UserSource): Promise<void> => {
    const audioPath = UserRec.audioPath(source);
    log.info('deleteUserRec', {source, audioPath});
    const confirm = await new Promise<boolean>((resolve, reject) => {
      Alert.alert('Delete this recording?', "This can't be undone.", [
        {style: 'cancel',  text: 'Cancel', onPress: () => resolve(false)},
        {style: 'default', text: 'Delete', onPress: () => resolve(true)},
      ]);
    });
    if (confirm) {
      await fs.unlink(audioPath);
      await this.loadSavedsFromFs();
    }
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
    type Section = SectionListData<Saved>;
    const sections: Array<Section> = (
      _(this.state.saveds)
      .groupBy(saved => showDateNoTime(_.get(saved.location.state, 'timestamp',
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
              marginTop: Styles.tabBarText.marginTop,
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
              keyExtractor={(saved, index) => `${index}`}
              ListEmptyComponent={(
                <View style={[Styles.center, {padding: 30}]}>
                  <Text style={material.subheading}>
                    No saved items
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
              renderItem={({item: saved, index, section}: {item: Saved, index: number, section: Section}) => (

                // Swipe to delete/edit
                local(() => {
                  const swipeableRef: RefObject<Swipeable> = React.createRef();
                  return (
                    <Swipeable
                      ref={swipeableRef}
                      // TODO Make content backgroundColor:red (like Overcast)
                      //  - Upgrade 1.0.8 -> 1.1.0 so we can try childrenContainerStyle
                      //    - https://github.com/kmagiera/react-native-gesture-handler/releases
                      useNativeAnimations={true} // (Blindly enabled, not sure if helpful)
                      renderLeftActions={(progress, dragX) => (
                        <RectButton
                          style={[Styles.center, {
                            width: 60, // Approximate, since height flows from text contents
                            backgroundColor: iOSColors.purple,
                          }]}
                          onPress={async () => {
                            const swipeable = swipeableRef.current!; // (Before await, else ref becomes null)
                            swipeable.close(); // .close before update (via loadSavedsFromFs) else react errors
                            await matchSaved(saved, {
                              search: async saved => {}, // TODO Is it meaningful to edit a saved search? [don't have saved searches yet]
                              record: async saved => await matchSource(saved.source, {
                                xc:   async source => {}, // TODO Is it meaningful to edit a saved xc rec? [don't have saved xc recs yet]
                                user: async source => await this.editUserRec(source),
                              }),
                            });
                          }}
                        >
                          <Feather name='edit' style={{
                            color: iOSColors.white,
                            fontSize: 30,
                          }}/>
                        </RectButton>
                      )}
                      renderRightActions={(progress, dragX) => (
                        <RectButton
                          style={[Styles.center, {
                            width: 60, // Approximate, since height flows from text contents
                            backgroundColor: iOSColors.red,
                          }]}
                          onPress={async () => {
                            const swipeable = swipeableRef.current!; // (Before await, else ref becomes null)
                            swipeable.close(); // .close before update (via loadSavedsFromFs) else react errors
                            await matchSaved(saved, {
                              search: async saved => {}, // TODO Unsave search [don't have saved searches yet]
                              record: async saved => await matchSource(saved.source, {
                                xc:   async source => {}, // TODO Unsave xc rec [don't have saved xc recs yet]
                                user: async source => await this.deleteUserRec(source),
                              }),
                            });
                          }}
                        >
                          <Feather name='trash-2' style={{
                            color: iOSColors.white,
                            fontSize: 30,
                          }}/>
                        </RectButton>
                      )}
                    >

                      {/* Tap to edit/view rec (in RecordScreen) */}
                      <RectButton
                        style={{
                          backgroundColor: iOSColors.white, // Opaque bg so swipe buttons don't show through during swipe
                        }}
                        onPress={() => {
                          this.props.go(saved.tab, {path: saved.location.pathname});
                        }}
                      >
                        <View style={{
                          flex: 1,
                          flexDirection: 'row',
                          ...Styles.center,
                          padding: 5,
                          // Highlight active location per tab
                          backgroundColor: (Saved.isOpenInTab(saved, this.props.tabLocations)
                            ? `${tabProps[saved.tab].color}22`
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
                            color: iOSColors.gray,
                          }}
                            name={this.props.iconForTab[saved.tab]}
                          />
                          <View style={{
                            flex: 1,
                            flexDirection: 'column',
                            paddingLeft: 5,
                          }}>
                            {/* TODO Dedupe with RecentScreen.render */}
                            <Text style={material.body1}>

                              {/* Item title */}
                              {matchSaved(saved, {
                                record: saved => matchSource(saved.source, {
                                  xc:   source => 'XC', // TODO [we don't yet save xc recs]
                                  user: source => (
                                    // TODO Show more metadata fields: species, ...
                                    ifEmpty(puts(source.metadata.title), () => 'Untitled')
                                  ),
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
                            <Text style={material.caption}>

                              {/* Item caption */}
                              {matchSaved(saved, {
                                record: saved => Source.show(saved.source, {
                                  species:  this.props.xc,
                                  long:     true, // e.g. 'User recording: ...' / 'XC recording: ...'
                                  showDate: showTime,
                                }),
                                search: saved => null,
                              })}

                            </Text>
                          </View>
                        </View>
                      </RectButton>

                    </Swipeable>
                  );
                })

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
