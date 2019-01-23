import _ from 'lodash';
import React, { RefObject, PureComponent } from 'react';
import {
  ActivityIndicator, Dimensions, FlatList, Image, Platform, Text, TouchableWithoutFeedback, View, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';

import {
  EditRec, matchRecordPathParams, matchSearchPathParams, matchSource, recordPathParamsFromLocation, Rec,
  searchPathParamsFromLocation, Source, SourceId, UserRec, XCRec,
} from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { debug_print, Log, puts, rich } from 'app/log';
import { Go, TabName } from 'app/router';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  global, into, json, local, mapNil, mapNull, mapUndefined, match, matchNil, matchNull, matchUndefined, mergeArraysWith,
  shallowDiffPropsState, showDate, throw_, typed, yaml,
} from 'app/utils';
import { XC } from 'app/xc';

const log = new Log('SavedScreen');

interface Props {
  // App globals
  go:         Go;
  xc:         XC;
  ebird:      Ebird;
  // RecentScreen
  iconForTab: {[key in TabName]: string};
}

interface State {
  status: 'loading' | 'ready';
  saveds: Array<Saved>;
}

interface Saved {
  tab:  TabName;
  path: string;
}

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

    // Reload saveds when a new user/edit recs is created
    ['user', 'edit'].forEach(k => Rec.emitter.addListener(k, async (source: Source) => {
      log.info('Rec.emitter.listener', {source});
      await this.loadSavedsFromFs();
    }));

    await this.loadSavedsFromFs();

  }

  loadSavedsFromFs = async () => {
    log.info('loadSavedsFromFs');

    // Load user/edit recs
    //  - Current representation of "saved" is all user/edit recs in the fs
    //  - TODO Add a delete/unsave button so user can clean up unwanted recs
    const userRecSources = await UserRec.listAudioSources();
    const editSources    = await EditRec.listAudioSources();

    // Order saveds
    const saveds = (
      _<Source[][]>([
        userRecSources,
        editSources,
      ])
      .flatten()
      .sortBy(x => matchSource(x, {
        xc:   source => throw_(`Impossible xc case: ${x}`),
        user: source => source.created,
        edit: source => source.edit.created,
      }))
      .value()
      .slice().reverse() // (Copy b/c reverse mutates)
      .map<Saved>(source => ({
        tab:  'record',
        path: `/edit/${encodeURIComponent(Source.stringify(source))}`,
      }))
    );

    this.setState({
      status: 'ready',
      saveds,
    });

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
              Saved
            </Text>
          </View>
        </TouchableWithoutFeedback>

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
            <FlatList <Saved>
              ref={this.flatListRef}
              style={{
                ...Styles.fill,
              }}
              contentInset={{
                top:    -1, // Hide top elem border under bottom border of title bar
                bottom: -1, // Hide bottom elem border under top border of tab bar
              }}
              initialNumToRender={20} // Enough to fill one screen (and not much more)
              data={this.state.saveds} // TODO
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
                    this.props.go(saved.tab, {path: saved.path});
                  }}
                >
                  <View style={{
                    flex: 1,
                    flexDirection: 'row',
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

                        {/* {saved.path} */}

                        {/* TODO Dedupe with RecentScreen.render */}
                        {local(() => {

                          // Mock a location from saved.path for *PathParamsFromLocation
                          //  - TODO Refactor *PathParamsFromLocation so we don't need to mock junk fields
                          const location = {
                            pathname: saved.path,               // Used
                            search:   '',                       // Used
                            hash:     '',                       // Ignored
                            key:      undefined,                // Ignored
                            state:    {timestamp: new Date(0)}, // Ignored
                          };

                          return match<TabName, string>(saved.tab,
                            ['record', () => matchRecordPathParams(recordPathParamsFromLocation(location), {
                              root: () => (
                                '[ROOT]'
                              ),
                              edit: ({sourceId}) => (
                                SourceId.show(sourceId, {
                                  species: this.props.xc,
                                  long:    true, // e.g. 'User recording: ...' / 'XC recording: ...'
                                })
                              ),
                            })],
                            ['search', () => matchSearchPathParams(searchPathParamsFromLocation(location), {
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
                          );

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
