import _ from 'lodash';
import React, { RefObject, PureComponent } from 'react';
import {
  Dimensions, FlatList, Image, Platform, Text, TouchableWithoutFeedback, View, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';

import {
  EditRec, matchRecordPathParams, matchSearchPathParams, recordPathParamsFromLocation, Rec,
  searchPathParamsFromLocation, Source, SourceId, UserRec, XCRec,
} from '../datatypes';
import { Ebird } from '../ebird';
import { Log, rich } from '../log';
import { Go, TabName } from '../router';
import { Styles } from '../styles';
import { StyleSheet } from '../stylesheet';
import {
  global, into, json, local, mapNil, mapNull, mapUndefined, match, matchNil, matchNull, matchUndefined, mergeArraysWith,
  shallowDiffPropsState, showDate, yaml,
} from '../utils';
import { XC } from '../xc';

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
  saveds: Array<Saved>;
}

interface Saved {
  tab:  TabName;
  path: string;
}

export class SavedScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state = {
    saveds: [],
  };

  // Refs
  flatListRef: RefObject<FlatList<Saved>> = React.createRef();

  componentDidMount = async () => {
    log.info('componentDidMount');

    // Load user/edit recs
    //  - Current representation of "saved" is all user/edit recs in the fs
    //  - TODO Add a delete/unsave button so user can clean up unwanted recs
    const userRecSources = (
      (await UserRec.listAudioFilenames())
      .map(filename => UserRec.sourceFromAudioFilename(filename))
    );
    const editSources = (
      (await EditRec.listAudioFilenames())
      .map(filename => EditRec.sourceFromAudioFilename(filename))
    );

    // Order saveds
    //  - TODO How to order? Or, just let user order manually and skip this?
    const saveds = (
      _.concat<Source>(
        userRecSources, // Show last
        editSources,    // Show first
      )
      .slice().reverse() // (Copy b/c reverse mutates)
      .map<Saved>(source => ({
        tab:  'record',
        path: `/edit/${Source.stringify(source)}`,
      }))
    );

    this.setState({
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

          {/* TODO SectionList with dates [of what?] as section headers */}
          <FlatList <Saved>
            ref={this.flatListRef}
            style={{
              ...Styles.fill,
            }}
            contentInset={{
              top:    -1, // Hide top elem border under bottom border of title bar
              bottom: -1, // Hide bottom elem border under top border of tab bar
            }}
            data={this.state.saveds}
            keyExtractor={(saved, index) => `${index}`}
            ListHeaderComponent={(
              // Simulate top border on first item
              <View style={{
                height: 0,
                borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
              }}/>
            )}
            renderItem={({item: saved}) => (
              <RectButton
                onPress={() => {
                  this.props.go(saved.tab, {path: saved.path});
                }}
              >
                <View style={{
                  flex: 1,
                  flexDirection: 'row',
                  padding: 5,
                  borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
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

        </View>

      </View>
    );
  }
}

const styles = StyleSheet.create({
});
