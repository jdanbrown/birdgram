import _ from 'lodash';
import { Location, MemoryHistory } from 'history';
import React, { PureComponent } from 'react';
import { Dimensions, FlatList, Image, Platform, SectionList, Text, View, WebView } from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import {
  matchSearchPathParams, MetadataSpecies, Place, showSourceId, Species,
} from '../datatypes';
import { Ebird } from '../ebird';
import { Log, rich } from '../log';
import { Go, Histories } from '../router';
import { SettingsWrites } from '../settings';
import { Styles } from '../styles';
import { StyleSheet } from '../stylesheet';
import { deepEqual, global, json, matchNull, pretty, shallowDiffPropsState, yaml } from '../utils';

const log = new Log('PlacesScreen');

interface Props {
  // App globals
  location:        Location;
  history:         MemoryHistory;
  histories:       Histories;
  go:              Go;
  metadataSpecies: MetadataSpecies;
  ebird:           Ebird;
  // Settings
  settings:        SettingsWrites;
  showDebug:       boolean;
  place:           null | Place;
  places:          Array<Place>;
}

interface State {
}

export class PlacesScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state = {
  };

  componentDidMount = async () => {
    log.info('componentDidMount');
    global.PlacesScreen = this; // XXX Debug
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
            Saved Places
          </Text>
        </View>

        <View style={{
          flex: 1,
          // backgroundColor: iOSColors.customGray,
        }}>

          <FlatList <null | Place>
            style={{
              ...Styles.fill,
            }}
            contentInset={{
              top:    -1, // Hide top elem border under bottom border of title bar
              bottom: -1, // Hide bottom elem border under top border of tab bar
            }}
            data={[null, ...this.props.places]}
            keyExtractor={(place, index) => `${index}`}
            ListHeaderComponent={(
              // Simulate top border on first item
              <View style={{
                height: 0,
                borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
              }}/>
            )}
            renderItem={({item: place, index}) => (
              <RectButton
                onPress={() => this.props.settings.set('place', place)}
              >
                <View style={{
                  flex: 1,
                  flexDirection: 'column',
                  backgroundColor: deepEqual(place, this.props.place) ? iOSColors.lightGray : iOSColors.white,
                  padding: 5,
                  borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                }}>
                  {/* TODO(ebird_priors): Better formatting */}
                  {matchNull(place, {
                    null: () => (
                      <View style={{flex: 1}}>
                        <Text style={material.body1}>
                          All species (no place filter)
                        </Text>
                        <Text style={material.caption}>
                        </Text>
                        <Text style={material.caption}>
                          {/* TODO Move db up to App (from SearchScreen) so we can show total species count here */}
                          {/*   - this.props.metadataSpecies is all ebird species (i.e. 10k), which isn't what we want */}
                        </Text>
                      </View>
                    ),
                    x: place => (
                      <View style={{flex: 1}}>
                        <Text style={material.body1}>
                          {place.name}
                        </Text>
                        <Text style={material.caption}>
                          {yaml(place.props).slice(1, -1)}
                        </Text>
                        <Text style={material.caption}>
                          {place.species.length} species
                        </Text>
                      </View>
                    ),
                  })}
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
