import _ from 'lodash';
import React, { PureComponent } from 'react';
import { Dimensions, FlatList, Image, Platform, SectionList, Text, View, WebView } from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { matchSearchPathParams, MetadataSpecies, Place, Species } from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { Log, rich } from 'app/log';
import { Go, Histories, History, Location } from 'app/router';
import { SettingsWrites } from 'app/settings';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import { deepEqual, global, json, matchNull, pretty, shallowDiffPropsState, yaml } from 'app/utils';

const log = new Log('PlacesScreen');

interface Props {
  // App globals
  location:        Location;
  history:         History;
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

  state: State = {
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
            Places
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
            ListEmptyComponent={(
              <View style={[Styles.center, {padding: 30}]}>
                <Text style={material.subheading}>
                  No places
                </Text>
              </View>
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
                  // Bottom border on all items, top border on first item
                  borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                  ...(index != 0 ? {} : {
                    borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: 'black',
                  }),
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
