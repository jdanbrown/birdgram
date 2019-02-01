import _ from 'lodash';
import moment from 'moment';
import React, { PureComponent } from 'react';
import {
  Dimensions, FlatList, Image, Platform, SectionList, Text, TextProps, View, ViewProps, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { Geo, GeoCoords, GeoError } from 'app/components/Geo';
import { matchSearchPathParams, MetadataSpecies, Place, Species } from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { Log, rich } from 'app/log';
import { Go, Histories, History, Location } from 'app/router';
import { SettingsWrites } from 'app/settings';
import { normalizeStyle, Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  fastIsEqual, global, json, mapNull, matchNull, pretty, shallowDiffPropsState, typed, yaml, yamlPretty,
} from 'app/utils';

const log = new Log('PlacesScreen');

interface Props {
  // App globals
  location:        Location;
  history:         History;
  histories:       Histories;
  go:              Go;
  metadataSpecies: MetadataSpecies;
  ebird:           Ebird;
  geo:             Geo;
  // Settings
  settings:        SettingsWrites;
  showDebug:       boolean;
  place:           Place | null;
  places:          Array<Place>;
}

interface State {
  // For debugging gps
  lastSeenGeoCoords: GeoCoords | null;
  lastSeenGeoError:  GeoError  | null;
}

export class PlacesScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state: State = {
    lastSeenGeoCoords: null,
    lastSeenGeoError:  null,
  };

  componentDidMount = async () => {
    log.info('componentDidMount', {
      geo: this.props.geo, // XXX Debug: Why is this.props.geo sometimes null? (via this.geoRef.current! in App)
    });
    global.PlacesScreen = this; // XXX Debug

    // Subscribe to geo updates + grab initial geo
    //  - Subscribe before grab, to avoid missing events
    this.props.geo.emitter.addListener('coords', (coords: GeoCoords) => this.setState({lastSeenGeoCoords: coords}));
    this.props.geo.emitter.addListener('error',  (error:  GeoError)  => this.setState({lastSeenGeoError:  error}));
    this.setState({
      lastSeenGeoCoords: this.props.geo.coords,
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

          <FlatList
            style={{
              ...Styles.fill,
            }}
            contentInset={{
              top:    -1, // Hide top elem border under bottom border of title bar
              bottom: -1, // Hide bottom elem border under top border of tab bar
            }}
            data={typed<Array<Place | null>>([null, ...this.props.places])}
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
                  backgroundColor: fastIsEqual(place, this.props.place) ? iOSColors.lightGray : iOSColors.white,
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

        {/* Debug info */}
        <this.DebugView style={{
          width: '100%',
        }}>
          <this.DebugText>
            {yamlPretty({
              lastSeenGeoCoords: mapNull(this.state.lastSeenGeoCoords, coords => ({
                ...coords,
                timestampStr: mapNull(coords, x => moment(x.timestamp).toISOString(true)), // true for local i/o utc
              })),
              lastSeenGeoError: this.state.lastSeenGeoError,
            })}
          </this.DebugText>
        </this.DebugView>

      </View>
    );
  }

  // Debug components
  //  - [Tried and gave up once to make well-typed generic version of these (DebugFoo = addStyle(Foo, ...) = withProps(Foo, ...))]
  DebugView = (props: ViewProps & {children: any}) => (
    !this.props.showDebug ? null : (
      <View {...{
        ...props,
        style: [Styles.debugView, ...normalizeStyle(props.style)],
      }}/>
    )
  );
  DebugText = (props: TextProps & {children: any}) => (
    !this.props.showDebug ? null : (
      <Text {...{
        ...props,
        style: [Styles.debugText, ...normalizeStyle(props.style)],
      }}/>
    )
  );

}

const styles = StyleSheet.create({
});
