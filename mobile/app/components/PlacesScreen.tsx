import { EventSubscription } from 'fbemitter';
import geolib from 'geolib';
import _ from 'lodash';
import moment from 'moment';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import {
  ActivityIndicator, Animated, Dimensions, FlatList, FlexStyle, GestureResponderEvent, Image, ImageStyle, Keyboard,
  KeyboardAvoidingView, LayoutChangeEvent, Modal, Platform, RegisteredStyle, ScrollView, SectionList, SectionListData,
  StyleProp, Text, TextInput, TextProps, TextStyle, TouchableHighlight, View, ViewProps, ViewStyle, WebView,
} from 'react-native';
import { BaseButton, BorderlessButton, RectButton } from 'react-native-gesture-handler';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import SearchBar from 'react-native-material-design-searchbar'
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';

import { config } from 'app/config';
import { Geo, GeoCoords, GeoError } from 'app/components/Geo';
import { matchSearchPathParams, Place, Species } from 'app/datatypes';
import { BarchartProps, Ebird, HotspotFindResult, HotspotGeoResult, RegionFindResult } from 'app/ebird';
import { debug_print, Log, puts, rich } from 'app/log';
import { Go, Histories, History, Location } from 'app/router';
import { SettingsWrites } from 'app/settings';
import { normalizeStyle, Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  fastIsEqual, Fun, global, ifNull, json, local, mapNull, mapPop, mapUndefined, match, matchKey, matchNull,
  parseFloatElseNaN, pretty, setAdd, setToggle, shallowDiffPropsState, typed, yaml, yamlPretty,
} from 'app/utils';

const log = new Log('PlacesScreen');

interface Props {
  // App globals
  location:        Location;
  history:         History;
  histories:       Histories;
  go:              Go;
  ebird:           Ebird;
  geo:             Geo;
  nSpecies:        number;
  // Settings
  settings:        SettingsWrites;
  showDebug:       boolean;
  place:           Place; // TODO Replace settings .place->.places everywhere (small refactor)
  savedPlaces:     Array<Place>;
  places:          Set<Place>;
}

interface State {
  regionSearchText:     string;
  hotspotSearchText:    string;
  showSearchResults:    false | 'region' | 'hotspot';
  regionSearchResults:  null | Array<PlaceSearchResult>; // null: need more input, []: empty results
  hotspotSearchResults: null | Array<PlaceSearchResult>; // null: need more input, []: empty results
  // For debugging gps
  debugShowGeoCoords: GeoCoords | null;
  debugShowGeoError:  GeoError  | null;
}

export interface PlaceSearchResult {
  r:         string;
  name:      string;
  lat:       null | number;
  lng:       null | number;
  _version:  string;
  _kind:     string;
  _original: object;
}

export class PlacesScreen extends PureComponent<Props, State> {

  static defaultProps = {
  };

  state: State = {
    regionSearchText:     '',
    hotspotSearchText:    '',
    showSearchResults:    false,
    regionSearchResults:  null,
    hotspotSearchResults: null,
    // For debugging gps
    debugShowGeoCoords: null,
    debugShowGeoError:  null,
  };

  // State
  _geoCoords: null | GeoCoords = null;
  _geoError:  null | GeoError  = null;

  // Listeners
  _listeners: Map<string, EventSubscription> = new Map();

  // Refs
  _regionSearchBarRef:  RefObject<SearchBar> = React.createRef();
  _hotspotSearchBarRef: RefObject<SearchBar> = React.createRef();

  componentDidMount = async () => {
    log.info('componentDidMount', {
      geo: this.props.geo, // XXX Debug: Why is this.props.geo sometimes null? (via this.geoRef.current! in App)
    });
    global.PlacesScreen = this; // XXX Debug

    this.addGeoListeners();
    // this.updateSearchResults(null, null); // Not helpful, simpler without it

  }

  componentWillUnmount = async () => {
    log.info('componentWillUnmount');

    // Unregister listeners (all types)
    this._listeners.forEach((listener, k) => listener.remove());

  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));

    this.updateSearchResults(prevProps, prevState);

  }

  addGeoListeners = () => {

    // Listeners for geo updates
    //  - Careful: setState only if showDebug, else we'd continually re-render as geo updates over time (even when tab inactive)
    const onGeoCoords = (coords: GeoCoords) => {
      this._geoCoords = coords;
      if (!this.props.showDebug) this.setState({debugShowGeoCoords: coords});
    };
    const onGeoError = (error: GeoError) => {
      this._geoError = error;
      if (!this.props.showDebug) this.setState({debugShowGeoError: error});
    };

    // Add listeners (first) + grab initial state (second, to ensure no gaps)
    this._listeners.set('coords', this.props.geo.emitter.addListener('coords', onGeoCoords));
    this._listeners.set('error',  this.props.geo.emitter.addListener('error',  onGeoError));
    if (this.props.geo.coords) {
      onGeoCoords(this.props.geo.coords);
    }

  }

  updateSearchResults = async (prevProps: Props, prevState: State) => {
    const {state} = this;
    const {ebird} = this.props;
    log.debug('updateSearchResults', () => rich({
      state,
      prevState,
      diff: shallowDiffPropsState(prevProps, prevState, this.props, this.state),
    }));

    // Respond to the field that changed
    //  - Adding/removing chars updates search results
    //  - 'x'-ing field clears search results
    if (
      // Input changed
      state.regionSearchText !== prevState.regionSearchText
      // Or null results and we gained focus (e.g. for first focus after mount)
      || state.regionSearchResults == null && state.showSearchResults === 'region' && prevState.showSearchResults !== 'region'
    ) {

      // Region search
      const regionSearchResults = await match<number, Promise<State['regionSearchResults']>>(state.regionSearchText.length,
        // 0-1 chars -> noop (polite like ebird.org)
        [0,             async () => null],
        [1,             async () => null],
        // ≥2 chars -> search regions (by name)
        [match.default, async () => this.fromRegionFind(await ebird.regionFind(state.regionSearchText))],
      );
      this.setState({
        showSearchResults: state.showSearchResults && 'region',
        regionSearchResults,
      });

    } else if (
      // Input changed
      state.hotspotSearchText !== prevState.hotspotSearchText
      // Or null results and we gained focus (e.g. for first focus after mount)
      || state.hotspotSearchResults == null && state.showSearchResults === 'hotspot' && prevState.showSearchResults !== 'hotspot'
    ) {

      // Hotspot search
      const hotspotSearchResults = await match<number, Promise<State['hotspotSearchResults']>>(state.hotspotSearchText.length,
        // 0 chars -> nearby hotspots (if we have gps coords)
        [0,             async () => await matchNull(this._geoCoords, {
          x:    async coords => this.fromHotspotGeo(coords, await ebird.hotspotGeo(coords)),
          null: async ()     => typed<State['hotspotSearchResults']>(null), // No gps coords
        })],
        // 1-2 chars -> noop (polite like ebird.org)
        [1,             async () => null],
        [2,             async () => null],
        // ≥3 chars -> search hotspots (by name)
        [match.default, async () => this.fromHotspotFind(await ebird.hotspotFind(state.hotspotSearchText))],
      );
      this.setState({
        showSearchResults: state.showSearchResults && 'hotspot',
        hotspotSearchResults,
      });

    }

  }

  fromRegionFind = (xs: Array<RegionFindResult>): Array<PlaceSearchResult> => {
    return (
      _(xs)
      // Adapt from RegionFindResult
      .map(x => {
        const [locId, lat, lng] = x.code.split(',');
        return {
          r:         x.code,
          name:      x.name,
          lat:       null,
          lng:       null,
          _version:  'v0',
          _kind:     'RegionFindResult',
          _original: x,
        };
      })
      .value()
    );
  }

  fromHotspotFind = (xs: Array<HotspotFindResult>): Array<PlaceSearchResult> => {
    return (
      _(xs)
      // Adapt from HotspotFindResult
      .map(x => {
        const [r, lat, lng] = x.code.split(',');
        return {
          r,
          name:      x.name,
          lat:       parseFloatElseNaN(lat),
          lng:       parseFloatElseNaN(lng),
          _version:  'v0',
          _kind:     'HotspotFindResult',
          _original: x,
        };
      })
      .value()
    );
  }

  fromHotspotGeo = (coords: GeoCoords, xs: Array<HotspotGeoResult>): Array<PlaceSearchResult> => {
    return (
      _(xs)
      // Adapt from HotspotGeoResult
      .map(x => ({
        r:         x.locId,
        name:      `${x.locName}, ${x.subnational1Code}`,
        lat:       x.lat,
        lng:       x.lng,
        _version:  'v0',
        _kind:     'HotspotGeoResult',
        _original: x,
      }))
      // Sort by proximity
      .sortBy(x => geolib.getDistance(coords.coords, {
        latitude:  x.lat,
        longitude: x.lng,
      }))
      .value()
    );
  }

  placeFromResult = async (result: PlaceSearchResult): Promise<Place> => {
    const {ebird} = this.props;
    const {r, name} = result;
    const props: BarchartProps = {
      r,
      // TODO Let user control month/year (bmo,emo,byr,eyr)
      byr: 2008, // Hardcode last ~10y
    };
    const species = await ebird.barchartSpecies(props);
    return {name, species, props};
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

          {/* Search bars for region/hotspot */}
          <View style={{
            flexDirection: 'row',
          }}>
            {local(() => {
              const regionClearSearch = () => {
                this.setState({
                  showSearchResults: false,
                  regionSearchText:  '',
                });
                // HACK Also have to _textInput.clear() to propagate our '' to the native component
                mapNull(this._regionSearchBarRef.current, x => (x as any)._textInput.clear());
              };
              const hotspotClearSearch = () => {
                this.setState({
                  showSearchResults: false,
                  hotspotSearchText: '',
                });
                // HACK Also have to _textInput.clear() to propagate our '' to the native component
                mapNull(this._hotspotSearchBarRef.current, x => (x as any)._textInput.clear());
              };
              return [
                // (Want to de-dupe these, but I couldn't make typing work in the setState objects)
                {
                  placeholder:          'Region',
                  ref:                  this._regionSearchBarRef,
                  value:                this.state.regionSearchText,
                  alwaysShowBackButton: this.state.showSearchResults === 'region',
                  onSearchChange:       this.debounceOnSearchChange((x: string) => this.setState({
                    regionSearchText:  x,
                  })),
                  onBackPress: () => regionClearSearch(),
                  onFocus: () => this.setState(state => ({
                    showSearchResults: 'region',
                    regionSearchText:  ifNull(state.regionSearchText, () => ''), // null -> '' on first focus
                  })),
                  onBlur: () => {
                    if (this.state.showSearchResults === 'region' && this.state.regionSearchResults === null) {
                      regionClearSearch();
                    }
                  },
                }, {
                  placeholder:          'Hotspot',
                  ref:                  this._hotspotSearchBarRef,
                  value:                this.state.hotspotSearchText,
                  alwaysShowBackButton: this.state.showSearchResults === 'hotspot',
                  onSearchChange:       this.debounceOnSearchChange((x: string) => this.setState({
                    hotspotSearchText: x,
                  })),
                  onBackPress: () => hotspotClearSearch(),
                  onFocus: () => this.setState(state => ({
                    showSearchResults: 'hotspot',
                    hotspotSearchText: ifNull(state.hotspotSearchText, () => ''), // null -> '' on first focus
                  })),
                  onBlur: () => {
                    if (this.state.showSearchResults === 'hotspot' && this.state.hotspotSearchResults === null) {
                      hotspotClearSearch();
                    }
                  },
                }
              ].map((props, i) => (
                <View
                  key={i}
                  style={{
                    flex: 1,
                  }}
                >
                  <SearchBar // (cf. SearchBar in BrowseScreen)
                    ref={props.ref}
                    // Style
                    height={40}
                    padding={0}
                    inputStyle={{
                      // Disable border from SearchBar (styles.searchBar)
                      borderWidth:       0,
                      // Replace with a border that matches the title bar border
                      borderBottomWidth: Styles.tabBar.borderTopWidth,
                      borderLeftWidth:   i > 0 ? Styles.tabBar.borderTopWidth : 0,
                      borderColor:       Styles.tabBar.borderTopColor,
                    }}
                    // Icons
                    //  - Full-sized hitbox i/o icon-sized hitbox: iconPadding=0 + height/width=40
                    iconPadding={0}
                    iconSearchComponent={(
                      <View style={[Styles.center, {height: 40, width: 40}]}>
                        <Feather size={18} color={iOSColors.gray} name={'search'} />
                      </View>
                    )}
                    iconBackComponent={(
                      <View style={[Styles.center, {height: 40, width: 40}]}>
                        <Feather size={18} color={iOSColors.gray} name={'x'} />
                      </View>
                    )}
                    iconCloseComponent={(<View/>)} // Disable close button (the right 'x')
                    // Force show back button ('x' button) when results are shown, so user can 'x' to hide them
                    alwaysShowBackButton={props.alwaysShowBackButton}
                    // Listeners
                    onFocus={props.onFocus}
                    onBlur={props.onBlur}
                    onSearchChange={props.onSearchChange}
                    onBackPress={props.onBackPress}
                    // TextInputProps
                    inputProps={{
                      placeholder:                   props.placeholder,
                      // WARNING Don't set value/defaultValue because they both interfere with debounce in bad ways
                      //  - Almost everything seems to work fine without them...
                      //  - Except for onBackPress, where we have to hack in _textInput.clear()
                      // value:                      props.value,
                      // defaultValue:               props.value,
                      autoCorrect:                   false,
                      autoCapitalize:                'none',
                      // enablesReturnKeyAutomatically: true,
                      returnKeyType:                 'done',
                      selectTextOnFocus:             false,
                      clearTextOnFocus:              false,
                      keyboardType:                  'default',
                    }}
                  />
                </View>
              ));
            })}
          </View>

          {local(() => {
            if (this.state.showSearchResults) {

              // Search results
              const data = match(this.state.showSearchResults,
                ['region',  () => this.state.regionSearchResults],
                ['hotspot', () => this.state.hotspotSearchResults],
              );
              return (
                <FlatList
                  style={{
                    ...Styles.fill,
                    // Visual cue to indicate that these are search results, not saved places
                    backgroundColor: `${iOSColors.green}33`,
                  }}
                  contentInset={{
                    top:    -1, // Hide top elem border under bottom border of title bar
                    bottom: -1, // Hide bottom elem border under top border of tab bar
                  }}
                  data={data || []}
                  keyExtractor={(result, index) => `${index}`}
                  ListEmptyComponent={
                    data === null ? (
                      <View/> // Blank (i/o "No search results", which would be misleading)
                    ) : (
                      <View style={[Styles.center, {padding: 30}]}>
                        <Text style={material.subheading}>
                          No results
                        </Text>
                      </View>
                    )
                  }
                  renderItem={({item: result, index}) => (
                    <RectButton
                      onPress={async () => {
                        const place = await this.placeFromResult(result);
                        // Hide search results
                        //  - Easy to pull them back up (by tapping back in search bar)
                        this.setState({
                          showSearchResults: false,
                        });
                        // Add place
                        //  - TODO Show some indication that loading is happening
                        this.props.settings.set(settings => ({
                          savedPlaces: [
                            // Add place (to top of list)
                            place,
                            // Remove place (id'd by barchart props) if already exists (e.g. further down in list)
                            ...settings.savedPlaces.filter(x => !_.isEqual(x.props, place.props)),
                          ],
                        }));
                      }}
                    >
                      <View style={{
                        flex: 1,
                        flexDirection: 'column',
                        padding: 5,
                        // Bottom border on all items, top border on first item
                        borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                        ...(index != 0 ? {} : {
                          borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: 'black',
                        }),
                      }}>
                        <View style={{flex: 1}}>
                          <Text style={material.body1}>
                            {result.name}
                          </Text>
                          <Text style={material.caption}>
                            {result.r}
                          </Text>
                        </View>
                      </View>
                    </RectButton>
                  )}
                />
              );

            } else {

              // Saved places
              return (
                <FlatList
                  style={{
                    ...Styles.fill,
                  }}
                  contentInset={{
                    top:    -1, // Hide top elem border under bottom border of title bar
                    bottom: -1, // Hide bottom elem border under top border of tab bar
                  }}
                  data={typed<Array<Place>>([
                    this.props.ebird.allPlace,
                    ...this.props.savedPlaces,
                  ])}
                  keyExtractor={(place, index) => yaml(place.props)}
                  ListEmptyComponent={(
                    <View style={[Styles.center, {padding: 30}]}>
                      <Text style={material.subheading}>
                        No places
                      </Text>
                    </View>
                  )}
                  renderItem={({item: place, index}) => (
                    <Swipeable
                      // TODO Make content backgroundColor:red (like Overcast)
                      //  - Upgrade 1.0.8 -> 1.1.0 so we can try childrenContainerStyle
                      //    - https://github.com/kmagiera/react-native-gesture-handler/releases
                      useNativeAnimations={true} // (Blindly enabled, not sure if helpful)
                      renderRightActions={index === 0
                        // Disallow deleting the allPlace (don't present swipe/trash at all)
                        ? undefined
                        // Allow deleting all other places
                        : (progress, dragX) => (
                          <RectButton
                            style={[Styles.center, {
                              width: 60, // Approximate, since height flows from text contents
                              backgroundColor: iOSColors.red,
                            }]}
                            onPress={() => {
                              // Remove place (id'd by barchart props)
                              this.props.settings.set(settings => ({
                                savedPlaces: settings.savedPlaces.filter(x => !_.isEqual(x.props, place.props)),
                                places:      new Set(), // HACK(place_id): Only way to clear orphaned junk, until place.id
                                place:       this.mergePlaces(new Set()), // XXX Back compat (until we settings .place->.places)
                              }));
                            }}
                          >
                            <Feather name={'trash-2'} style={{
                              color: iOSColors.white,
                              fontSize: 30,
                            }}/>
                          </RectButton>
                        )
                      }
                    >
                      <RectButton
                        onPress={() => {
                          // FIXME Perf: really slow in dev (but no log.timed surfaces the bottleneck...)
                          this.props.settings.set(settings => {
                            const places = setToggle(settings.places, place); // FIXME(place_id): Add place.id to do this correctly
                            return {
                              places,
                              place: this.mergePlaces(places), // XXX Back compat (until we settings .place->.places)
                            };
                          });
                          // this.props.go('search'); // TODO Helpful or not helpful?
                        }}
                      >
                        <View style={{
                          flex: 1,
                          flexDirection: 'column',
                          // FIXME(place_id): Add place.id to do this correctly
                          backgroundColor: (
                            this.props.places.size === 0 && index === 0 // ebird.allPlace
                            || this.props.places.has(place)
                            ? iOSColors.lightGray : iOSColors.white
                          ),
                          padding: 5,
                          // Bottom border on all items, top border on first item
                          borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'black',
                          ...(index != 0 ? {} : {
                            borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: 'black',
                          }),
                        }}>
                          {/* TODO(ebird_priors): Better formatting */}
                          <View style={{flex: 1}}>
                            <Text style={material.body1}>
                              {place.name}
                            </Text>
                            <Text style={[material.caption, {color: iOSColors.black}]}>
                              {/* TODO(place_n_species): Make this less confusing (n1/n2/n3) */}
                              {/*   - place.species often includes species that aren't in the app (e.g. CR place in US app) */}
                              {/*   - Most prominently, we want to show the number of place species that are in the app */}
                              {/*   - Less prominently, we want to also show the total number of species for the place */}
                              TODO / {place.species.length} species
                            </Text>
                            <Text style={material.caption}>
                              {yaml(place.props || '').slice(1, -1)}
                            </Text>
                          </View>
                        </View>
                      </RectButton>
                    </Swipeable>
                  )}
                />
              );

            }
          })}

        </View>

        {/* Debug info */}
        <this.DebugView style={{
          width: '100%',
        }}>
          <this.DebugText>
            {yamlPretty({
              debugShowGeoCoords: mapNull(this.state.debugShowGeoCoords, coords => ({
                ...coords,
                timestampStr: mapNull(coords, x => moment(x.timestamp).toISOString(true)), // true for local i/o utc
              })),
              debugShowGeoError: this.state.debugShowGeoError,
            })}
          </this.DebugText>
        </this.DebugView>

      </View>
    );
  }

  // Debounce search bar edits to throttle ebird api calls
  debounceOnSearchChange = <F extends Fun>(onSearchChange: F): F => {
    return _.debounce(onSearchChange, 250);
  }

  // Jam multi-select places through the existing settings.state.place code
  //  - TODO Expand settings.state .place -> .places for other screens [will require a small amount of refactoring]
  mergePlaces = (places: Set<Place>): null | Place => {
    return (places.size === 0
      ? null // -> ebird.allPlace (via App.place)
      : {
        species: _.uniq(_.flatMap(Array.from(places), x => x.species)), // Here's what we're actually after
        name:    `${places.size} places`,                               // Good enough
        props:   {r: 'XXX'},                                            // Hopefully this doesn't break anything...
      }
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
