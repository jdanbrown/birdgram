import * as d3sc from 'd3-scale-chromatic';
import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import RN, {
  ActivityIndicator, Animated, Dimensions, FlatList, FlexStyle, GestureResponderEvent, Image, ImageStyle, Keyboard,
  KeyboardAvoidingView, LayoutChangeEvent, Modal, Platform, RegisteredStyle, ScrollView, SectionList, SectionListData,
  StyleProp, Text, TextInput, TextStyle, TouchableHighlight, View, ViewStyle, WebView,
} from 'react-native';
import ActionSheet from 'react-native-actionsheet'; // [Must `import ActionSheet` i/o `import { ActionSheet }`, else barf]
import FastImage from 'react-native-fast-image';
import * as Gesture from 'react-native-gesture-handler';
import {
  BaseButton, BorderlessButton, BorderlessButtonProperties, LongPressGestureHandler, PanGestureHandler,
  PinchGestureHandler, RectButton, TapGestureHandler,
  // FlatList, ScrollView, Slider, Switch, TextInput, // TODO Needed?
} from 'react-native-gesture-handler';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import SearchBar from 'react-native-material-design-searchbar'
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { IconProps } from 'react-native-vector-icons/Icon';
import timer from 'react-native-timer';
import Feather from 'react-native-vector-icons/Feather';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import { Link, matchPath, Redirect, Route, Switch } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
import stringHash from "string-hash";
const fs = RNFB.fs;

import { ActionSheetBasic } from 'app/components/ActionSheets';
import {
  MetadataColumnBelow, MetadataColumnsBelow, MetadataColumnLeft, MetadataColumnsLeft, metadataLabel, MetadataLabel,
  MetadataText,
} from 'app/components/MetadataColumns';
import { TabBarStyle } from 'app/components/TabRoutes';
import { config } from 'app/config';
import {
  ModelsSearch, matchRec, matchSearchPathParams, Place, Quality, Rec, SearchPathParams, searchPathParamsFromLocation,
  SearchRecs, ServerConfig, Source, SourceId, Species, SpeciesMetadata, SpectroPathOpts, UserRec, XCRec,
} from 'app/datatypes';
import { DB } from 'app/db';
import { Ebird } from 'app/ebird';
import { debug_print, Log, puts, rich, tap } from 'app/log';
import { NativeSearch } from 'app/native/Search';
import { NativeSpectro } from 'app/native/Spectro';
import { Go, History, Location, locationKeyIsEqual, locationPathIsEqual } from 'app/router';
import { SettingsWrites } from 'app/settings';
import Sound from 'app/sound';
import { SQL, sqlf } from 'app/sql';
import { StyleSheet } from 'app/stylesheet';
import { normalizeStyle, LabelStyle, labelStyles, Styles } from 'app/styles';
import {
  all, any, assert, chance, Clamp, Dim, ensureParentDir, fastIsEqual, finallyAsync, getOrSet, global, ifNull, into,
  json, local, mapMapValues, mapNull, mapUndefined, match, matchEmpty, matchNull, matchUndefined, noawait,
  objectKeysTyped, Omit, Point, pretty, QueryString, round, setAdd, setDelete, setToggle, shallowDiffPropsState, Style,
  throw_, Timer, typed, yaml, yamlPretty, zipSame,
} from 'app/utils';
import { XC } from 'app/xc';

const log = new Log('SearchScreen');

//
// Utils
//

const sidewaysTextWidth = 14;
const recEditingButtonWidth = 30;

interface ScrollViewState {
  contentOffset: Point;
  // (More fields available in NativeScrollEvent)
}

// (Callers: RecentScreen, SavedScreen)
export type Query = QueryNone | QueryRandom | QuerySpecies | QueryRec;
export type QueryNone    = {kind: 'none'}; // e.g. so we can show nothing on redirect from '/'
export type QueryRandom  = {kind: 'random',  filters: Filters, seed: number};
export type QuerySpecies = {kind: 'species', filters: Filters, species: string};
export type QueryRec     = {kind: 'rec',     filters: Filters, source: Source};
export function matchQuery<X>(query: Query, cases: {
  none:    (query: QueryNone)    => X,
  random:  (query: QueryRandom)  => X,
  species: (query: QuerySpecies) => X,
  rec:     (query: QueryRec)     => X,
}): X {
  switch (query.kind) {
    case 'none':    return cases.none(query);
    case 'random':  return cases.random(query);
    case 'species': return cases.species(query);
    case 'rec':     return cases.rec(query);
  }
}

export const Query = {

  // null if source not found
  //  - (Callers: updateForLocation, RecentScreen, SavedScreen)
  loadFromLocation: async (location: Location): Promise<Query | null> => {
    return await matchSearchPathParams<Promise<Query | null>>(searchPathParamsFromLocation(location), {
      root:    async ()                    => ({kind: 'species', filters: {}, species: ''}),
      random:  async ({filters, seed})     => ({kind: 'random',  filters, seed}),
      species: async ({filters, species})  => ({kind: 'species', filters, species}),
      rec:     async ({filters, sourceId}) => {
        if (SourceId.isOldStyleEdit(sourceId)) {
          return null; // Treat old-style edit recs (e.g. from history) like source not found
        } else {
          // Load source
          //  - Propagate null from Source.load as source not found (e.g. user deleted a user rec, or xc dataset changed)
          return mapNull(
            await Source.load(sourceId),
            source => typed<QueryRec>({kind: 'rec', filters, source}),
          );
        }
      },
    });
  },

};

// TODO(put_all_query_state_in_location)
export interface Filters {
  quality?: Array<Quality>;
  placeId?: string;
  text?:    string; // TODO(text_filter)
}

export const Filters = {

  fromQueryString: (q: QueryString): Filters => ({
    // HACK Typing
    quality: _.get(q, 'quality', '').split(',').filter(x => (Quality.values as Array<string>).includes(x)) as Array<Quality>,
    placeId: _.get(q, 'placeId'),
    text:    _.get(q, 'text'),
  }),

  toQueryString: (x: Filters): QueryString => _.pickBy({
    quality: (x.quality || []).join(','),
    placeId: x.placeId,
    text:    x.text,
  }, (v, k) => v !== undefined) as {[key: string]: string} // HACK Typing

};

//
// SearchScreen
//

interface Props {
  // App globals
  serverConfig:            ServerConfig;
  modelsSearch:            ModelsSearch;
  location:                Location;
  history:                 History;
  go:                      Go;
  xc:                      XC;
  ebird:                   Ebird;
  // Settings
  settings:                SettingsWrites;
  db:                      DB;
  showDebug:               boolean;
  showMetadataLeft:        boolean;
  showMetadataBelow:       boolean;
  metadataColumnsLeft:     Array<MetadataColumnLeft>;
  metadataColumnsBelow:    Array<MetadataColumnBelow>;
  editing:                 boolean;
  seekOnPlay:              boolean;
  playOnTap:               boolean;
  playingProgressEnable:   boolean;
  playingProgressInterval: number;
  spectroScale:            number;
  place:                   null | Place;
  places:                  Array<Place>;
  // SearchScreen
  f_bins:                  number;
  spectroBase:             Dim<number>;
  spectroScaleClamp:       Clamp<number>;
  default_n_recs:          number;
  default_n_sp:            number;
  default_n_per_sp:        number;
  searchRecsMaxDurationS:  number;
}

interface State {
  scrollViewKey: string;
  scrollViewState: ScrollViewState;
  showGenericModal: null | (() => Element);
  searchFilter: string; // For BrowseModal
  showHelp: boolean;
  totalRecs?: number;
  non_f_preds_cols?: Array<string>;
  f_preds_cols?: Array<string>;
  query: null | Query;
  refreshQuery: boolean; // TODO(put_all_query_state_in_location)
  // TODO Persist filters with settings
  //  - Top-level fields instead of nested object so we can use state merging when updating them in isolation
  filterQueryText?: string;
  filterQuality: Array<Quality>;
  n_recs: number;   // For non-rec queries
  n_sp: number;     // For rec queries
  n_per_sp: number; // For rec queries
  sortResults: 'slp,d_pc' | 'd_pc';
  excludeSpecies: Set<string>;
  includeSpecies: Set<string>;
  excludeSpeciesGroups: Set<string>;
  includeSpeciesGroups: Set<string>;
  excludeRecs: Set<string>;
  recs: StateRecs;
  recsQueryTime?: number;
  sourceIdForActionModal?: SourceId;
  playing?: {
    rec: Rec,
    sound: Sound,
    startTime?: number,
    // progressTime?: number,
  };
  playingCurrentTime?: number;
  _spectroScale: number; // Sync from/to Settings (1/3)
};

type StateRecs = 'loading' | 'notfound' | Array<Rec>;

function matchStateRecs<X>(recs: StateRecs, cases: {
  loading:  (x: 'loading')  => X,
  notfound: (x: 'notfound') => X,
  recs:     (x: Array<Rec>) => X,
}): X {
  if (recs === 'loading')  return cases.loading(recs);
  if (recs === 'notfound') return cases.notfound(recs);
  else                     return cases.recs(recs);
}

export class SearchScreen extends PureComponent<Props, State> {

  static defaultProps = {
    spectroBase:            {height: 20, width: Dimensions.get('window').width},
    spectroScaleClamp:      {min: 1, max: 8},
    default_n_recs:         30, // For non-rec queries
    default_n_sp:           10, // For rec queries
    default_n_per_sp:       3,  // For rec queries
    searchRecsMaxDurationS: 10.031,  // HACK Query max(search_recs.duration_s) from db on startup
  };

  // Else we have to do too many setState's, which makes animations jump (e.g. ScrollView momentum)
  _scrollViewState: ScrollViewState = {
    contentOffset: {x: 0, y: 0},
  };

  state: State = {
    scrollViewKey:        '',
    scrollViewState:      this._scrollViewState,
    showGenericModal:     null,
    // showGenericModal:     () => this.BrowseModal(), // XXX(family_list): Debug
    searchFilter:         '', // For BrowseModal
    // searchFilter:         'sparrow', // XXX(family_list): Debug
    showHelp:             false,
    query:                null,
    refreshQuery:         false,
    filterQuality:        ['A', 'B'],
    n_recs:               this.props.default_n_recs,
    n_sp:                 this.props.default_n_sp,
    n_per_sp:             this.props.default_n_per_sp,
    sortResults:          'slp,d_pc',
    excludeSpecies:       new Set(),
    // excludeSpecies:       new Set(['VIRA', 'SWTH']), // XXX(family_list): Debug
    includeSpecies:       new Set(),
    excludeSpeciesGroups: new Set(),
    // excludeSpeciesGroups: new Set(['Waterfowl', 'Wood-Warblers']), // XXX(family_list): Debug
    // excludeSpeciesGroups: new Set(['Waterfowl', 'Wood-Warblers', 'Gnatcatchers', 'New World Sparrows', 'Penduline-Tits and Long-tailed Tits', 'Tyrant Flycatchers: Pewees, Kingbirds, and Allies', 'Martins and Swallows', 'Catbirds, Mockingbirds, and Thrashers', 'Cardinals, Grosbeaks, and Allies', 'Blackbirds', 'Finches, Euphonias, and Allies']), // XXX(family_list): Debug
    includeSpeciesGroups: new Set(),
    excludeRecs:          new Set(),
    recs:                 'loading',
    _spectroScale:        this.props.spectroScale, // Sync from/to Settings (2/3)
  };

  // Getters for props
  get spectroPathOpts(): SpectroPathOpts { return {
    f_bins:  this.props.f_bins, // Higher res for user recs, ignored for xc recs (which are all f_bins=40)
    denoise: true,              // For predict (like Bubo/py/model.swift:Features.denoise=true)
  }}

  // Getters for state
  get filters(): object { return _.pickBy(this.state, (v, k) => k.startsWith('filter')); }
  get recsOrEmpty(): Array<Rec> {
    return matchStateRecs(this.state.recs, {
      loading:  ()   => [],
      notfound: ()   => [],
      recs:     recs => recs,
    });
  }
  get query_rec(): null | Rec {
    return matchStateRecs(this.state.recs, {
      loading:  ()   => null,
      notfound: ()   => null,
      recs:     recs => recs[0] || null, // null if empty recs
    });
  }

  // Private attrs
  soundsCache: Map<SourceId, Promise<Sound> | Sound> = new Map();

  // (Unused, kept for reference)
  // sortActionSheet: RefObject<ActionSheet> = React.createRef();

  // Refs
  scrollViewRef: RefObject<ScrollView> = React.createRef();

  // Avoid constructor
  //  - "If you don't initialize state and you don't bind methods, you don't need to implement a constructor"
  //  - https://reactjs.org/docs/react-component.html#constructor
  // constructor(props) { ... }

  // Avoid getDerivedStateFromProps
  //  - Use only to update state derived from (changed) props -- but avoid deriving state from props!
  //  - Prefer componentDidUpdate to side effect in response to changed props (e.g. fetch data, start animation)
  //  - Prefer memoization to re-run an expensive computation when a prop changes
  //  - Prefer fully controlled / fully uncontrolled with key to reset state when a prop changes
  //  - https://reactjs.org/docs/react-component.html#static-getderivedstatefromprops
  //  - https://reactjs.org/blog/2018/06/07/you-probably-dont-need-derived-state.html
  // static getDerivedStateFromProps(props, state) { ... }

  // After component is first inserted into the DOM
  //  - Commit phase (impure, may read/write DOM, called once per commit)
  //  - Best practices
  //    - Do fetch data (-> setState() -> additional render(), which is ok)
  //    - Do subscribe listeners / scheduler timers (clean up in componentWillUnmount)
  //  - Details
  //    - First render() happens before this (don't try to avoid it, it's ok)
  //    - Immediate setState() will trigger render() a second time before the first screen draw
  componentDidMount = async () => {
    log.info('componentDidMount');
    global.SearchScreen = this; // XXX Debug

    // Configure react-native-sound
    //  - TODO Experiment to figure out which "playback mode" and "audio session mode" we want
    //  - https://github.com/zmxv/react-native-sound/wiki/API#soundsetcategoryvalue-mixwithothers-ios-only
    //  - https://apple.co/2q2osEd
    //  - https://developer.apple.com/documentation/avfoundation/avaudiosession/audio_session_modes
    //  - https://apple.co/2R22tcg
    Sound.setCategory(
      'Playback', // Enable playback in silence mode [cargo-culted from README]
      true,       // mixWithOthers
    );
    Sound.setMode(
      'Default',        // "The default audio session mode"
      // 'Measurement', // TODO Like https://github.com/jsierles/react-native-audio/blob/master/index.js#L42
    );

    // Tell other apps we're using the audio device
    Sound.setActive(true);

    // Query db size (once)
    log.info('componentDidMount: Querying db size');
    await this.props.db.query<{totalRecs: number}>(`
      select count(*) as totalRecs
      from search_recs
    `)(async results => {
      log.info('componentDidMount: state.totalRecs');
      const [{totalRecs}] = results.rows.raw();
      this.setState({
        totalRecs,
      });
    });

    // Query search_recs cols (once)
    log.info('componentDidMount: Querying search_recs cols');
    await this.props.db.query<XCRec>(`
      select *
      from search_recs
      limit 1
    `)(async results => {
      log.info('componentDidMount: state.{non_f_preds_cols,f_preds_cols}');
      const [rec] = results.rows.raw();
      const non_f_preds_cols = Object.keys(rec).filter(k => !k.startsWith('f_preds_'));
      var   f_preds_cols     = Object.keys(rec).filter(k => k.startsWith('f_preds_'));
      f_preds_cols = _.range(f_preds_cols.length).map(i => `f_preds_${i}`); // Reconstruct array to ensure ordering
      this.setState({
        non_f_preds_cols,
        f_preds_cols,
      });
    });

    // Show this.props.location
    await this.updateForLocation(null, null);

  }

  // Before a component is removed from the DOM and destroyed
  //  - Commit phase (impure, may read/write DOM, called once per commit)
  //  - Best practices
  //    - Do unsubscribe listeners / cancel timers (created in componentDidMount)
  //    - Don't setState(), since no more render() will happen for this instance
  componentWillUnmount = async () => {
    log.info('componentWillUnmount');

    // Tell other apps we're no longer using the audio device
    Sound.setActive(false);

    // Release cached sound resources
    await this.releaseSounds();

    // Clear timers
    timer.clearTimeout(this);

  }

  // After props/state change; not called for the initial render()
  //  - Commit phase (impure, may read/write DOM, called once per commit)
  //  - Best practices
  //    - Do operate on DOM in response to changed props/state
  //    - Do fetch data, conditioned on changed props/state (else update loops)
  //    - Do setState(), conditioned on changed props (else update loops)
  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));

    // Reset view state if location changed
    //  - TODO Pass props.key to reset _all_ state? [https://reactjs.org/blog/2018/06/07/you-probably-dont-need-derived-state.html#recap]
    if (!fastIsEqual(this.props.location, prevProps.location)) {
      log.info('componentDidUpdate: Reset view state');
      this.setState({
        searchFilter:         '', // For BrowseModal
        filterQueryText:      undefined,
        n_recs:               this.props.default_n_recs,
        n_sp:                 this.props.default_n_sp,
        n_per_sp:             this.props.default_n_per_sp,
        excludeSpecies:       new Set(),
        includeSpecies:       new Set(),
        excludeSpeciesGroups: new Set(),
        includeSpeciesGroups: new Set(),
        excludeRecs:          new Set(),
      });
    }

    // Else _scrollViewState falls behind on non-scroll/non-zoom events (e.g. +/- buttons)
    this._scrollViewState = this.state.scrollViewState;

    // Sync from/to Settings (3/3)
    //  - These aren't typical: we only use this for (global) settings keys that we also keep locally in state so we can
    //    batch-update them with other local state keys (e.g. global spectroScale + local scrollViewKey)
    //  - QUESTION What's a better pattern for "batch setState(x,y,z) locally + persist settings.set(x) globally"?
    if (this.state._spectroScale !== prevState._spectroScale) {
      noawait(this.props.settings.set({spectroScale: this.state._spectroScale}));
    }

    // Show this.props.location
    await this.updateForLocation(prevProps, prevState);

  }

  randomPath = (seed?: number): string => {
    seed = seed !== undefined ? seed : chance.natural({max: 1e6});
    return `/random/${seed}`;
  }

  // Spectro dims: spectroDimImage (with padding) / spectroDimContent (no padding)
  //  - xc spectro image widths are padded out to ~10s, regardless of duration_s
  //  - user/edit spectro image widths are sized to duration_s
  spectroDimImage = (rec: Rec): Dim<number> => {
    return {
      height: this.props.spectroBase.height * this.state._spectroScale,
      width:  this.scrollViewContentWidths.image * matchRec(rec, {
        xc:   rec => 1,                                                  // Image width ~ 10s
        user: rec => rec.duration_s / this.props.searchRecsMaxDurationS, // Image width ~ duration_s
      }),
    };
  }
  spectroDimContent = (rec: Rec): Dim<number> => {
    return {...this.spectroDimImage(rec),
      width: this.scrollViewContentWidths.image * (
        rec.duration_s / this.props.searchRecsMaxDurationS // Content width ~ duration_s
      ),
    };
  }

  // Manually specify widths of all components that add up to the ScrollView content width so we can explicitly compute
  // and set it, else the ScrollView won't scroll horizontally (the overflow direction)
  //  - https://github.com/facebook/react-native/issues/8579#issuecomment-233162695
  //  - I also tried using onLayout to automatically get subcomponent widths from the DOM instead of manually
  //    maintaining them all here, but that got gnarly and I bailed (bad onLayout/setState interactions causing infinite
  //    update loops, probably because I'm missing conditions in lifecycle methods like componentDidUpdate)
  get scrollViewContentWidth() {
    // Grow width to fit query_rec (if longer than search_recs)
    const recsMaxDurationS = Math.max(
      this.props.searchRecsMaxDurationS,
      matchNull(this.query_rec, {null: () => -Infinity, x: x => x.duration_s}),
    );
    return _.sum(_.values(this.scrollViewContentWidths)) / this.props.searchRecsMaxDurationS * recsMaxDurationS;
  }
  get scrollViewContentWidths() {
    const sidewaysText = sidewaysTextWidth;
    const debugInfo    = !(this.props.showDebug && this.props.showMetadataLeft) ? 0 : 80; // Wide enough for 'n_recs: 123'
    const metadataLeft = !(this.props.showMetadataLeft && !this.props.showMetadataBelow) ? 0 : 65; // Wide enough for 'XC123456'
    return {
      // NOTE Conditions duplicated elsewhere (render, ...)
      recEditing:     !this.props.editing ? 0 : recEditingButtonWidth * this._recEditingButtons.length,
      sidewaysText,
      debugInfo,
      metadataLeft,
      image:          this.props.spectroBase.width * this.state._spectroScale + _.sum([
        -sidewaysText, // Squeeze image so that sidewaysText doesn't increase horizontal scroll
        -debugInfo,    // Squeeze image so that debugInfo doesn't increase horizontal scroll
        -metadataLeft, // Squeeze image so that metadataLeft doesn't increase horizontal scroll
      ]),
    };
  }

  edit_n_sp = (sign: number) => {
    this.setState((state, props) => ({
      n_sp:         state.n_sp + sign * this.props.default_n_sp,
      refreshQuery: true, // XXX(put_all_query_state_in_location)
    }));
  }

  edit_n_per_sp = (sign: number) => {
    this.setState((state, props) => ({
      n_per_sp:     state.n_per_sp + sign * 1,
      refreshQuery: true, // XXX(put_all_query_state_in_location)
    }));
  }

  get queryDesc(): string {
    return matchNull(this.state.query, {
      null: ()    => '...',
      x:    query => matchQuery(query, {
        none:    ()                   => 'none',
        random:  ({filters, seed})    => `random/${seed}`,
        species: ({filters, species}) => species,
        rec:     ({filters, source})  => Source.show(source, {
          species: this.props.xc,
        }),
      }),
    });
  }

  updateForLocation = async (prevProps: null | Props, prevState: null | State) => {
    log.debug('updateForLocation', () => (prevProps === null && prevState === null
      ? rich({prevProps, prevState, props: '[OMITTED]', state: '[OMITTED]'}) // null->{props,state} is very noisy (e.g. xcode logs)
      : rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state))
    ));
    if (
      !(
        // Don't noop if any filters/limits changed [XXX(put_all_query_state_in_location)]
        //  - Use refreshQuery as a proxy for various filters/limits changing
        this.state.refreshQuery ||
        //  - And test other filters/limits here
        !fastIsEqual(this.props.place,                _.get(prevProps, 'place')) ||
        !fastIsEqual(this.state.sortResults,          _.get(prevState, 'sortResults')) ||
        !fastIsEqual(this.state.excludeSpecies,       _.get(prevState, 'excludeSpecies')) ||
        !fastIsEqual(this.state.includeSpecies,       _.get(prevState, 'includeSpecies')) ||
        !fastIsEqual(this.state.excludeSpeciesGroups, _.get(prevState, 'excludeSpeciesGroups')) ||
        !fastIsEqual(this.state.includeSpeciesGroups, _.get(prevState, 'includeSpeciesGroups')) ||
        !fastIsEqual(this.state.excludeRecs,          _.get(prevState, 'excludeRecs'))
      ) && (
        // Noop if location didn't change
        locationPathIsEqual(this.props.location, _.get(prevProps, 'location')) ||
        // Noop if we don't know f_preds_cols yet (assume we'll be called again once we do)
        !this.state.f_preds_cols
      )
    ) {
      log.info('updateForLocation: Skipping');
    } else {
      const timer = new Timer();

      // Set loading state
      //  - TODO Fade previous recs instead of showing a blank screen while loading
      log.info("updateForLocation: state.recs = 'loading'");
      this.setState({
        query: null,
        recs: 'loading',
        refreshQuery: false,
      });
      await this.releaseSounds(); // (Safe to do after clearing state.recs, since it uses this.soundsCache)

      // Load location -> query
      //  - We ignore state.filterQueryText b/c TextInput.onSubmitEditing -> history.push -> navParams.species
      const query = await Query.loadFromLocation(this.props.location);
      log.info('updateForLocation: Query', () => pretty({query}));

      // Prepare exit behavior
      const _setRecs = ({recs}: {recs: StateRecs}): void => {
        log.info(`updateForLocation: state.recs = ${matchStateRecs(recs, {
          loading:  x    => x,
          notfound: x    => x,
          recs:     recs => `(${recs.length} recs)`,
        })}`);
        this.setState({
          query,
          recs,
          recsQueryTime: timer.time(),
        });
      };

      // Handle /rec/:sourceId not found (e.g. user deleted a user rec, or xc dataset changed)
      if (query === null) {
        return _setRecs({recs: 'notfound'});
      }

      // Global filters
      //  - TODO(put_all_query_state_in_location)
      //  - TODO(family_list): How to includeSpecies/includeSpeciesGroups?
      const qualityFilter = (table: string) => (
        sqlf`and ${SQL.raw(table)}.quality in (${this.state.filterQuality})`
      );
      const placeFilter   = (table: string) => matchNull(this.props.place, {
        null: ()    => '',
        x:    place => sqlf`and ${SQL.raw(table)}.species in (${place.species})`,
      });
      const speciesFilter = (table: string) => (
        sqlf`and ${SQL.raw(table)}.species not in (${Array.from(this.state.excludeSpecies)})`
      );
      const speciesGroupFilter = (table: string) => (
        sqlf`and ${SQL.raw(table)}.species_species_group not in (${Array.from(this.state.excludeSpeciesGroups)})`
      );
      const recFilter = (table: string) => (
        sqlf`and ${SQL.raw(table)}.source_id not in (${Array.from(this.state.excludeRecs)})`
      );

      // TODO Factor these big matchQuery cases into functions, for readability
      return _setRecs(await matchQuery<Promise<{recs: StateRecs}>>(query, {

        none: async () => {
          log.info(`updateForLocation: QueryNone -> 'notfound'...`);
          return {recs: 'notfound'};
        },

        // TODO Weight species uniformly (e.g. select random species, then select random recs)
        // TODO Get deterministic results from seed [how? sqlite doesn't support random(seed) or hash()]
        random: async ({filters, seed}) => {
          log.info(`updateForLocation: Querying random recs`, {seed});
          return await this.props.db.query<XCRec>(sqlf`
            select *
            from (
              select
                *,
                cast(taxon_order as real) as taxon_order_num
              from search_recs S
              where true
                ${SQL.raw(placeFilter('S'))}
                ${SQL.raw(qualityFilter('S'))}
                ${SQL.raw(speciesFilter('S'))}
                ${SQL.raw(speciesGroupFilter('S'))}
              order by
                random()
              limit ${this.state.n_recs}
            )
            order by
              taxon_order_num asc,
              source_id desc
          `, {
            // logTruncate: null, // XXX(family_list): Debug
          })(async results => {
            const recs = results.rows.raw();
            return {recs};
          });
        },

        // TODO(family_list): Don't render BrowseModal x/+ buttons when query.kind === 'species'
        species: async ({filters, species}) => {
          log.info('updateForLocation: Querying recs for species', {species});
          return await this.props.db.query<XCRec>(sqlf`
            select *
            from (
              select
                *,
                cast(taxon_order as real) as taxon_order_num
              from search_recs S
              where true
                and species in (${species.split(',').map(x => _.trim(x).toUpperCase())})
                ${SQL.raw(placeFilter('S'))} -- No results if selected species is outside of placeFilter
                ${SQL.raw(qualityFilter('S'))}
              order by
                source_id desc
              limit ${this.state.n_recs}
            )
            order by
              taxon_order_num asc,
              source_id desc
          `, {
            // logTruncate: null, // XXX Debug
          })(async results => {
            const recs = results.rows.raw();
            return {recs};
          });
        },

        rec: async ({filters, source}) => {
          return await log.timedAsync<{recs: StateRecs}>('updateForLocation.rec', async () => {
            log.info('updateForLocation: Loading recs for query_rec', {source});

            // Compute top n_per_sp recs per species by d_pc (cosine_distance)
            //  - Dev notes: notebooks/190226_mobile_dev_search_sqlite
            //    - Interactive dev/perf for the (complex) sql query
            //  - TODO Replace union query with windowed query after sqlite ≥3.25.x
            //    - https://github.com/andpor/react-native-sqlite-storage/issues/310
            //    - Wait for future ios version to upgrade built-in sqlite to ≥3.25.x (see github thread)
            //  - Approach w/o window functions
            //    - Query query_rec from db.search_recs
            //      - (query_rec.preds is query_sp_p (= search.predict_probs(query_rec)))
            //    - Take top n_recs/n_per_sp species from query_rec.preds
            //    - Construct big sql query with one union per species (O(n_recs/n_per_sp)):
            //      - (... where species=? order by d_pc limit n_per_sp) union all (...) ...

            // Params
            const f_preds_cols = this.state.f_preds_cols || [];
            const n_sp         = this.state.n_sp;
            const n_per_sp     = this.state.n_per_sp;
            const n_recs       = n_sp * n_per_sp + 1;

            // Load query_rec from db
            const query_rec = await this.props.db.loadRec(source);
            if (query_rec === null) {
              // query_rec not found (e.g. user deleted a user rec, or xc dataset changed)
              return {recs: 'notfound'};
            }

            // Ensure spectro exists
            //  - e.g. in case this is a user/edit rec from an old code version and the spectroCachePath's have moved
            const spectroPath = Rec.spectroPath(query_rec, this.spectroPathOpts);
            if (!await fs.exists(spectroPath)) {
              matchRec(query_rec, {
                xc:   _ => { throw `updateForLocation: Missing spectro asset for xc query_rec: ${query_rec.source_id}`; },
                user: _ => log.info(`updateForLocation: Caching spectro for user query_rec: ${query_rec.source_id}`),
              });
              await NativeSpectro.renderAudioPathToSpectroPath(
                Rec.audioPath(query_rec),
                await ensureParentDir(spectroPath),
                {
                  f_bins: this.props.f_bins,
                  denoise: true, // Like Bubo/py/model.swift:Features.denoise=true
                },
              );
            }

            // Read sp_p's (species probs) from query_rec.f_preds
            const sp_ps: Map<string, number> = new Map(zipSame(
              this.props.modelsSearch.classes_,
              query_rec.f_preds,
            ));

            // Compute slp's (species (negative) log prob) from sp_p's
            const slp = (sp_p: number): number => Math.abs(-Math.log(sp_p)) // (abs for 1->0 i/o -0)
            const slps: Map<string, number> = mapMapValues(sp_ps, slp);

            // Compute in sql: cosine_distance(S.f_preds_*, Q.f_preds_*)
            //  - cosine_distance(x,y) = 1 - dot(x,y) / norm(x) / norm(y)
            const sqlDot = (f_preds_cols
              // .slice(0, 2) // XXX Debug: smaller query
              .map(c => sqlf`S.${SQL.raw(c)}*Q.${SQL.raw(c)}`).join(' + ') || '0'
            );
            const sqlCosineDist = sqlf`
              1 - (${SQL.raw(sqlDot)}) / S.norm_f_preds / Q.norm_f_preds
            `;

            // Query which species are left after applying filters
            //  - This is difficult to make exact, but we can get very close
            //    - e.g.
            //  - TODO Perf: Cache this once per filter update i/o redoing on every rec search
            //  - NOTE Perf: If this is slow (e.g. for a place with many species), make sure it's using a covering index
            //    - Query plan should say `USING COVERING INDEX`, not `USING INDEX` (or `SCAN TABLE`)
            //    - https://www.sqlite.org/optoverview.html#covering_indices
            //    - https://www.sqlite.org/queryplanner.html#_covering_indices
            log.info('updateForLocation: Querying species for filters', rich({query_rec}));
            const filteredSpecies: Set<Species> = await this.props.db.query<{species: Species}>(sqlf`
              select distinct species
              from search_recs S
              where true
                -- Filters duplicated below (in rec query)
                ${SQL.raw(placeFilter('S'))}        -- Safe for covering index (species)
                -- ${SQL.raw(qualityFilter('S'))}   -- Unsafe for covering index (quality)
                ${SQL.raw(speciesFilter('S'))}      -- Safe for covering index (species)
                ${SQL.raw(speciesGroupFilter('S'))} -- Safe for covering index (species)
                -- ${SQL.raw(recFilter('S'))}       -- Unsafe for covering index (source_id) [why? index is (species, source_id)]
            `, {
              logTruncate: null,
              // logQueryPlan: true, // XXX Debug: ensure covering index (see above)
            })(async results => {
              const recs = results.rows.raw();
              return new Set(recs.map(rec => rec.species));
            });

            // Rank species by slp (slp asc b/c sgn(slp) ~ -sgn(sp_p))
            //  - Filter species to match rec filters, else we'll return too few rec results below
            const topSlps: Array<{species: string, slp: number}> = (
              _(Array.from(slps.entries()))
              .map(([species, slp]) => ({species, slp}))
              .filter(({species}) => filteredSpecies.has(species))
              .sortBy(({slp}) => slp)
              .slice(0, n_sp + 1) // FIXME +1 else we get n_sp-1 species -- why?
              .value()
            );

            // Construct query
            //  - Union `limit n` queries per species (b/c we don't have windowing)
            //  - Perf: We exclude .f_preds_* cols for faster load (ballpark ~2x)
            //  - TODO this.state.sortResults ('slp,d_pc' / 'd_pc')
            const non_f_preds_cols = this.state.non_f_preds_cols!; // Set in componentDidMount
            const sqlPerSpecies = (topSlps
              // .slice(0, 2) // XXX Debug: smaller query
              .map(({species, slp}) => sqlf`
                select *, ${ifNull(slp, () => 1e38)} as slp
                from S_filter_dist
                where species = ${species}
                order by d_pc asc
                limit ${n_per_sp}
              `)
            );
            const sql = sqlf`
              with
                -- For sqlCosineDist ('Q' = query_rec)
                Q as (
                  select *
                  from search_recs
                  where source_id = ${query_rec.source_id}
                  limit 1 -- Should always be ≤1, but safeguard perf in case of data bugs
                ),
                -- For nested sqlPerSpecies queries
                S_filter_dist as (
                  select
                    ${SQL.raw(non_f_preds_cols.map(x => `S.${x}`).join(', '))},
                    ${SQL.raw(sqlCosineDist)} as d_pc
                  from search_recs S
                    left join Q on true -- (1 row)
                  where true
                    -- Filters duplicated above (in species query)
                    ${SQL.raw(placeFilter('S'))}
                    ${SQL.raw(qualityFilter('S'))}
                    ${SQL.raw(speciesFilter('S'))}
                    ${SQL.raw(speciesGroupFilter('S'))}
                    ${SQL.raw(recFilter('S'))}
                    and S.source_id != ${query_rec.source_id} -- Exclude query_rec from results
                )
              select *
              from (
                -- Must wrap subqueries in 'select * from (...)' else union complains about nested order by
                ${SQL.raw(sqlPerSpecies.map(x => `select * from (${x})`).join(' union all '))}
              )
              order by
                slp asc,
                d_pc asc
              limit ${n_recs}
            `;

            // Run query
            log.info('updateForLocation: Querying recs for query_rec', rich({query_rec}));
            return await this.props.db.query<XCRec>(sql, {
              logTruncate: null, // XXX Debug (safe to always log full query, no perf concerns)
            })(async results => {
              const recs = results.rows.raw();

              // XXX(family_list): Debug
              // debug_print('timed', pretty(recs
              //   .map(rec => _.pick(rec, ['species', 'source_id', 'sp_d_pc_i', 'slp', 'd_pc']))
              //   .map(rec => yaml(rec))
              // ));

              // HACK Inject query_rec as first result so it's visible at top
              //  - TODO Replace this with a proper display of query_rec at the top
              return {recs: [query_rec, ...recs]};

            });

          });
        },

      }));

    }
  }

  releaseSounds = async () => {
    log.info(`releaseSounds: Releasing ${this.soundsCache.size} cached sounds`);
    await Promise.all(
      Array.from(this.soundsCache).map(async ([sourceId, soundAsync]) => {
        log.debug('releaseSounds: Releasing sound',
          sourceId, // Noisy (but these log lines don't de-dupe anyway when rndebugger timestamps are shown)
        );
        (await soundAsync).release();
      }),
    );
    this.soundsCache = new Map();
    this.setState({
      playing: undefined, // In case we were playing a sound, mark that we aren't anymore
    });
  }

  getOrAllocateSoundAsync = async (rec: Rec): Promise<Sound> => {
    // Is sound already allocated (in the cache)?
    let soundAsync = this.soundsCache.get(rec.source_id);
    if (!soundAsync) {
      log.debug('getOrAllocateSoundAsync: Allocating sound',
        rec.source_id, // Noisy (but these log lines don't de-dupe anyway when rndebugger timestamps are shown)
      );
      // Allocate + cache sound resource
      //  - Cache the promise so that get+set is atomic, else we race and allocate multiple sounds per rec.source_id
      //  - (Observable via log counts in the console: if num alloc > num release, then we're racing)
      this.soundsCache.set(rec.source_id, Sound.newAsync(Rec.audioPath(rec)));
      soundAsync = this.soundsCache.get(rec.source_id);
    }
    return await soundAsync!;
  }

  toggleRecPlaying = (rec: Rec) => {

    // Eagerly allocate Sound resource for rec
    //  - TODO How eagerly should we cache this? What are the cpu/mem costs and tradeoffs?
    const soundAsync = this.getOrAllocateSoundAsync(rec);

    return async (event: Gesture.TapGestureHandlerStateChangeEvent) => {
      const {nativeEvent: {state, oldState, x, absoluteX}} = event; // Unpack SyntheticEvent (before async)
      if (
        // [Mimic Gesture.BaseButton]
        oldState === Gesture.State.ACTIVE &&
        state !== Gesture.State.CANCELLED
      ) {
        log.debug('toggleRecPlaying', () => pretty({x, sourceId: rec.source_id,
          playing: this.state.playing && {sourceId: this.state.playing.rec.source_id},
        }));

        // FIXME Race conditions: tap many spectros really quickly and watch the "Playing rec" logs pile up
        //  - Maybe have to replace react-native-audio with full expo, to get Expo.Audio?
        //    - https://docs.expo.io/versions/latest/sdk/audio
        //    - https://docs.expo.io/versions/latest/sdk/av.html
        //    - https://github.com/expo/expo/tree/master/packages/expo/src/av
        //    - https://github.com/expo/expo/blob/master/packages/expo/package.json

        const {playing} = this.state;

        // Workaround: Manually clean up on done/stop b/c the .play done callback doesn't trigger on .stop
        const onDone = async () => {
          log.info('toggleRecPlaying: Done', {source_id: rec.source_id});
          timer.clearInterval(this, 'playingCurrentTime');
          this.setState({
            playing: undefined,
          });
        };

        // Stop any recs that are currently playing
        //  - TODO(stop_button): Extract a stopRecPlaying(), which looks like a bit complicated and error prone
        if (playing) {
          const {rec, sound} = playing;
          global.sound = sound; // XXX Debug

          // Stop sound playback
          log.info('toggleRecPlaying: Stopping', {source_id: rec.source_id});
          if (sound.isLoaded()) { // Else we'll hang if sound was released while playing (e.g. play -> load new search)
            await sound.stopAsync();
          }
          await onDone();

        }

        // If touched rec was the currently playing rec, then we're done (it's stopped)
        //  - Unless seekOnPlay [TODO(stop_button)]
        // Else, play the (new) touched rec
        if (
          !this.recIsPlaying(rec.source_id, playing)
          // || this.props.seekOnPlay // TODO(stop_button)
        ) {
          const sound = await soundAsync;
          global.sound = sound; // XXX Debug

          // Compute startTime to seek rec (if enabled)
          let startTime;
          if (this.props.seekOnPlay) {
            startTime = this.spectroTimeFromX(rec, x);
          } else {
            // startTime = 0; // TODO Show some kind of visual feedback when not seekOnPlay
          }

          // Play rec (if startTime is valid)
          if (!startTime || startTime < sound.getDuration()) {
            log.info('toggleRecPlaying: Playing', {source_id: rec.source_id});

            // setState
            this.setState({
              playing: {
                rec,
                sound,
                startTime,
              },
              playingCurrentTime: 0,
            });

            // Update playingCurrentTime on interval (if enabled)
            //  - HACK react-native-sound doesn't have an onProgress callback, so we have to hack it ourselves :/
            //    - Ugh, waaay slow and cpu inefficient: 16ms (60fps) kills rndebugger in Debug and pegs cpu in Release
            //    - TODO Explore alternatives: if setState->render is the bottleneck, then investigate Animated...
            //  - WARNING Don't separate timers per rec.source_id until we resolve "FIXME Race conditions" above ("tap many")
            if (this.props.playingProgressEnable && this.props.playingProgressInterval !== 0) {
              timer.setInterval(this, 'playingCurrentTime',
                async () => {
                  const {seconds, isPlaying} = await sound.getCurrentTimeAsync();
                  if (isPlaying) {
                    this.setState({
                      playingCurrentTime: seconds,
                    });
                  }
                },
                this.props.playingProgressInterval,
              );
            }

            // Seek + play + clean up when done
            //  - Don't await: .playAsync promise fulfills after playback completes (/ is stopped / fails)
            if (startTime) sound.setCurrentTime(startTime);
            finallyAsync(sound.playAsync(), async () => {
              await onDone();
            });

          }

        }

      }
    };
  }

  spectroTimeFromX = (rec: Rec, x: number): number => {
    const {contentOffset} = this._scrollViewState;
    const duration_s = rec.duration_s;
    const {width} = this.spectroDimContent(rec); // Exclude image width padding
    const {audio_s} = this.props.serverConfig.api.recs.search_recs.params;
    const time = x / width * duration_s;
    // log.debug('spectroTimeFromX', () => pretty({time, x, contentOffset, width, audio_s, duration_s}));
    return time;
  }

  spectroXFromTime = (rec: Rec, time: number): number => {
    const {contentOffset} = this._scrollViewState;
    const duration_s = rec.duration_s;
    const {width} = this.spectroDimContent(rec); // Exclude image width padding
    const {audio_s} = this.props.serverConfig.api.recs.search_recs.params;
    const x = time / duration_s * width;
    // log.debug('spectroXFromTime', () => pretty({x, time, contentOffset, width, audio_s, duration_s}));
    return x;
  }

  recIsPlaying = (sourceId: SourceId, playing: undefined | {rec: Rec}): boolean => {
    return !playing ? false : playing.rec.source_id === sourceId;
  }

  onSpectroLongPress = (rec: Rec) => async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    if (state === Gesture.State.ACTIVE) {
      this.showRecActionModal(rec);
    }
  }

  onBottomControlsLongPress = async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    await match(state,
      [Gesture.State.ACTIVE, () => this.setState({showHelp: true})],
      [Gesture.State.END,    () => this.setState({showHelp: false})],
    );
  }

  BrowseModal = () => {
    return (
      <BrowseModal
        searchFilter={this.state.searchFilter}
        showExcludeIncludeButtons={
          matchNull(this.state.query, {
            // null: ()    => false, // FIXME(family_list): Buttons flicker in 'random'/'rec' during updateForLocation()
            null: ()    => true, // HACK
            x:    query => matchQuery(query, {
              none:    () => false,
              random:  () => true,
              species: () => false,
              rec:     () => true,
            }),
          })
        }
        excludeSpecies={this.state.excludeSpecies}
        includeSpecies={this.state.includeSpecies}
        excludeSpeciesGroups={this.state.excludeSpeciesGroups}
        includeSpeciesGroups={this.state.includeSpeciesGroups}
        go={this.props.go}
        ebird={this.props.ebird}
        place={this.props.place}
        parent={this}
        stateParent={this}
      />
    );
  }

  FiltersModal = () => {
    const Separator = () => (
      <View style={{height: 5}}/>
    );
    return (
      <this.GenericModal>
        <this.GenericModalTitle title='Filters' />

        <Separator/>
        <View>
          <Text
            style={{
              ...material.body1Object,
            }}
          >
            Place: {matchNull(this.props.place, {
              null: () => 'none',
              x: ({name, props}) => `${name} (${yaml(props).slice(1, -1)})`,
            })}
          </Text>
        <View>

        <Separator/>
        <View style={{flexDirection: 'row'}}>
          <Text>
            Species: {}
          </Text>
          <TextInput
            style={{
              ...material.body1Object,
              width: 200,
              borderWidth: 1, borderColor: 'gray',
            }}
            value={this.state.filterQueryText}
            onChangeText={x => this.setState({filterQueryText: x})}
            onSubmitEditing={() => this.state.filterQueryText && (
              this.props.go('search', {path: `/species/${encodeURIComponent(this.state.filterQueryText)}`})
            )}
            autoCorrect={false}
            autoCapitalize='characters'
            enablesReturnKeyAutomatically={true}
            placeholder={this.queryDesc}
            returnKeyType='search'
          />
        </View>

        <Separator/>
        </View>
          <Text>[TODO quality]</Text>
          <Text>[TODO text search]</Text>
        </View>

      </this.GenericModal>
    );
  }

  showRecActionModal = (rec: Rec) => {
    this.setState({
      sourceIdForActionModal: rec.source_id,
      showGenericModal: () => (
        this.RecActionModal(rec)
      )
    });
  }

  RecActionModal = (rec: Rec) => {
    const Separator = () => (
      <View style={{height: 5}}/>
    );
    const defaults = {
      buttonStyle: {
        marginVertical: 1,
        marginHorizontal: 5,
        paddingVertical: 2,
        paddingHorizontal: 5,
      } as ViewStyle,
    };
    return (
      <this.GenericModal>

        <this.GenericModalTitle
          title={`${rec.species}/${Source.show(Rec.source(rec), {
            species: this.props.xc,
          })}`}
        />

        {/* Spectro */}
        <Animated.Image
          style={{
            // ...this.spectroDimImage(rec), // XXX Bad(info_modal)
            height: this.spectroDimImage(rec).height,
            width: '100%',
          }}
          foo
          resizeMode='stretch' // TODO(info_modal) Wrap to show whole spectro i/o stretching
          source={{uri: Rec.spectroPath(rec, this.spectroPathOpts)}}
        />

        <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: rec.species,
              iconName: 'search',
              buttonColor: iOSColors.blue,
              onPress: () => this.props.go('search', {path: `/species/${encodeURIComponent(rec.species)}`}),
            }, {
              ...defaults,
              label: Source.show(Rec.source(rec), {
                species: this.props.xc,
              }),
              iconName: 'search',
              buttonColor: iOSColors.blue,
              onPress: () => this.props.go('search', {path: `/rec/${encodeURIComponent(rec.source_id)}`}),
            },
          ]})}
        </View>

        <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: rec.species,
              iconName: 'x',
              buttonColor: iOSColors.red,
              onPress: () => this.setState((state: State, props: Props) => ({
                excludeSpecies: setAdd(state.excludeSpecies, rec.species),
              })),
            }, {
              ...defaults,
              label: Source.show(Rec.source(rec), {
                species: this.props.xc,
              }),
              iconName: 'x',
              buttonColor: iOSColors.red,
              onPress: () => this.setState((state: State, props: Props) => ({
                excludeRecs: setAdd(state.excludeRecs, rec.source_id),
              })),
            }
          ]})}
        </View>

        <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: rec.species_species_group,
              iconName: 'x',
              buttonColor: iOSColors.red,
              onPress: () => this.setState((state: State, props: Props) => ({
                excludeSpeciesGroups: setAdd(state.excludeSpeciesGroups, rec.species_species_group),
              })),
            }
          ]})}
        </View>

        <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: 'Edit',
              iconName: 'edit',
              buttonColor: iOSColors.purple,
              onPress: () => this.props.go('record', {path: `/edit/${encodeURIComponent(rec.source_id)}`}),
            }
          ]})}
        </View>

        {/* TODO(more_results) Put these in the footer of the ScrollView */}
        {/* <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: 'More species',
              iconName: 'plus-circle',
              buttonColor: iOSColors.purple,
              onPress: () => {},
            }, {
              ...defaults,
              label: 'Fewer species',
              iconName: 'minus-circle',
              buttonColor: iOSColors.purple,
              onPress: () => {},
            },
          ]})}
        </View>
        <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: 'More recs per species',
              iconName: 'plus-circle',
              buttonColor: iOSColors.purple,
              onPress: () => {},
            }, {
              ...defaults,
              label: 'Fewer recs per species',
              iconName: 'minus-circle',
              buttonColor: iOSColors.purple,
              onPress: () => {},
            },
          ]})}
        </View>
        <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: 'Add a species manually',
              iconName: 'plus-circle',
              buttonColor: iOSColors.purple,
              onPress: () => {},
            },
          ]})}
        </View> */}

        {/* TODO(saved_lists) */}
        {/* <Separator/>
        {this.ActionModalButtons({actions: [
          {
            ...defaults,
            label: `Save to list (${rec.species})`,
            iconName: 'bookmark',
            buttonColor: iOSColors.orange,
            onPress: () => {},
          }, {
            ...defaults,
            label: `Save to list (${Source.show(Rec.source(rec), {species: this.props.xc})})`,
            iconName: 'bookmark',
            buttonColor: iOSColors.orange,
            onPress: () => {},
          }, {
            ...defaults,
            label: 'Save all to new list',
            iconName: 'bookmark',
            buttonColor: iOSColors.orange,
            onPress: () => {},
          }, {
            ...defaults,
            label: 'Add all to existing list',
            iconName: 'bookmark',
            buttonColor: iOSColors.orange,
            onPress: () => {},
          },
        ]})} */}

        {/* TODO(share_results) */}
        {/* <Separator/>
        {this.ActionModalButtons({actions: [
          {
            ...defaults,
            label: 'Share list',
            iconName: 'share',
            buttonColor: iOSColors.green,
            onPress: () => {},
          },
        ]})} */}

        <Separator/>
        <View style={{
          // width: Dimensions.get('window').width, // Fit within the left-most screen width of ScrollView content
          flexDirection: 'column',
          // // borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: iOSColors.black, // TODO Make full width
          // marginTop: 3,
          // // marginBottom: 3,
        }}>
          {/* Ignore invalid keys. Show in order of MetadataColumnsLeft. */}
          {objectKeysTyped(MetadataColumnsBelow).map(c => this.props.metadataColumnsBelow.includes(c) && (
            <MetadataText
              key={c}
              style={{
                marginBottom: 3,
              }}
            >
              <MetadataLabel col={c} /> {MetadataColumnsBelow[c](rec)}
            </MetadataText>
          ))}
        </View>

      </this.GenericModal>
    );
  }

  // [Scratch] Scroll SectionList so that same ~top recs are showing after drawing with new item/section heights
  //  - TODO More experimentation needed
  // requestAnimationFrame(() => {
  //   if (this.scrollViewRef.current) {
  //     this.scrollViewRef.current.scrollToLocation({
  //       animated: false,
  //       sectionIndex: 3, itemIndex: 3, // TODO Calculate real values to restore
  //       viewPosition: 0, // 0: top, .5: middle, 1: bottom
  //     });
  //   }
  // });

  scaleSpectros = async (delta: number) => {
    this.setState((state, props) => {
      // Round current _spectroScale so that +1/-1 deltas snap back to non-fractional scales (e.g. after pinch zooms)
      const _spectroScale = this.clampSpectroScaleY(Math.round(state._spectroScale) + delta);
      return {
        _spectroScale,
        scrollViewState: {
          // FIXME Zoom in -> scroll far down+right -> use '-' button to zoom out -> scroll view clipped b/c contentOffset nonzero
          contentOffset: {
            x: this._scrollViewState.contentOffset.x * _spectroScale / state._spectroScale,
            y: this._scrollViewState.contentOffset.y * _spectroScale / state._spectroScale,
          },
        },
      };
    });
  }

  clampSpectroScaleY = (spectroScale: number): number => _.clamp(
    spectroScale,
    this.props.spectroScaleClamp.min,
    this.props.spectroScaleClamp.max,
  );

  BottomControls = (props: {}) => (
    <View style={styles.bottomControls}>
      {/* Browse */}
      <this.BottomControlsButton
        help='Browse'
        iconProps={{name: 'list'}}
        onPress={() => this.setState({
          showGenericModal: () => this.BrowseModal(),
        })}
      />
      {/* Filters */}
      <this.BottomControlsButton
        help='Filters'
        iconProps={{name: 'filter'}}
        onPress={() => this.setState({
          showGenericModal: () => this.FiltersModal()
        })}
      />
      {/* Toggle sort: species probs / rec dist */}
      <this.BottomControlsButton
        help='Sort'
        iconProps={{name: 'chevrons-down'}}
        // iconProps={{name: 'chevron-down'}}
        // iconProps={{name: 'arrow-down'}}
        // iconProps={{name: 'arrow-down-circle'}}
        onPress={() => this.setState({
          showGenericModal: () => (
            <this.ActionModal title='Sort' actions={[
              // this.state.queryRec ? [ // TODO queryRec
              {
                label: 'Species match, then similar recs',
                iconName: 'chevrons-down',
                textColor: iOSColors.black,
                buttonColor: this.state.sortResults === 'slp,d_pc' ? iOSColors.orange : iOSColors.customGray,
                onPress: () => this.setState({
                  sortResults: 'slp,d_pc',
                })
              }, {
                label: 'Similar recs only (ignore species match)',
                iconName: 'chevrons-down',
                textColor: iOSColors.black,
                buttonColor: this.state.sortResults === 'd_pc' ? iOSColors.orange : iOSColors.customGray,
                onPress: () => this.setState({
                  sortResults: 'd_pc',
                })
              },
            ]} />
          )
        })}
      />
      {/* Query that returns no results [XXX For dev] */}
      <this.BottomControlsButton
        help='Blank'
        iconProps={{name: 'power'}}
        onPress={() => this.props.go('search', {path: `/species/${encodeURIComponent('_BLANK')}`})} // HACK No results via junk species
      />
      {/* Random recs */}
      <this.BottomControlsButton
        help='Random'
        // iconProps={{name: 'refresh-ccw'}}
        iconProps={{name: 'shuffle'}}
        onPress={() => this.props.go('search', {path: this.randomPath()})}
      />
      {/* Toggle metadata: left */}
      <this.BottomControlsButton
        help='Info'
        active={!this.props.showMetadataBelow && this.props.showMetadataLeft}
        iconProps={{
          // name: 'file-minus',
          name: 'sidebar',
          style: (!this.props.showMetadataBelow ? {} : {
            color: iOSColors.gray,
          }),
        }}
        onPress={() => !this.props.showMetadataBelow && (
          this.props.settings.toggle('showMetadataLeft')
        )}
        onLongPress={() => !this.props.showMetadataBelow && this.setState({
          showGenericModal: () => (
            <this.ActionModal title='Show columns' actions={
              objectKeysTyped(MetadataColumnsLeft).map(c => ({
                label: metadataLabel(c),
                textColor: iOSColors.black,
                buttonColor: this.props.metadataColumnsLeft.includes(c) ? iOSColors.tealBlue : iOSColors.customGray,
                buttonStyle: {
                  marginVertical:  2, // Compact so we can fit many buttons
                  paddingVertical: 5, // Compact so we can fit many buttons
                },
                dismiss: false,
                onPress: () => this.props.settings.update('metadataColumnsLeft', cs => (
                  (cs.includes(c) ? _.without(cs, c) : [...cs, c])
                  .filter(c => c in MetadataColumnsLeft) // Clean up invalid keys
                )),
              }))
            } />
          )
        })}
      />
      {/* Toggle metadata: below */}
      <this.BottomControlsButton
        help='Info'
        active={this.props.showMetadataBelow}
        // iconProps={{name: 'file-text'}}
        iconProps={{name: 'credit-card', style: Styles.flipVertical}}
        // iconProps={{name: 'sidebar', style: Styles.rotate270}}
        onPress={() => this.props.settings.toggle('showMetadataBelow')}
        onLongPress={() => this.setState({
          showGenericModal: () => (
            <this.ActionModal title='Show columns' actions={
              objectKeysTyped(MetadataColumnsBelow).map(c => ({
                label: metadataLabel(c),
                textColor: iOSColors.black,
                buttonColor: this.props.metadataColumnsBelow.includes(c) ? iOSColors.tealBlue : iOSColors.customGray,
                buttonStyle: {
                  marginVertical:  2, // Compact so we can fit many buttons
                  paddingVertical: 5, // Compact so we can fit many buttons
                },
                dismiss: false,
                onPress: () => this.props.settings.update('metadataColumnsBelow', cs => (
                  (cs.includes(c) ? _.without(cs, c) : [...cs, c])
                  .filter(c => c in MetadataColumnsBelow) // Clean up invalid keys
                )),
              }))
            } />
          )
        })}
      />
      {/* Toggle seekOnPlay crosshairs */}
      <this.BottomControlsButton
        help='Seek'
        active={this.props.playOnTap && this.props.seekOnPlay}
        iconProps={{
          ...(this.props.playOnTap ? {
            name: 'crosshair',
          } : {
            name: 'slash',
            style: {color: iOSColors.red},
          }),
        }}
        onPress={async () => {
          var {playOnTap, seekOnPlay} = this.props;
          [playOnTap, seekOnPlay] = (
            playOnTap && seekOnPlay  ? [true,  false] : // blue  -> black
            playOnTap && !seekOnPlay ? [false, false] : // black -> gray
                                       [true,  true]    // gray  -> blue
          );
          await this.props.settings.set({seekOnPlay, playOnTap});
        }}
      />
      {/* Zoom more/fewer recs (spectro height) */}
      {/* - TODO Disable when spectroScale is min/max */}
      <this.BottomControlsButton
        help='Dense'
        disabled={this.state._spectroScale === this.props.spectroScaleClamp.min}
        // iconProps={{name: 'align-justify'}} // 4 horizontal lines
        iconProps={{name: 'zoom-out'}}
        onPress={async () => await this.scaleSpectros(-1)}
      />
      <this.BottomControlsButton
        help='Tall'
        disabled={this.state._spectroScale === this.props.spectroScaleClamp.max}
        // iconProps={{name: 'menu'}}          // 3 horizontal lines
        iconProps={{name: 'zoom-in'}}
        onPress={async () => await this.scaleSpectros(+1)}
      />
    </View>
  );

  BottomControlsButton = (props: {
    help: string,
    iconProps: IconProps,
    onPress?: (pointerInside: boolean) => void,
    onLongPress?: () => void,
    active?: boolean,
    disabled?: boolean,
  }) => {
    const {style: iconStyle, ...iconProps} = props.iconProps;
    return (
      <LongPressGestureHandler
        onHandlerStateChange={event => (
          props.onLongPress
            ? event.nativeEvent.state === Gesture.State.ACTIVE && props.onLongPress()
            : this.onBottomControlsLongPress(event)
        )}
      >
        <BorderlessButton
          style={styles.bottomControlsButton}
          onPress={props.disabled ? undefined : props.onPress}
        >
          {this.state.showHelp && (
            <Text style={styles.bottomControlsButtonHelp}>{props.help}</Text>
          )}
          <Feather
            style={[
              styles.bottomControlsButtonIcon,
              iconStyle,
              (
                props.disabled ? {color: iOSColors.gray} :
                props.active   ? {color: iOSColors.blue} :
                {}
              ),
            ]}
            {...iconProps}
          />
        </BorderlessButton>
      </LongPressGestureHandler>
    );
  }

  // (Unused, keeping for reference)
  // sectionsForRecs = (recs: Array<Rec>): Array<SectionListData<Rec>> => {
  //   const sections = [];
  //   let section;
  //   for (let rec of recs) {
  //     const title = rec.species;
  //     if (!section || title !== section.title) {
  //       if (section) sections.push(section);
  //       section = {
  //         title,
  //         data: [] as Rec[],
  //         species: rec.species,
  //         species_taxon_order: rec.species_taxon_order,
  //         species_com_name: rec.species_com_name,
  //         species_sci_name: rec.species_sci_name,
  //         recs_for_sp: rec.recs_for_sp,
  //       };
  //     }
  //     section.data.push(rec);
  //   }
  //   if (section) sections.push(section);
  //   return sections;
  // }

  RecEditingButtons = (props: {rec: Rec}) => (
    <View style={styles.recEditingView}>
      {this._recEditingButtons.map((f, i) => f(i, props.rec))}
    </View>
  );

  get _recEditingButtons() {
    return [

      (i: number, rec: Rec) => (
        <this.RecEditingButton
          key={i}
          iconName='move'
          onPress={() => {}}
        />
      ),

    ];
  }

  RecEditingButton = (props: {
    buttonStyle?: Style,
    iconStyle?: Style,
    iconName: string,
    onPress: (pointerInside: boolean) => void,
  }) => (
    <BorderlessButton
      style={[styles.recEditingButton, props.buttonStyle]}
      onPress={props.onPress}
    >
      <Feather
        style={[styles.recEditingIcon, props.iconStyle,
          // Compact icon to fit within tiny rows
          this.state._spectroScale >= 2 ? {} : {
            fontSize: this.state._spectroScale / 2 * material.headlineObject.fontSize!,
            lineHeight: this.state._spectroScale / 2 * material.headlineObject.lineHeight!,
          },
        ]}
        name={props.iconName}
      />
    </BorderlessButton>
  );

  GenericModal = (props: {
    style?: ViewStyle,
    children: ReactNode,
    // onDismiss?: () => void, // TODO Add this [requires more coupling with ActionModalButtons, which also does dismiss]
  }) => (
    // Background overlay: semi-transparent background + tap outside modal to dismiss
    <BaseButton
      onPress={() => this.setState({
        showGenericModal: null, // Dismiss modal
      })}
      style={{
        width: '100%', height: '100%', // Full screen
        backgroundColor: `${iOSColors.black}88`, // Semi-transparent overlay
        justifyContent: 'center', alignItems: 'center', // (vertical, horizontal)
      }}
    >
      {/* Modal */}
      <View style={{
        backgroundColor: iOSColors.white,
        padding: 15,
        ...props.style,
      }}>
        {/* TODO When keyboard shown and tap is outside of modal, don't dismiss modal along with keyboard */}
        <KeyboardDismissingView>
          {props.children}
        </KeyboardDismissingView>
      </View>
    </BaseButton>
  );

  GenericModalTitle = (props: {
    title: string,
    viewStyle?: ViewStyle,
    textStyle?: TextStyle,
  }) => (
    <View style={{
      ...props.viewStyle,
    }}>
      <Text style={{
        alignSelf: 'center', // (horizontal)
        marginBottom: 5,
        ...material.titleObject,
        ...props.textStyle,
      }}>
        {props.title}
      </Text>
    </View>
  );

  ActionModal = (props: {
    title: string,
    titleStyle?: TextStyle,
    actions: Array<{
      label: string,
      iconName?: string,
      buttonColor?: string,
      textColor?: string,
      buttonStyle?: ViewStyle,
      dismiss?: boolean,
      onPress: () => void,
    }>,
  }) => (
    <this.GenericModal>
      <this.GenericModalTitle textStyle={props.titleStyle} title={props.title} />
      {this.ActionModalButtons({actions: props.actions})}
    </this.GenericModal>
  );

  ActionModalButtons = (props: {
    actions: Array<{
      label: string,
      iconName?: string,
      buttonColor?: string,
      textColor?: string,
      buttonStyle?: ViewStyle,
      dismiss?: boolean,
      onPress: () => void,
    }>,
  }) => (
    props.actions.map(({
      label,
      iconName,
      buttonColor,
      textColor,
      buttonStyle,
      dismiss,
      onPress,
    }, i) => (
      <RectButton
        key={i}
        style={{
          // flex:             1, // Makes everything big
          flexDirection:    'row',
          alignItems:       'center',
          padding:          10,
          marginHorizontal: 10,
          marginVertical:   2,
          backgroundColor:  _.defaultTo(buttonColor, iOSColors.customGray),
          ..._.defaultTo(buttonStyle, {}),
        }}
        onPress={() => {
          if (_.defaultTo(dismiss, true)) {
            this.setState({
              showGenericModal: null, // Dismiss modal
            });
          }
          onPress();
        }}
      >
        {iconName && (
          <Feather
            style={{
              // ...material.headlineObject,
              ...material.buttonObject,
              marginRight: 5,
              color: _.defaultTo(textColor, iOSColors.white),
            }}
            name={iconName}
          />
        )}
        <Text
          style={{
            // ...material.buttonObject,
            ...material.body2Object,
            color: _.defaultTo(textColor, iOSColors.white),
          }}
          children={label}
        />
      </RectButton>
    ))
  );

  // Assign color sequentially to species
  //  - Ensures no collisions unless specieses.length > color.length
  stylesForSpecies = (specieses: Array<string>, styles: Array<LabelStyle> = labelStyles): Map<string, LabelStyle> => {
    return new Map(specieses.map<[string, LabelStyle]>((species, i) => (
      [species, styles[i % styles.length]]
    )));
  }

  // Assign colors randomly to species
  stylesForSpeciesHash = (specieses: Array<string>, styles: Array<LabelStyle> = labelStyles): Map<string, LabelStyle> => {
    return new Map(specieses.map<[string, LabelStyle]>(species => (
      [species, styles[stringHash(species) % styles.length]]
    )));
  }

  ModalsAndActionSheets = () => (
    <View>
      <Modal
        visible={!!this.state.showGenericModal}
        animationType='none' // 'none' | 'slide' | 'fade'
        transparent={true}
        children={this.state.showGenericModal && this.state.showGenericModal()}
      />
      {/* (Unused, kept for reference)
      <ActionSheetBasic
        innerRef={this.sortActionSheet}
        options={[
          ['Sort by species, then by recs', () => {}],
          ['Sort by recs only',             () => {}],
          ['Sort recs by similarity',       () => {}],
          ['Order manually',                () => {}],
        ]}
      />
      */}
    </View>
  );

  // Map props/state to a DOM node
  //  - Render phase (pure, no read/write DOM, may be called multiple times per commit or interrupted)
  render = () => {
    log.info('render');
    const styleForSpecies = this.stylesForSpecies(_.uniq(this.recsOrEmpty.map(rec => rec.species)));
    return (
      <View style={{
        flex: 1,
      }}>

        {/* Redirect: '/' -> '/random/:seed' */}
        <Route exact path='/' render={() => (
          <Redirect to={this.randomPath()} />
        )}/>

        {/* Loading spinner */}
        {this.state.recs === 'loading' && (
          <View style={{
            flex: 1,
            justifyContent: 'center',
          }}>
            <ActivityIndicator size='large' />
          </View>
        )}

        {/* Recs list (with pan/pinch) */}
        {/* - We use ScrollView instead of SectionList to avoid _lots_ of opaque pinch-to-zoom bugs */}
        {/* - We use ScrollView instead of manual gestures (react-native-gesture-handler) to avoid _lots_ of opaque animation bugs */}
        {this.state.recs !== 'loading' && (
          <ScrollView
            ref={this.scrollViewRef}

            // Scroll/zoom
            //  - Force re-layout on zoom change, else bad things (that I don't understand)
            key={this.state.scrollViewKey}
            contentContainerStyle={{
              // ScrollView needs manually computed width to scroll in overflow direction (horizontal)
              //  - https://github.com/facebook/react-native/issues/8579#issuecomment-233162695
              width: this.scrollViewContentWidth,
            }}
            // This is (currently) the only place we use state.scrollViewState i/o this._scrollViewState
            contentOffset={tap(this.state.scrollViewState.contentOffset, x => {
              // log.debug('render.contentOffset', {x}); // XXX Debug
            })}
            bounces={false}
            bouncesZoom={false}
            directionalLockEnabled={true} // Don't scroll vertical and horizontal at the same time (ios only)
            minimumZoomScale={this.props.spectroScaleClamp.min / this.state._spectroScale}
            maximumZoomScale={this.props.spectroScaleClamp.max / this.state._spectroScale}
            onScrollEndDrag={async ({nativeEvent}) => {
              // log.debug('onScrollEndDrag', {nativeEvent}); // XXX Debug
              const {contentOffset, zoomScale, velocity} = nativeEvent;
              this._scrollViewState = {contentOffset};
              if (
                zoomScale !== 1              // Don't trigger zoom if no zooming happened (e.g. only scrolling)
                // && velocity !== undefined // [XXX Unreliable] Don't trigger zoom on 1/2 fingers released, wait for 2/2
              ) {
                const scale = zoomScale * this.state._spectroScale;
                // log.debug('ZOOM', {nativeEvent}); // XXX Debug
                // Trigger re-layout so non-image components (e.g. text) redraw at non-zoomed size
                this.setState({
                  scrollViewState: this._scrollViewState,
                  _spectroScale: this.clampSpectroScaleY(scale),
                  scrollViewKey: chance.hash(), // Else bad things (that I don't understand)
                });
              }
            }}

            // TODO Sticky headers: manually calculate indices of species header rows
            // stickyHeaderIndices={!this.props.showMetadataBelow ? undefined : ...}

            // TODO Add footer with "Load more" button
            //  - Mimic SectionList.ListFooterComponent [https://facebook.github.io/react-native/docs/sectionlist#listfootercomponent]

          >
            {/* Mimic a FlatList */}

            {/*
            // (Unused, keeping for reference)
            // _.flatten(this.sectionsForRecs(this.state.recs).map(({
            //   title,
            //   data: recs,
            //   species,
            //   species_taxon_order,
            //   species_com_name,
            //   species_sci_name,
            //   recs_for_sp,
            // }, sectionIndex) => [
            //
            //   Species header
            //   this.props.showMetadataBelow && (
            //     <View
            //       key={`section-${sectionIndex}-${title}`}
            //       style={styles.sectionSpecies}
            //     >
            //       <Text numberOfLines={1} style={styles.sectionSpeciesText}>
            //         {species_com_name} (<Text style={{fontStyle: 'italic'}}>{species_sci_name}</Text>)
            //       </Text>
            //       {this.props.showDebug && (
            //         // FIXME Off screen unless zoom=1
            //         <this.DebugText numberOfLines={1} style={[{marginLeft: 'auto', alignSelf: 'center'}]}>
            //           ({recs_for_sp} recs)
            //         </this.DebugText>
            //       )}
            //     </View>
            //   ),
            */}

            {/* Rec rows */}
            <View style={{flex: 1}}>
              {this.state.recs === 'notfound' ? (

                <View style={[Styles.center, {padding: 30,
                  width: Dimensions.get('window').width, // HACK Fix width else we drift right with scrollViewContentWidth
                }]}>
                  <Text style={material.subheading}>
                    Recording not found
                  </Text>
                </View>

              ) : fastIsEqual(this.state.recs, []) ? (

                <View style={[Styles.center, {padding: 30,
                  width: Dimensions.get('window').width, // HACK Fix width else we drift right with scrollViewContentWidth
                }]}>
                  <Text style={material.subheading}>
                    No results
                  </Text>
                </View>

              ) : (
                this.state.recs.map((rec, recIndex) => [

                  // Rec row (with editing buttons)
                  <Animated.View
                    key={`row-${recIndex}-${rec.source_id}`}
                    style={{
                      flex: 1, flexDirection: 'row',
                      // Alternating row colors
                      // backgroundColor: recIndex % 2 == 0 ? iOSColors.white : iOSColors.lightGray,
                      // Compact controls/labels when zoom makes image smaller than controls/labels
                      ...(this.props.showMetadataBelow ? {} : {
                        height: this.spectroDimImage(rec).height,
                      }),
                    }}
                  >

                    {/* Rec editing buttons */}
                    {/* - NOTE Condition duplicated in scrollViewContentWidths */}
                    {this.props.editing && (
                      <this.RecEditingButtons rec={rec} />
                    )}

                    {/* Rec region without the editing buttons  */}
                    <LongPressGestureHandler onHandlerStateChange={this.onSpectroLongPress(rec)}>
                      <Animated.View style={{
                        flex: 1, flexDirection: 'column',
                      }}>

                        {/* Rec row */}
                        <View
                          style={{
                            flexDirection: 'row',
                            ...(this.props.showMetadataBelow ? {} : {
                              // Compact controls/labels when zoom makes image smaller than controls/labels
                              height: this.spectroDimImage(rec).height,
                            }),
                          }}
                        >

                          {/* Rec debug info */}
                          {this.props.showMetadataLeft && (
                            <this.DebugView style={{
                              padding: 0, // Reset padding:3 from debugView
                              width: this.scrollViewContentWidths.debugInfo,
                            }}>
                              <MetadataText style={Styles.debugText}>slp: {rec.slp && round(rec.slp, 2)}</MetadataText>
                              <MetadataText style={Styles.debugText}>d_pc: {rec.d_pc && round(rec.d_pc, 2)}</MetadataText>
                              <MetadataText style={Styles.debugText}>n_recs: {rec.recs_for_sp}</MetadataText>
                            </this.DebugView>
                          )}

                          {/* Rec metadata left */}
                          {this.props.showMetadataLeft && !this.props.showMetadataBelow && (
                            <View style={{
                              flexDirection: 'column',
                              width: this.scrollViewContentWidths.metadataLeft,
                              borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: iOSColors.midGray,
                            }}>
                              {/* Ignore invalid keys. Show in order of MetadataColumnsLeft. */}
                              {objectKeysTyped(MetadataColumnsLeft).map(c => this.props.metadataColumnsLeft.includes(c) && (
                                <MetadataText key={c} children={MetadataColumnsLeft[c](rec)} />
                              ))}
                            </View>
                          )}

                          {/* Sideways species label */}
                          {/* - After controls/metadata so that label+spectro always abut (e.g. if scrolled all the way to the right) */}
                          {/* - NOTE Keep outside of TapGestureHandler else spectroTimeFromX/spectroXFromTime have to adjust */}
                          <View style={[styles.recSpeciesSidewaysView, {
                            backgroundColor: styleForSpecies.get(rec.species)!.backgroundColor,
                          }]}>
                            <View style={styles.recSpeciesSidewaysViewInner}>
                              <Text numberOfLines={1} style={[styles.recSpeciesSidewaysText, {
                                fontSize: this.state._spectroScale >= 2 ? 11 : 6, // Compact species label to fit within tiny rows
                                color: styleForSpecies.get(rec.species)!.color,
                              }]}>
                                {rec.species}
                              </Text>
                            </View>
                          </View>

                          {/* Spectro (tap) */}
                          <TapGestureHandler onHandlerStateChange={
                            // Toggle play/pause normally, but show modal if playOnTap is disabled
                            //  - UX HACK to allow a faster workflow for hiding lots of families/species/recs in a row
                            this.props.playOnTap ? this.toggleRecPlaying(rec) : ev => this.showRecActionModal(rec)
                          }>
                            <Animated.View>

                              {/* Image */}
                              <Animated.Image
                                style={this.spectroDimImage(rec)}
                                resizeMode='stretch'
                                source={{uri: Rec.spectroPath(rec, this.spectroPathOpts)}}
                              />

                              {/* Start time cursor (if playing + startTime) */}
                              {this.recIsPlaying(rec.source_id, this.state.playing) && (
                                this.state.playing!.startTime && (
                                  <View style={{
                                    position: 'absolute', width: 1, height: '100%',
                                    left: this.spectroXFromTime(this.state.playing!.rec, this.state.playing!.startTime!),
                                    backgroundColor: iOSColors.gray,
                                  }}/>
                                )
                              )}

                              {/* Progress time cursor (if playing + playingCurrentTime) */}
                              {this.recIsPlaying(rec.source_id, this.state.playing) && (
                                this.state.playing!.startTime && this.state.playingCurrentTime !== undefined && (
                                  <View style={{
                                    position: 'absolute', width: 1, height: '100%',
                                    left: this.spectroXFromTime(this.state.playing!.rec, this.state.playingCurrentTime),
                                    backgroundColor: iOSColors.black,
                                  }}/>
                                )
                              )}

                              {/* HACK Visual feedback for playing rec [XXX after adding progress bar by default] */}
                              {this.recIsPlaying(rec.source_id, this.state.playing) && (
                                <View style={{
                                  position: 'absolute', height: '100%', width: 5,
                                  left: 0,
                                  backgroundColor: iOSColors.red,
                                }}/>
                              )}

                              {/* HACK Visual feedback for long-press ActionModal rec */}
                              {/* - HACK Condition on showGenericModal b/c we can't (yet) onDismiss to unset sourceIdForActionModal */}
                              {this.state.showGenericModal && this.state.sourceIdForActionModal === rec.source_id && (
                                <View style={{
                                  position: 'absolute', height: '100%', width: 5,
                                  left: 0,
                                  backgroundColor: iOSColors.black,
                                }}/>
                              )}

                            </Animated.View>
                          </TapGestureHandler>

                        </View>

                        {/* Rec metadata below */}
                        {this.props.showMetadataBelow && (
                          <View style={{
                            width: Dimensions.get('window').width, // Fit within the left-most screen width of ScrollView content
                            flexDirection: 'column',
                            // borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: iOSColors.black, // TODO Make full width
                            marginTop: 3,
                            // marginBottom: 3,
                          }}>
                            {/* Ignore invalid keys. Show in order of MetadataColumnsLeft. */}
                            {objectKeysTyped(MetadataColumnsBelow).map(c => this.props.metadataColumnsBelow.includes(c) && (
                              <MetadataText
                                key={c}
                                style={{
                                  marginBottom: 3,
                                }}
                              >
                                <MetadataLabel col={c} /> {MetadataColumnsBelow[c](rec)}
                              </MetadataText>
                            ))}
                          </View>
                        )}

                      </Animated.View>
                    </LongPressGestureHandler>

                  </Animated.View>

                ])
              )}
            </View>

            {/* Footer */}
            <View style={{
              ...Styles.center,
              width: Dimensions.get('window').width, // HACK Fix width else we drift right with scrollViewContentWidth
              paddingVertical: 5,
              flexDirection: 'row',
            }}>
              {/* Add more results */}
              <BorderlessButton style={{marginHorizontal: 5}} onPress={() => this.edit_n_sp(+1)}>
                <Feather
                  style={styles.bottomControlsButtonIcon}
                  name='plus'
                />
              </BorderlessButton>
              <BorderlessButton style={{marginHorizontal: 5}} onPress={() => this.edit_n_sp(-1)}>
                <Feather
                  style={styles.bottomControlsButtonIcon}
                  name='minus'
                />
              </BorderlessButton>
              {/* Add more results per sp */}
              {/* - TODO Move to per sp (use rec long press) */}
              <BorderlessButton style={{marginHorizontal: 5}} onPress={() => this.edit_n_per_sp(+1)}>
                <Feather
                  style={styles.bottomControlsButtonIcon}
                  name='plus-circle'
                />
              </BorderlessButton>
              <BorderlessButton style={{marginHorizontal: 5}} onPress={() => this.edit_n_per_sp(-1)}>
                <Feather
                  style={styles.bottomControlsButtonIcon}
                  name='minus-circle'
                />
              </BorderlessButton>
            </View>

          </ScrollView>
        )}

        {/* Debug info */}
        <this.DebugView>
          <this.DebugText>queryDesc: {this.queryDesc}</this.DebugText>
          <this.DebugText>
            Recs: {typeof this.state.recs === 'string'
              ? `.../${this.state.totalRecs || '?'}`
              : `${this.state.recs.length}/${this.state.totalRecs || '?'} (${sprintf('%.3f', this.state.recsQueryTime)}s)`
            }
          </this.DebugText>
          <this.DebugText>Filters: {yaml(this.filters)}</this.DebugText>
        </this.DebugView>

        {/* Bottom controls */}
        <this.BottomControls/>

        {/* Modals + action sheets */}
        <this.ModalsAndActionSheets/>

      </View>
    );
  }

  // Debug components
  //  - [Tried and gave up once to make well-typed generic version of these (DebugFoo = addStyle(Foo, ...) = withProps(Foo, ...))]
  DebugView = (props: RN.ViewProps & {children: any}) => (
    !this.props.showDebug ? null : (
      <View {...{
        ...props,
        style: [Styles.debugView, ...normalizeStyle(props.style)],
      }}/>
    )
  );
  DebugText = (props: RN.TextProps & {children: any}) => (
    !this.props.showDebug ? null : (
      <Text {...{
        ...props,
        style: [Styles.debugText, ...normalizeStyle(props.style)],
      }}/>
    )
  );

}

//
// BrowseModal
//

interface BrowseModalProps {
  searchFilter:              string; // State lifted up into SearchScreen, so it persists across close/reopen
  showExcludeIncludeButtons: boolean;
  excludeSpecies:            State['excludeSpecies'];
  includeSpecies:            State['includeSpecies'];
  excludeSpeciesGroups:      State['excludeSpeciesGroups'];
  includeSpeciesGroups:      State['includeSpeciesGroups'];
  go:                        Props['go'];
  ebird:                     Props['ebird'];
  place:                     Props['place'];
  parent:                    SearchScreen;
  stateParent:               SearchScreen;
}

interface BrowseModalState {
}

export class BrowseModal extends PureComponent<BrowseModalProps, BrowseModalState> {

  log = new Log('BrowseModal');

  state = {
  };

  // Getters
  parent = this.props.parent;

  // Refs
  sectionListRef: RefObject<SectionList<Rec>> = React.createRef();

  // State
  _firstSectionHeaderHeight: number = 0; // For SectionList.scrollToLocation({viewOffset})

  componentDidMount = () => {
    this.log.info('componentDidMount');
    global.BrowseModal = this; // XXX Debug
  };

  componentWillUnmount = () => {
    this.log.info('componentWillUnmount');
  };

  componentDidUpdate = (prevProps: BrowseModalProps, prevState: BrowseModalState) => {
    // Noisy (in xcode)
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  };

  render = () => {
    // this.log.info('render'); // Debug

    // Perf: lots of redundant computation here, but this isn't the bottleneck
    //  - Bottleneck is render for SectionList -> section/item components
    const matchesSearchFilter: (metadata: SpeciesMetadata) => boolean = log.timed('matchesSearchFilter', () => {
      const tokenize = (v: string): string[] => v.toLowerCase().replace(/[^a-z ]+/, '').split(' ').filter(x => !_.isEmpty(x));
      const searches = this.props.searchFilter.split('/').map(search => tokenize(search)).filter(x => !_.isEmpty(x));
      const ks: Array<keyof SpeciesMetadata> = [
        'shorthand',
        'sci_name',
        'com_name',
        'species_code',
        'species_group',
        // 'family', // XXX Confusing because field not visible to user (can't see why it matched)
        // 'order',  // XXX Confusing because field not visible to user (can't see why it matched)
      ];
      return (metadata: SpeciesMetadata): boolean => {
        const vs = _.flatMap(ks, k => tokenize(metadata[k]));
        return _.isEmpty(searches) || _.some(searches, search => _.every(search, term => _.some(vs, v => v.includes(term))));
      };
    });

    // Precompute sections so we can figure out various indexes
    type Section = SectionListData<SpeciesMetadata>;
    const data = log.timed('data', () => typed<SpeciesMetadata[]>(_.sortBy(
      matchNull(this.props.place, {
        null: () => [],
        x: place => (
          _(place.species)
          .flatMap(species => matchUndefined(this.props.ebird.speciesMetadataFromSpecies.get(species), {
            undefined: () => [],
            x:         m  => [m],
          }))
          .filter(m => matchesSearchFilter(m))
          .value()
        ),
      }),
      m => parseFloat(m.taxon_order),
    )));
    const sections: Array<Section> = log.timed('sections', () => (
      _(data)
      .groupBy(m => m.species_group)
      .entries().map(([title, data]) => ({title, data}))
      .value()
    ));
    const firstSection   = _.head(sections);
    const lastSection    = _.last(sections);
    const isFirstSection = (section: Section) => firstSection && section.title === firstSection.title;
    const isLastSection  = (section: Section) => lastSection  && section.title === lastSection.title;
    const isLastItem     = (section: Section, index: number) => isLastSection(section) && index === section.data.length - 1;
    const nSpeciesShown  = _.sum(sections.map(x => x.data.length));

    return (
      <View style={{
        width: '100%', height: '100%', // Full screen
        backgroundColor: iOSColors.white, // Opaque overlay (else SearchScreen shows through)
        flexDirection: 'column',
      }}>

        {/* Title + scroll to top + close button */}
        <BaseButton
          style={{
            // flex: 1,
          }}
          onPress={() => {
            // Scroll to top
            mapNull(this.sectionListRef.current, sectionList => { // Avoid transient nulls [why do they happen?]
              if (sectionList.scrollToLocation) { // (Why typed as undefined? I think only for old versions of react-native?)
                sectionList.scrollToLocation({
                  sectionIndex: 0, itemIndex: 0,              // First section, first item
                  viewOffset: this._firstSectionHeaderHeight, // Else first item covered by first section header
                });
              }
            });
          }}
        >
          <View style={{
            flexDirection:     'row',
            alignItems:        'center', // Vertical (row i/o column)
            backgroundColor:   Styles.tabBar.backgroundColor,
            borderBottomWidth: Styles.tabBar.borderTopWidth,
            borderBottomColor: Styles.tabBar.borderTopColor,
          }}>

            {/* Title */}
            <Text style={{
              flexGrow: 1,
              ...material.body2Object,
              marginHorizontal: 5,
            }}>
              {matchNull(this.props.place, {
                x:    place => `${place.name} (${nSpeciesShown}/${place.species.length} species)`,
                null: ()    => '(No place selected)',
              })}
            </Text>

            {/* Close button */}
            <RectButton
              style={{
                justifyContent: 'center', // Vertical
                alignItems:     'center', // Horizontal
                width:          35,
                height:         35,
              }}
              onPress={() => {
                // Dismiss modal
                this.props.stateParent.setState({
                  showGenericModal: null,
                });
              }}
            >
              <Feather style={{
                // ...material.titleObject,
                ...material.headlineObject,
              }}
                // name={'check'}
                name={'x'}
              />
            </RectButton>

          </View>
        </BaseButton>

        {/* Search bar */}
        <SearchBar
          // Listeners
          onSearchChange={searchFilter => this.props.stateParent.setState({
            searchFilter,
          })}
          // Style
          height={40}
          padding={0}
          inputStyle={{
            // Disable border from SearchBar (styles.searchBar)
            borderWidth:       0,
            // Replace with a border that matches the title bar border
            backgroundColor:   Styles.tabBar.backgroundColor,
            borderBottomWidth: Styles.tabBar.borderTopWidth,
            borderBottomColor: Styles.tabBar.borderTopColor,
          }}
          // Disable back button
          //  - By: always showing back button and making it look and behave like the search icon
          alwaysShowBackButton={true}
          iconBackName='md-search'
          onBackPress={() => {}}
          // TextInputProps
          inputProps={{
            autoCorrect:                   false,
            autoCapitalize:                'none',
            // enablesReturnKeyAutomatically: true,
            placeholder:                   'Species',
            defaultValue:                  this.props.searchFilter,
            returnKeyType:                 'done',
            selectTextOnFocus:             true,
            keyboardType:                  'default',
          }}
          // TODO Prevent dismissing keyboard on X button, so that it only clears the input
          iconCloseComponent={(<View/>)} // Disable close button [TODO Nope, keep so we can easily clear text]
          // onClose={() => this.props.stateParent.setState({searchFilter: ''})} // FIXME Why doesn't this work?
        />

        {/* SectionList */}
        <SectionList
          ref={this.sectionListRef as any} // HACK Is typing for SectionList busted? Can't make it work
          style={{
            flexGrow: 1,
          }}
          sections={sections}
          // Disable lazy loading, else fast scrolling down hits a lot of partial bottoms before the real bottom
          initialNumToRender={data.length}
          maxToRenderPerBatch={data.length}
          keyExtractor={species => species.shorthand} // [Why needed in addition to key props below? key warning without this]
          ListEmptyComponent={(
            <View style={[Styles.center, {padding: 30}]}>
              <Text style={material.subheading}>
                No species
              </Text>
            </View>
          )}
          // Perf: split out section/item components so that it stays mounted across updates
          //  - (Why does it sometimes unmount/mount anyway?)
          renderSectionHeader={({section}) => {
            const {species_group} = section.data[0];
            return (
              <BrowseModalSectionHeader
                key={species_group}
                species_group={species_group}
                showExcludeIncludeButtons={this.props.showExcludeIncludeButtons}
                isFirstSection={isFirstSection(section)}
                excluded={this.props.excludeSpeciesGroups.has(species_group)}
                included={this.props.includeSpeciesGroups.has(species_group)}
                parent={this}
                stateParent={this.props.stateParent}
              />
            );
          }}
          renderItem={({item: species, index, section}) => {
            return (
              <BrowseModalItem
                key={species.shothand}
                species={species} // WARNING Perf: this will trigger many unnecessary updates if object identity ever changes
                showExcludeIncludeButtons={this.props.showExcludeIncludeButtons}
                isLastItem={isLastItem(section, index)}
                excluded={( // exc || group exc && !inc
                  this.props.excludeSpecies.has(species.shorthand) || (
                    this.props.excludeSpeciesGroups.has(species.species_group) &&
                    !this.props.includeSpecies.has(species.shorthand)
                  )
                )}
                included={( // inc || group inc && !exc
                  this.props.includeSpecies.has(species.shorthand) || (
                    this.props.includeSpeciesGroups.has(species.species_group) &&
                    !this.props.excludeSpecies.has(species.shorthand)
                  )
                )}
                parent={this}
                stateParent={this.props.stateParent}
              />
            );
          }}
        />
      </View>
    );

  };

  onFirstSectionHeaderLayout = async (event: LayoutChangeEvent) => {
    const {nativeEvent: {layout: {x, y, width, height}}} = event; // Unpack SyntheticEvent (before async)
    this._firstSectionHeaderHeight = height;
  }

}

function BrowseItemButton(props: {
  iconName:          string,
  activeButtonColor: string,
  active:            boolean,
  onPress:           () => void,
}) {
  return (
    <RectButton
      style={{
        backgroundColor:  !props.active ? iOSColors.gray : props.activeButtonColor,
        justifyContent:   'center',
        alignItems:       'center',
        width:            30,
        height:           30,
        borderRadius:     15,
        marginHorizontal: 2,
      }}
      onPress={props.onPress}
    >
        <Feather style={{
          ...material.buttonObject,
          color: iOSColors.white,
        }}
        name={props.iconName}
      />
    </RectButton>
  );
}

//
// BrowseModalSectionHeader
//  - Split out component so that it stays mounted across updates
//

interface BrowseModalSectionHeaderProps {
  species_group:             string;
  showExcludeIncludeButtons: boolean;
  isFirstSection:            boolean | undefined;
  excluded:                  boolean;
  included:                  boolean;
  parent:                    BrowseModal;
  stateParent:               SearchScreen;
}

interface BrowseModalSectionHeaderState {
}

export class BrowseModalSectionHeader extends PureComponent<BrowseModalSectionHeaderProps, BrowseModalSectionHeaderState> {

  log = new Log(`BrowseModalSectionHeader[${this.props.species_group}]`);

  state = {
  };

  // Getters
  parent = this.props.parent;

  componentDidMount = () => {
    this.log.info('componentDidMount');
  };

  componentWillUnmount = () => {
    this.log.info('componentWillUnmount');
  };

  componentDidUpdate = (prevProps: BrowseModalSectionHeaderProps, prevState: BrowseModalSectionHeaderState) => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  };

  render = () => {
    // this.log.info('render'); // Debug
    const {species_group} = this.props;
    return (
      <View
        style={[Styles.fill, {
          flexDirection:   'row',
          justifyContent:  'center',
          alignItems:      'center',
          paddingVertical: 3,
          backgroundColor: iOSColors.lightGray,
        }]}
        // For SectionList.scrollToLocation({viewOffset})
        onLayout={!this.props.isFirstSection ? undefined : this.parent.onFirstSectionHeaderLayout}
      >

        <Text style={{
          flexGrow: 1,
          paddingHorizontal: 5,
          ...material.captionObject,
          fontWeight: 'bold',
          color:      '#444444',
        }}>{species_group}</Text>

        {this.props.showExcludeIncludeButtons && (
          <BrowseItemButton
            iconName='x'
            activeButtonColor={iOSColors.red}
            active={this.props.excluded}
            onPress={() => {
              this.props.stateParent.setState((state, props) => {
                const x = species_group;
                const {excludeSpeciesGroups: exc, includeSpeciesGroups: inc} = state;
                return {
                  excludeSpeciesGroups: !exc.has(x) ? setAdd    (exc, x) : setDelete(exc, x),
                  includeSpeciesGroups: !exc.has(x) ? setDelete (inc, x) : inc,
                };
              });
            }}
          />
        )}

        {this.props.showExcludeIncludeButtons && (
          <BrowseItemButton
            iconName='plus'
            activeButtonColor={iOSColors.green}
            active={this.props.included}
            onPress={() => {
              this.props.stateParent.setState((state, props) => {
                const x = species_group;
                const {excludeSpeciesGroups: exc, includeSpeciesGroups: inc} = state;
                return {
                  includeSpeciesGroups: !inc.has(x) ? setAdd    (inc, x) : setDelete(inc, x),
                  excludeSpeciesGroups: !inc.has(x) ? setDelete (exc, x) : exc,
                };
              });
            }}
          />
        )}

      </View>
    );
  };

}

//
// BrowseModalItem
//  - Split out component so that it stays mounted across updates
//

interface BrowseModalItemProps {
  species:                   SpeciesMetadata; // WARNING Perf: this will trigger many unnecessary updates if object identity ever changes
  showExcludeIncludeButtons: boolean;
  isLastItem:                boolean | undefined;
  excluded:                  boolean;
  included:                  boolean;
  parent:                    BrowseModal;
  stateParent:               SearchScreen;
}

interface BrowseModalItemState {
}

export class BrowseModalItem extends PureComponent<BrowseModalItemProps, BrowseModalItemState> {

  log = new Log(`BrowseModalItem[${this.props.species.species_group}/${this.props.species.shorthand}]`);

  state = {
  };

  // Getters
  parent = this.props.parent;

  componentDidMount = () => {
    this.log.info('componentDidMount');
  };

  componentWillUnmount = () => {
    this.log.info('componentWillUnmount');
  };

  componentDidUpdate = (prevProps: BrowseModalItemProps, prevState: BrowseModalItemState) => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  };

  render = () => {
    // this.log.info('render'); // Debug
    const {species} = this.props;
    return (
      <View style={{
        flexDirection:   'row',
        justifyContent:  'center',
        alignItems:      'center',
        paddingVertical: 3,
        // Vertical borders
        //  - Internal borders: top border on non-first items per section
        //  - Plus bottom border on last item of last section
        borderTopWidth: 1,
        borderTopColor: iOSColors.lightGray,
        ...(!this.props.isLastItem ? {} : {
          borderBottomWidth: 1,
          borderBottomColor: iOSColors.lightGray,
        }),
      }}>

        <BrowseItemButton
          iconName='search'
          activeButtonColor={iOSColors.blue}
          active={true}
          onPress={() => {
            // Dismiss modal
            this.props.stateParent.setState({
              showGenericModal: null,
            });
            // Show species
            this.parent.props.go('search', {path: `/species/${encodeURIComponent(species.shorthand)}`});
          }}
        />

        <View style={{
          flexGrow: 1,
          paddingHorizontal: 5,
        }}>
          <Text style={[material.captionObject, {color: 'black'}]}>
            {species.com_name}
          </Text>
          <Text style={[material.captionObject, {fontSize: 10}]}>
            {species.sci_name}
          </Text>
        </View>

        {this.props.showExcludeIncludeButtons && (
          <BrowseItemButton
            iconName='x'
            activeButtonColor={iOSColors.red}
            active={this.props.excluded}
            onPress={() => {
              this.props.stateParent.setState((state, props) => {
                const x = species.shorthand;
                const {excludeSpecies: exc, includeSpecies: inc} = state;
                return {
                  excludeSpecies: !exc.has(x) ? setAdd    (exc, x) : setDelete(exc, x),
                  includeSpecies: !exc.has(x) ? setDelete (inc, x) : inc,
                };
              });
            }}
          />
        )}

        {this.props.showExcludeIncludeButtons && (
          <BrowseItemButton
            iconName='plus'
            activeButtonColor={iOSColors.green}
            active={this.props.included}
            onPress={() => {
              this.props.stateParent.setState((state, props) => {
                const x = species.shorthand;
                const {excludeSpecies: exc, includeSpecies: inc} = state;
                return {
                  includeSpecies: !inc.has(x) ? setAdd    (inc, x) : setDelete(inc, x),
                  excludeSpecies: !inc.has(x) ? setDelete (exc, x) : exc,
                };
              });
            }}
          />
        )}

      </View>
    );
  };

}

//
// KeyboardDismissingView
//

interface KeyboardDismissingViewProps extends RN.ViewProps {
}

interface KeyboardDismissingViewState {
  isKeyboardShown: boolean;
}

// TODO Why is this slow to respond after keyboard shows? -- adding logging to find the bottleneck
export class KeyboardDismissingView extends PureComponent<KeyboardDismissingViewProps, KeyboardDismissingViewState> {

  log = new Log('KeyboardDismissingView');

  state = {
    isKeyboardShown: false,
  };

  // State
  _keyboardDidShowListener?: {remove: () => void};
  _keyboardDidHideListener?: {remove: () => void};

  componentDidMount = () => {
    this.log.info('componentDidMount');
    this._keyboardDidShowListener = Keyboard.addListener('keyboardDidShow', this.keyboardDidShow);
    this._keyboardDidHideListener = Keyboard.addListener('keyboardDidHide', this.keyboardDidHide);
  };

  componentWillUnmount = () => {
    this.log.info('componentWillUnmount');
    this._keyboardDidShowListener!.remove();
    this._keyboardDidHideListener!.remove();
  };

  componentDidUpdate = (prevProps: KeyboardDismissingViewProps, prevState: KeyboardDismissingViewState) => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  };

  keyboardDidShow = () => this.setState({isKeyboardShown: true});
  keyboardDidHide = () => this.setState({isKeyboardShown: false});

  render = () => (
    <TapGestureHandler
      enabled={this.state.isKeyboardShown}
      onHandlerStateChange={({nativeEvent: {state}}) => Keyboard.dismiss()}
      // TODO Need to condition on state?
      // onHandlerStateChange={({nativeEvent: {state}}) => state === Gesture.State.ACTIVE && Keyboard.dismiss()}
    >
      <Animated.View {...this.props} />
    </TapGestureHandler>
  );

}

// XXX
// function KeyboardDismissingView(props: RN.ViewProps) {
//   return (
//     <TapGestureHandler
//       onHandlerStateChange={({nativeEvent: {state}}) => state === Gesture.State.ACTIVE && Keyboard.dismiss()}
//     >
//       <Animated.View {...props} />
//     </TapGestureHandler>
//   );
// }

//
// styles
//

const styles = StyleSheet.create({
  filtersModal: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 300,
    backgroundColor: iOSColors.green,
  },
  bottomControls: {
    flexDirection: 'row',
    height: 48, // Approx tab bar height (see TabRoutes.TabBarStyle)
    alignItems: 'center',
    paddingVertical: 5,
    backgroundColor: iOSColors.midGray,
  },
  bottomControlsButton: {
    flex: 1,
    alignItems: 'center',
  },
  bottomControlsButtonIcon: {
    ...material.headlineObject,
  },
  bottomControlsButtonHelp: {
    ...material.captionObject,
  },
  summaryText: {
    ...material.captionObject,
  },
  sectionSpecies: {
    flexDirection: 'row',
    // ...material.body1Object, backgroundColor: iOSColors.customGray, // Black on white
    ...material.body1WhiteObject, backgroundColor: iOSColors.gray, // White on black
  },
  sectionSpeciesText: {
    alignSelf: 'center', // Align text vertically
  },
  recSpeciesSidewaysView: {
    justifyContent: 'center',        // Else sideways text is to the above
    alignItems: 'center',            // Else sideways text is to the right
    width: sidewaysTextWidth,        // HACK Manually shrink outer view width to match height of sideways text
    zIndex: 1,                       // Over spectro image
  },
  recSpeciesSidewaysViewInner: {     // View>View>Text else the text aligment doesn't work
    transform: [{rotate: '270deg'}], // Rotate text sideways
    width: 100,                      // Else text is clipped to outer view's (smaller) width
  },
  recSpeciesSidewaysText: {
    alignSelf: 'center',             // Else sideways text is to the bottom
    // fontSize: ...,                // Set dynamically
    // ...material.captionObject,    // (Sticking with default color:'black')
  },
  recEditingView: {
    flexDirection: 'row',
    zIndex: 1, // Over spectro image
  },
  recEditingButton: {
    width: recEditingButtonWidth, // Need explicit width (i/o flex:1) else view shows with width:0
    justifyContent: 'center', // Align icon vertically
    backgroundColor: iOSColors.midGray,
  },
  recEditingIcon: {
    // ...material.titleObject,
    ...material.headlineObject,
    alignSelf: 'center', // Align icon horizontally
  },
});
