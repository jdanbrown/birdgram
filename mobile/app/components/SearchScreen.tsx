import * as d3sc from 'd3-scale-chromatic';
import dedent from 'dedent';
import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import RN, {
  ActivityIndicator, Animated, Dimensions, FlatList, FlexStyle, GestureResponderEvent, Image, ImageStyle, Keyboard,
  KeyboardAvoidingView, LayoutChangeEvent, Modal, Platform, RegisteredStyle, ScrollView, SectionList, SectionListData,
  StyleProp, Text, TextInput, TextStyle, TouchableHighlight, View, ViewStyle,
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
import timer from 'react-native-timer';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { IconProps } from 'react-native-vector-icons/Icon';
import Feather from 'react-native-vector-icons/Feather';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import { Link, matchPath, Redirect, Route, Switch } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
import stringHash from "string-hash";
const fs = RNFB.fs;

import { App, AppProps, AppState } from 'app/App';
import * as Colors from 'app/colors';
import { ActionSheetBasic } from 'app/components/ActionSheets';
import {
  MetadataColumnBelow, MetadataColumnsBelow, MetadataColumnLeft, MetadataColumnsLeft, metadataLabel, MetadataLabel,
  MetadataText,
} from 'app/components/MetadataColumns';
import { TabBarStyle } from 'app/components/TabRoutes';
import { HelpText, TitleBar, TitleBarWithHelp } from 'app/components/TitleBar';
import { config } from 'app/config';
import {
  ModelsSearch, matchRec, matchSearchPathParams, Place, Quality, Rec, SearchPathParams, searchPathParamsFromLocation,
  SearchRecs, ServerConfig, Source, SourceId, Species, SpeciesGroup, SpeciesMetadata, SpectroPathOpts, UserRec, XCRec,
} from 'app/datatypes';
import { DB } from 'app/db';
import { Ebird } from 'app/ebird';
import { debug_print, Log, logErrors, logErrorsAsync, puts, rich, tap } from 'app/log';
import { NativeSearch } from 'app/native/Search';
import { NativeSpectro } from 'app/native/Spectro';
import { Go, History, Location, locationKeyIsEqual, locationPathIsEqual, TabName } from 'app/router';
import { SettingsWrites } from 'app/settings';
import Sound from 'app/sound';
import { SQL, sqlf } from 'app/sql';
import { StyleSheet } from 'app/stylesheet';
import { normalizeStyle, LabelStyle, labelStyles, Styles } from 'app/styles';
import {
  all, any, assert, chance, Clamp, Dim, ensureParentDir, fastIsEqual, finallyAsync, getOrSet, global, ifEmpty, ifNull,
  into, json, local, mapMapValues, mapNull, mapUndefined, match, matchEmpty, matchKey, matchNull, matchUndefined,
  noawait, objectKeysTyped, Omit, Point, pretty, QueryString, recursively, round, setAdd, setDiff, setToggle,
  shallowDiffPropsState, showDate, Sign, Style, throw_, Timer, typed, yaml, yamlPretty, zipSame,
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
export type Query = QueryNone | QueryRandom | QuerySpeciesGroup | QuerySpecies | QueryRec | QueryCompare;
export type QueryNone         = {kind: 'none'}; // e.g. so we can show nothing on redirect from '/'
export type QueryRandom       = {kind: 'random',        filters: Filters, seed: number};
export type QuerySpeciesGroup = {kind: 'species_group', filters: Filters, species_group: string};
export type QuerySpecies      = {kind: 'species',       filters: Filters, species: string};
export type QueryRec          = {kind: 'rec',           filters: Filters, source: Source};
export type QueryCompare      = {kind: 'compare',       filters: Filters, queries: Array<Query>};
export function matchQuery<X>(query: Query, cases: {
  none:          (query: QueryNone)         => X,
  random:        (query: QueryRandom)       => X,
  species_group: (query: QuerySpeciesGroup) => X,
  species:       (query: QuerySpecies)      => X,
  rec:           (query: QueryRec)          => X,
  compare:       (query: QueryCompare)      => X,
}): X {
  switch (query.kind) {
    case 'none':          return cases.none(query);
    case 'random':        return cases.random(query);
    case 'species_group': return cases.species_group(query);
    case 'species':       return cases.species(query);
    case 'rec':           return cases.rec(query);
    case 'compare':       return cases.compare(query);
  }
}

export const Query = {

  // null if source not found
  //  - (Callers: updateForLocation, RecentScreen, SavedScreen)

  loadFromLocation: async (location: Location): Promise<Query | null> => {
    const searchPathParams = searchPathParamsFromLocation(location);
    return await Query.loadFromSearchPathParams(searchPathParams);
  },

  loadFromSearchPathParams: async (searchPathParams: SearchPathParams): Promise<Query | null> => {
    return await matchSearchPathParams<Promise<Query | null>>(searchPathParams, {
      root:          async ()                          => ({kind: 'species',       filters: {}, species: ''}),
      random:        async ({filters, seed})           => ({kind: 'random',        filters, seed}),
      species_group: async ({filters, species_group})  => ({kind: 'species_group', filters, species_group}),
      species:       async ({filters, species})        => ({kind: 'species',       filters, species}),
      rec:           async ({filters, sourceId})       => {
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
      compare: async ({filters, searchPathParamss}) => ({
        kind: 'compare',
        filters,
        queries: await Promise.all(searchPathParamss.map(async searchPathParams => (
          ifNull<Query>(
            await Query.loadFromSearchPathParams(searchPathParams),
            () => ({kind: 'none'}), // HACK What UX makes sense for nested nulls? / Should they even happen here?
          )
        ))),
      }),
    });
  },

};

// TODO(put_all_query_state_in_location)
export interface Filters {
  // quality?: Array<Quality>;
  text?:    string; // TODO(text_filter)
}

export const Filters = {

  fromQueryString: (q: QueryString): Filters => ({
    // HACK Typing
    // quality: _.get(q, 'quality', '').split(',').filter(x => (Quality.values as Array<string>).includes(x)) as Array<Quality>,
    text:    _.get(q, 'text'),
  }),

  toQueryString: (x: Filters): QueryString => _.pickBy({
    // quality: (x.quality || []).join(','),
    text:    x.text,
  }, (v, k) => v !== undefined) as {[key: string]: string} // HACK Typing

};

export type SortListResults =
  | 'species_then_random'
  | 'random'
  | 'xc_id'
  | 'month_day'
  | 'date'
  | 'lat'
  | 'lng'
  | 'country__state'
  | 'quality'

export type SortSearchResults =
  | 'slp__d_pc'
  | 'd_pc'

//
// SearchScreen
//

interface Props {
  // App globals
  visible:                 boolean; // Manual visible/dirty to avoid background updates
  serverConfig:            ServerConfig;
  modelsSearch:            ModelsSearch;
  location:                Location;
  history:                 History;
  go:                      Go;
  xc:                      XC;
  ebird:                   Ebird;
  db:                      DB;
  app:                     App;
  // Settings
  settings:                SettingsWrites;
  showHelp:                boolean;
  showDebug:               boolean;
  n_per_sp:                number;
  n_recs:                  number;
  range_n_per_sp:          Array<number>;
  range_n_recs:            Array<number>;
  filterQuality:           Set<Quality>;
  sortListResults:         SortListResults;
  sortSearchResults:       SortSearchResults;
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
  place:                   Place;
  excludeSpecies:          Set<Species>;
  excludeSpeciesGroups:    Set<SpeciesGroup>;
  unexcludeSpecies:        Set<Species>;
  // SearchScreen
  iconForTab:              {[key in TabName]: string};
  f_bins:                  number;
  spectroBase:             Dim<number>;
  spectroScaleClamp:       Clamp<number>;
  searchRecsMaxDurationS:  number;
}

interface State {
  dirtyUpdateForLocation: boolean;
  scrollViewKey: string;
  scrollViewState: ScrollViewState;
  showGenericModal: null | (() => ReactNode);
  totalRecs?: number;
  non_f_preds_cols?: Array<string>;
  f_preds_cols?: Array<string>;
  f_preds_col?: (i: number) => string; // More robust alternative to f_preds_cols[i]
  query: null | Query;
  refreshQuery: boolean; // TODO(put_all_query_state_in_location)
  // TODO Persist filters with settings
  //  - Top-level fields instead of nested object so we can use state merging when updating them in isolation
  filterQueryText?: string;
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

type StateRecs = StateRecsLoading | StateRecsNotFound | StateRecsRecs | StateRecsCompare;
type StateRecsLoading  = {kind: 'loading'};
type StateRecsNotFound = {kind: 'notfound'};
type StateRecsRecs     = {kind: 'recs',    recs:  Array<Rec>};
type StateRecsCompare  = {kind: 'compare', recss: Array<StateRecs>};
function matchStateRecs<X>(recs: StateRecs, cases: {
  loading:  (recs: StateRecsLoading)  => X,
  notfound: (recs: StateRecsNotFound) => X,
  recs:     (recs: StateRecsRecs)     => X,
  compare:  (recs: StateRecsCompare)  => X,
}): X {
  switch (recs.kind) {
    case 'loading':  return cases.loading(recs);
    case 'notfound': return cases.notfound(recs);
    case 'recs':     return cases.recs(recs);
    case 'compare':  return cases.compare(recs);
  }
}

export class SearchScreen extends PureComponent<Props, State> {

  static defaultProps = {
    range_n_per_sp:         _.range(1, 100 + 1),
    range_n_recs:           [10, 25, 50, 100, 250], // Stop at 250 b/c no build currently has >250 recs per sp
    spectroBase:            {height: 20, width: Dimensions.get('window').width},
    spectroScaleClamp:      {min: 1, max: 8},
    searchRecsMaxDurationS: 10.031,  // HACK Query max(search_recs.duration_s) from db on startup
  };

  // Else we have to do too many setState's, which makes animations jump (e.g. ScrollView momentum)
  _scrollViewState: ScrollViewState = {
    contentOffset: {x: 0, y: 0},
  };

  state: State = {
    dirtyUpdateForLocation: false,
    scrollViewKey:          '',
    scrollViewState:        this._scrollViewState,
    showGenericModal:       null,
    // showGenericModal:       () => this.FiltersModal(), // XXX Debug
    // showGenericModal:       () => this.SortModal(), // XXX Debug
    query:                  null,
    refreshQuery:           false,
    excludeRecs:            new Set(),
    recs:                   {kind: 'loading'},
    _spectroScale:          this.props.spectroScale, // Sync from/to Settings (2/3)
  };

  // Getters for props
  get spectroPathOpts(): SpectroPathOpts { return {
    f_bins:  this.props.f_bins, // Higher res for user recs, ignored for xc recs (which are all f_bins=40)
    denoise: true,              // For predict (like Bubo/py/model.swift:Features.denoise=true)
  }}

  get queryRecOrNull(): null | Rec {
    return recursively(this.state.recs, (recs, recur) => {
      return matchStateRecs(recs, {
        loading:  ()        => null,
        notfound: ()        => null,
        recs:     ({recs})  => recs[0] || null, // null if empty recs
        compare:  ({recss}) => recss.map(recur)[0] || null, // HACK
      });
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
  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
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
      const f_preds_col      = (i: number) => `f_preds_${i}`;
      f_preds_cols = _.range(f_preds_cols.length).map(i => f_preds_col(i)); // Reconstruct array to ensure ordering
      this.setState({
        non_f_preds_cols,
        f_preds_cols,
        f_preds_col,
      });
    });

    // Propagate state.excludeRecs.length -> App.state.nExcludeRecs
    await this.propagateExcludeRecsToApp(null, null);

    // Show this.props.location
    await this.updateForLocation(null, null);

  });

  // Before a component is removed from the DOM and destroyed
  //  - Commit phase (impure, may read/write DOM, called once per commit)
  //  - Best practices
  //    - Do unsubscribe listeners / cancel timers (created in componentDidMount)
  //    - Don't setState(), since no more render() will happen for this instance
  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    log.info('componentWillUnmount');

    // Tell other apps we're no longer using the audio device
    Sound.setActive(false);

    // Release cached sound resources
    await this.releaseSounds();

    // Clear timers
    timer.clearTimeout(this);

  });

  // After props/state change; not called for the initial render()
  //  - Commit phase (impure, may read/write DOM, called once per commit)
  //  - Best practices
  //    - Do operate on DOM in response to changed props/state
  //    - Do fetch data, conditioned on changed props/state (else update loops)
  //    - Do setState(), conditioned on changed props (else update loops)
  componentDidUpdate = async (prevProps: Props, prevState: State) => logErrorsAsync('componentDidUpdate', async () => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));

    // Reset view state if location changed
    //  - TODO Pass props.key to reset _all_ state? [https://reactjs.org/blog/2018/06/07/you-probably-dont-need-derived-state.html#recap]
    if (!fastIsEqual(this.props.location, prevProps.location)) {
      log.info('componentDidUpdate: Reset view state');
      this.setState({
        filterQueryText:      undefined,
        // TODO Confusing UX: excludeRecs resets on each search, but exclude/include species/groups persists
        //  - If we switch to persisting excludeRecs, then we need to also make it more visible/editable somehow
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

    // Propagate state.excludeRecs.length -> App.state.nExcludeRecs
    await this.propagateExcludeRecsToApp(prevProps, prevState);

    // Show this.props.location
    await this.updateForLocation(prevProps, prevState);

  });

  // Make these both randomSpeciesPath i/o randomRecsPath because randomSpeciesPath is surprisingly more fun
  //  - Also, randomRecsPath is really slow (TODO(slow_random)), so avoid it at least for defaultPath (else slow first UX)
  defaultPath = (): string => this.randomSpeciesPath(); // On a blank path, e.g. when the app loads for the first time
  shufflePath = (): string => this.randomSpeciesPath(); // On the 'shuffle' button

  randomSpeciesPath = (): string => {
    const {props} = this;
    // HACK Mimic sql filters in updateForLocation ("Global filters")
    const metadata = (
      _(props.ebird.allSpeciesMetadata)
      // placeFilter
      .filter(m => props.place.knownSpecies.includes(m.shorthand))
      // speciesFilter
      .filter(m => !props.excludeSpecies.has(m.shorthand))
      // speciesGroupFilter
      .filter(m => !props.excludeSpeciesGroups.has(m.species_group) || props.unexcludeSpecies.has(m.shorthand))
      .value()
    );
    const species = metadata.map(x => x.shorthand);
    return `/species/${encodeURIComponent(chance.pickone(species))}`;
  }

  randomRecsPath = (seed?: number): string => {
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
      matchNull(this.queryRecOrNull, {null: () => -Infinity, x: x => x.duration_s}),
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

  edit_n_per_sp = (sign: Sign) => this._edit_n(sign, 'n_per_sp', 'range_n_per_sp');
  edit_n_recs   = (sign: Sign) => this._edit_n(sign, 'n_recs',   'range_n_recs');

  _edit_n = async (
    sign:    Sign,
    k:       'n_per_sp'       | 'n_recs',
    range_k: 'range_n_per_sp' | 'range_n_recs',
  ): Promise<void> => {
    await this.props.settings.set(settings => ({
      [k]: this._next_n(sign, settings[k], this.props[range_k]),
    }));
  }

  _next_n = (sign: Sign, n: number, ns: Array<number>): number => {
    ns = _.sortBy(ns); // (i/o Array.sort, which sorts by .toString)
    // Find insertion index of n within ns
    //  - "Insertion index" so we're robust to n not being in ns
    //  - If n is in ns, then insertion index is just index
    const i = match<number>(_.findIndex(ns, m => n <= m),
      [-1,            i => ns.length], // Not found
      [match.default, i => i],         // Found
    );
    // Return number at -1/+1 index relative to n
    return ns[_.clamp(i + sign, 0, ns.length - 1)];
  }

  get queryDesc(): string {
    return recursively(this.state.query, (query, recur) => {
      return matchNull(query, {
        null: ()    => '...',
        x:    query => matchQuery(query, {
          none:          ()                         => 'none',
          random:        ({filters, seed})          => `random/${seed}`,
          species_group: ({filters, species_group}) => species_group,
          species:       ({filters, species})       => species,
          rec:           ({filters, source})        => Source.show(source, {species: this.props.xc}),
          compare:       ({filters, queries})       => _.join(queries.map(recur), '|'),
        }),
      });
    });
  }

  // HACK A circuitous way to make state.excludeRecs.length available in App.badgeForTab
  propagateExcludeRecsToApp = async (prevProps: null | Props, prevState: null | State) => {
    const prevNExcludeRecs = mapNull(prevState, x => x.excludeRecs.size);
    const nExcludeRecs     = this.state.excludeRecs.size;
    if (prevNExcludeRecs !== nExcludeRecs) {
      this.props.app.setState({
        nExcludeRecs,
      });
    }
  }

  updateForLocation = async (prevProps: null | Props, prevState: null | State) => {
    const {visible} = this.props;
    const dirty     = this.state.dirtyUpdateForLocation;
    log.debug('updateForLocation', () => (prevProps === null && prevState === null
      ? rich({prevProps, prevState, props: '[OMITTED]', state: '[OMITTED]'}) // null->{props,state} is very noisy (e.g. xcode logs)
      : rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state))
    ));
    if (
      !(
        // Don't noop if we're visible and dirty (i.e. we just became visible, and some props/state changed in the meantime)
        visible && dirty ||
        // Don't noop if any filters/limits changed [XXX(put_all_query_state_in_location)]
        //  - Use refreshQuery as a proxy for various filters/limits changing
        this.state.refreshQuery ||
        //  - And test other filters/limits here
        !fastIsEqual(this.props.place,                _.get(prevProps, 'place')) ||
        !fastIsEqual(this.props.excludeSpecies,       _.get(prevProps, 'excludeSpecies')) ||
        !fastIsEqual(this.props.excludeSpeciesGroups, _.get(prevProps, 'excludeSpeciesGroups')) ||
        !fastIsEqual(this.props.unexcludeSpecies,     _.get(prevProps, 'unexcludeSpecies')) ||
        !fastIsEqual(this.props.n_per_sp,             _.get(prevProps, 'n_per_sp')) ||
        !fastIsEqual(this.props.n_recs,               _.get(prevProps, 'n_recs')) ||
        !fastIsEqual(this.props.filterQuality,        _.get(prevProps, 'filterQuality')) ||
        !fastIsEqual(this.props.sortListResults,      _.get(prevProps, 'sortListResults')) ||
        !fastIsEqual(this.props.sortSearchResults,    _.get(prevProps, 'sortSearchResults')) ||
        !fastIsEqual(this.state.excludeRecs,          _.get(prevState, 'excludeRecs'))
      ) && (
        // Noop if location didn't change
        locationPathIsEqual(this.props.location, _.get(prevProps, 'location')) ||
        // Noop if we don't know f_preds_cols (et al.) yet (assume we'll be called again once we do)
        !this.state.f_preds_cols ||
        !this.state.f_preds_col
      )
    ) {
      log.info('updateForLocation: Skipping');
    } else {
      const timer = new Timer();

      // Manual visible/dirty to avoid background updates (type 2: updateFor*)
      //  - If some props/state changed, mark dirty
      if (!visible && !dirty) this.setState({dirtyUpdateForLocation: true});
      //  - If we just became visible, mark undirty
      if (visible && dirty) this.setState({dirtyUpdateForLocation: false});
      //  - If we aren't visible, noop
      if (!visible) return;

      // Set loading state
      //  - TODO Fade previous recs instead of showing a blank screen while loading
      log.info("updateForLocation: state.recs = 'loading'");
      this.setState({
        query: null,
        recs: {kind: 'loading'},
        refreshQuery: false,
      });
      await this.releaseSounds(); // (Safe to do after clearing state.recs, since it uses this.soundsCache)

      // Load location -> query
      //  - We ignore state.filterQueryText b/c TextInput.onSubmitEditing -> history.push -> navParams.species
      const query = await Query.loadFromLocation(this.props.location);
      log.info('updateForLocation: Query', () => pretty({query}));

      // Load query -> recs
      const recs = await this.recsFromQuery(query);

      // Set done state
      log.info(`updateForLocation: state.recs = ${
        recursively(recs, (recs, recur) => (
          matchStateRecs(recs, {
            loading:  ({kind})  => kind,
            notfound: ({kind})  => kind,
            recs:     ({recs})  => `(${recs.length} recs)`,
            compare:  ({recss}) => _.join(recss.map(recur), '|'),
          })
        ))
      }`);
      this.setState({
        query,
        recs,
        recsQueryTime: timer.time(),
      });

    }
  }

  // TODO Factor these big matchQuery cases into functions, for readability
  recsFromQuery = async (query: Query | null): Promise<StateRecs> => {

    // Global filters
    //  - TODO(put_all_query_state_in_location)
    const qualityFilter = (table: string) => (
      sqlf`and ${SQL.raw(table)}.quality in (${ifEmpty(Array.from(this.props.filterQuality), () => Quality.values)})`
    );
    const placeFilter   = (table: string) => (
      sqlf`and ${SQL.raw(table)}.species in (${this.props.place.knownSpecies})`
    );
    const speciesFilter = (table: string) => (
      sqlf`and ${SQL.raw(table)}.species not in (${Array.from(this.props.excludeSpecies)})`
    );
    const speciesGroupFilter = (table: string) => (
      sqlf`and (false
        or ${SQL.raw(table)}.species_species_group not in (${Array.from(this.props.excludeSpeciesGroups)})
        or ${SQL.raw(table)}.species in (${Array.from(this.props.unexcludeSpecies)})
      )`
    );
    const recFilter = (table: string) => (
      sqlf`and ${SQL.raw(table)}.source_id not in (${Array.from(this.state.excludeRecs)})`
    );

    // Handle /rec/:sourceId not found (e.g. user deleted a user rec, or xc dataset changed)
    if (query === null) {
      return {kind: 'notfound'};
    } else {
      return await matchQuery<Promise<StateRecs>>(query, {

        none: async () => {
          log.info(`updateForLocation: QueryNone -> 'notfound'...`);
          return {kind: 'notfound'};
        },

        // TODO(slow_random): Perf: Very slow (~5s on US) b/c full table scan
        //  - Needs some kind of index, which probably means we need to replace random() with a materialized `random` col
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
              limit ${this.props.n_recs}
            )
            order by
              ${SQL.raw(matchKey(this.props.sortListResults, {
                species_then_random: () => 'taxon_order_num asc, random()',
                random:              () => 'random()',
                xc_id:               () => 'xc_id desc',
                month_day:           () => 'month_day asc',
                date:                () => 'date desc',
                lat:                 () => 'lat desc', // +90 N -> -90 S
                lng:                 () => 'lng asc',  // -180 ~HI -> +180 ~NZ
                country__state:      () => 'country asc, state asc',
                quality:             () => 'quality desc',
              }))},
              xc_id desc
          `, {
            logTruncate: null, // XXX Debug (safe to always log full query, no perf concerns)
            // logQueryPlan: true, // XXX Debug
          })<StateRecs>(async results => {
            const recs = results.rows.raw();
            return {kind: 'recs', recs};
          });
        },

        species: async ({filters, species}) => {
          log.info('updateForLocation: Querying recs for species', {species});
          return await this.props.db.query<XCRec>(sqlf`
            select
              *,
              cast(taxon_order as real) as taxon_order_num
            from search_recs S
            where true
              and species in (${species.split(',').map(x => _.trim(x).toUpperCase())})
              ${SQL.raw(placeFilter('S'))}        -- NOTE No results if species is outside of placeFilter
              ${SQL.raw(qualityFilter('S'))}
              ${SQL.raw(speciesFilter('S'))}      -- NOTE No results if species is excluded
              ${SQL.raw(speciesGroupFilter('S'))} -- NOTE No results if species's species_group is excluded
              ${SQL.raw(recFilter('S'))}
            order by
              ${SQL.raw(matchKey(this.props.sortListResults, {
                species_then_random: () => 'taxon_order_num asc, random()',
                random:              () => 'random()',
                xc_id:               () => 'xc_id desc',
                month_day:           () => 'month_day asc',
                date:                () => 'date desc',
                lat:                 () => 'lat desc', // +90 N -> -90 S
                lng:                 () => 'lng asc',  // -180 ~HI -> +180 ~NZ
                country__state:      () => 'country asc, state asc',
                quality:             () => 'quality desc',
              }))},
              xc_id desc
            limit ${this.props.n_recs}
          `, {
            logTruncate: null, // XXX Debug (safe to always log full query, no perf concerns)
            // logQueryPlan: true, // XXX Debug
          })<StateRecs>(async results => {
            const recs = results.rows.raw();
            return {kind: 'recs', recs};
          });
        },

        species_group: async ({filters, species_group}) => {
          log.info('updateForLocation: Querying recs for species_group', {species_group});
          return await this.props.db.query<XCRec>(sqlf`
            with
              S_shuffled_per_sp as (
                select
                  *,
                  cast(taxon_order as real) as taxon_order_num, -- For sortListResults:'species_then_random'
                  row_number() over (
                    partition by species -- Per sp
                    order by random()    -- Shuffle
                  ) as i
                from search_recs S
                where true
                  and species_species_group in (${species_group.split(';').map(x => _.trim(x))}) -- HACK species_group can contain ','
                  ${SQL.raw(placeFilter('S'))}        -- NOTE No results if species_group's species are all outside of placeFilter
                  ${SQL.raw(qualityFilter('S'))}
                  ${SQL.raw(speciesFilter('S'))}      -- NOTE No results if species_group's species are all excluded
                  ${SQL.raw(speciesGroupFilter('S'))} -- NOTE No results if species_group is excluded
                  ${SQL.raw(recFilter('S'))}
              ),
              S_sampled_per_sp as (
                select *
                from S_shuffled_per_sp
                order by i asc, random()   -- Pick ~n_per_sp recs per sp, ordered by i (= row_number())
                limit ${this.props.n_recs} -- Limit by total rec count
              )
            select *
            from S_sampled_per_sp
            order by
              ${SQL.raw(matchKey(this.props.sortListResults, {
                species_then_random: () => 'taxon_order_num asc, random()',
                random:              () => 'random()',
                xc_id:               () => 'xc_id desc',
                month_day:           () => 'month_day asc',
                date:                () => 'date desc',
                lat:                 () => 'lat desc', // +90 N -> -90 S
                lng:                 () => 'lng asc',  // -180 ~HI -> +180 ~NZ
                country__state:      () => 'country asc, state asc',
                quality:             () => 'quality desc',
              }))},
              xc_id desc
          `, {
            logTruncate: null, // XXX Debug (safe to always log full query, no perf concerns)
            // logQueryPlan: true, // XXX Debug
          })<StateRecs>(async results => {
            const recs = results.rows.raw();
            return {kind: 'recs', recs};
          });
        },

        // NOTE Window functions don't appear to do this faster than the one-per-sp unions we currently have
        //  - I made 2 separate attempts at it, and the best I got was windowing at ~20% _slower_ than unions
        //  - Notes in notebooks/20190226_mobile_dev_search_sqlite
        rec: async ({filters, source}) => {
          return await log.timedAsync<StateRecs>('updateForLocation.rec', async () => {
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
            const f_preds_cols = this.state.f_preds_cols!; // Guarded above
            const f_preds_col  = this.state.f_preds_col!;  // Guarded above
            const {n_per_sp, n_recs} = this.props;

            // Load query_rec from db
            const query_rec = await this.props.db.loadRec(source);
            if (query_rec === null) {
              // query_rec not found (e.g. user deleted a user rec, or xc dataset changed)
              return {kind: 'notfound'};
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

            // Compute slps (species (negative) log prob) from query_rec.f_preds
            //  1. Read sp_ps (species probs) from query_rec.f_preds
            //  2. Compute slps from sp_ps
            const sp_ps: Map<string, number> = new Map(zipSame(
              this.props.modelsSearch.classes_,
              query_rec.f_preds,
            ));
            const slps: Map<string, number> = mapMapValues(sp_ps, sp_p => (
              Math.abs(-Math.log(sp_p)) // (abs so that 1 -> 0 i/o -0)
            ));

            // Compute in sql: cosine_distance(S.f_preds_*, query_rec.f_preds)
            //  - cosine_distance(x,y) = 1 - dot(x,y) / norm(x) / norm(y)
            const joinOpWithBalancedParens = (op: string, exprs: Array<string>): string => {
              // HACK Workaround a sqlite crash during query parsing when len(f_preds) is large (≥634)
              //  - Without parens, `x_0 + ... + x_n`, 634 terms causes sqlite to crash somewhere in expr parsing code
              //    - Error from sqlite parsing code: "EXC_BAD_ACCESS (code=2, address=0x...)"
              //    - Using terms like: `S.f_preds_634*0.00003595540329115465`
              //    - Using sqlite version: https://github.com/brodybits/react-native-sqlite-plugin-legacy-support#042f681
              //  - Googling for "EXC_BAD_ACCESS code 2" turns up a lot of different leads, one of which is some
              //    kind of stack overflow, and the sqlite stack upon crashing is pretty deep (~1k frames)
              //  - With balanced parens, `(x_0 + ... + x_k) + (x_k + ... + x_n)`, sqlite parsing succeeds with:
              //    - n=738 terms (US search_recs)
              //    - k=100 maxTerms (hardcoded below)
              //  - So if this hypothesis is correct, then this balanced-parens hack should work for all n < ~2^634
              const maxTerms = 100;
              const n = exprs.length;
              if (n <= maxTerms) {
                return exprs.join(` ${op} `);
              } else {
                const mid = Math.floor(n / 2);
                const y1 = joinOpWithBalancedParens(op, _.slice(exprs, 0, mid));
                const y2 = joinOpWithBalancedParens(op, _.slice(exprs, mid, n));
                return `(${y1}) ${op} (${y2})`;
              }
            };
            const sqlDot = (
              joinOpWithBalancedParens('+', _
                .range(f_preds_cols.length)
                // .slice(0, 3) // XXX Debug: smaller query
                .map(i => sqlf`S.${SQL.raw(f_preds_col(i))}*${query_rec.f_preds[i]}`)
              )
              || '0'
            );
            const sqlCosineDist = sqlf`
              1 - (${SQL.raw(sqlDot)}) / S.norm_f_preds / ${Rec.norm_f_preds(query_rec)}
            `;

            // Query which species are left after applying filters
            //  - This is difficult to make exact, but we can get very close
            //  - TODO Perf: Cache this once per filter update i/o redoing on every rec search
            //  - NOTE Perf: If this is slow (e.g. for a place with many species), make sure it's using a covering index
            //    - Query plan should say `USING COVERING INDEX`, not `USING INDEX` (or `SCAN TABLE`)
            //    - https://www.sqlite.org/optoverview.html#covering_indices
            //    - https://www.sqlite.org/queryplanner.html#_covering_indices
            //    - See py model.payloads.df_cache_hybrid for index definitions
            log.info('updateForLocation: Querying species for filters', rich({query_rec}));
            const filteredSpecies: Set<Species> = await this.props.db.query<{species: Species}>(sqlf`
              select distinct species
              from search_recs S
              where true
                -- Filters duplicated below (in final query)
                ${SQL.raw(placeFilter('S'))}        -- Safe for covering index (species)
                ${SQL.raw(qualityFilter('S'))}      -- Safe for covering index (quality)
                ${SQL.raw(speciesFilter('S'))}      -- Safe for covering index (species)
                ${SQL.raw(speciesGroupFilter('S'))} -- Safe for covering index (species_species_group)
                ${SQL.raw(recFilter('S'))}          -- Safe for covering index (source_id)
            `, {
              logTruncate: null,
              // logQueryPlan: true, // XXX Debug: ensure covering index (see above)
            })(async results => {
              const recs = results.rows.raw();
              return new Set(recs.map(rec => rec.species));
            });

            // Rank species by slp (slp asc b/c sgn(slp) ~ -sgn(sp_p))
            //  - Filter species to match rec filters, else we'll return too few rec results below
            const n_sp = Math.ceil(n_recs / n_per_sp);
            const topSlps: Array<{species: string, slp: number}> = (
              _(Array.from(slps.entries()))
              .map(([species, slp]) => ({species, slp}))
              .filter(({species}) => filteredSpecies.has(species))
              .sortBy(({slp}) => slp)
              .slice(0, n_sp)
              .value()
            );

            // Construct query
            //  - Union `limit n` queries per species (b/c we don't have windowing)
            //  - Perf: We exclude .f_preds_* cols for faster load (ballpark ~2x)
            const non_f_preds_cols = this.state.non_f_preds_cols!; // Set in componentDidMount
            const sqlPerSpecies = (
              ifEmpty(topSlps, () => [
                // Mock ≥1 species, else we generate bad sql: `select * from ()`
                {species: '_XXX', slp: 1e38},
              ])
              // .slice(0, 2) // XXX Debug: smaller query
              .map(({species, slp}) => {
                if (!_.isNumber(slp) || _.isNaN(slp)) {
                  log.error(`updateForLocation: null/nan slp[${slp}] for species[${species}]`);
                }
                return sqlf`
                  select *, ${slp} as slp
                  from S_filter_dist
                  where species = ${species}
                  order by d_pc asc
                  limit ${topSlps.length > 1
                    ? n_per_sp // If we're showing multiple sp
                    : n_recs   // Else "max recs" if only one sp
                  }
                `;
              })
            );
            const sql = sqlf`
              with
                -- For nested sqlPerSpecies queries
                S_filter_dist as (
                  select
                    ${SQL.raw(non_f_preds_cols.map(x => `S.${x}`).join(', '))},
                    ${SQL.raw(sqlCosineDist)} as d_pc
                  from search_recs S
                  where true
                    -- Filters duplicated above (in filteredSpecies query)
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
              order by ${SQL.raw(matchKey(this.props.sortSearchResults, {
                slp__d_pc: () => 'slp asc, d_pc asc',
                d_pc:      () => 'd_pc asc',
              }))}
            `;

            // Run query
            log.info('updateForLocation: Querying recs for query_rec', rich({query_rec}));
            return await this.props.db.query<XCRec>(sql, {
              logTruncate: null, // XXX Debug (safe to always log full query, no perf concerns)
              // logQueryPlan: true, // XXX Debug
            })<StateRecs>(async results => {
              const recs = results.rows.raw();

              // XXX Debug
              // debug_print('timed', pretty(recs
              //   .map(rec => _.pick(rec, ['species', 'source_id', 'sp_d_pc_i', 'slp', 'd_pc']))
              //   .map(rec => yaml(rec))
              // ));

              // Validate
              //  - slp shouldn't be null/nan (else species order is junk)
              //  - d_pc shouldn't be null/nan (else rec order is junk)
              recs.forEach(rec => {
                typed<Array<keyof typeof rec>>(['slp', 'd_pc']).forEach(k => {
                  const x = rec[k];
                  if (!_.isNumber(x) || _.isNaN(x)) {
                    log.error(`updateForLocation: null/nan ${k}[${x}] for rec: ${rec.source_id} (${rec.species})`);
                  }
                });
              });

              // Inject query_rec as first result so it's visible at top
              //  - TODO Replace this with a proper display of query_rec at the top
              return {
                kind: 'recs',
                recs: [query_rec, ...recs],
              };

            });

          });
        },

        compare: async ({filters, queries}) => {
          log.info('updateForLocation: Querying recs for compare', {queries});
          const recss = await Promise.all(queries.map(this.recsFromQuery));
          return {kind: 'compare', recss};
        },

      });
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

  toggleRecPlaying = async (rec: Rec, soundAsync: Promise<Sound>, x: number): Promise<void> => {
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

  onSpectroHandlerStateChange = (rec: Rec) => {

    // Eagerly allocate Sound resource for rec
    //  - TODO How eagerly should we cache this? What are the cpu/mem costs and tradeoffs?
    const soundAsync = this.getOrAllocateSoundAsync(rec);

    // Mimic Gesture.BaseButton
    return async (event: Gesture.TapGestureHandlerStateChangeEvent) => {
      const {nativeEvent: {state, oldState, x, absoluteX}} = event; // Unpack SyntheticEvent (before async)
      if (
        oldState === Gesture.State.ACTIVE &&
        state !== Gesture.State.CANCELLED
      ) {
        await this.onSpectroPress(rec, soundAsync, x);
      }
    };

  }

  onSpectroPress = async (rec: Rec, soundAsync: Promise<Sound>, x: number) => {
    // Toggle play/pause normally, but show modal if playOnTap is disabled
    //  - UX HACK to allow a faster workflow for hiding lots of families/species/recs in a row
    if (this.props.playOnTap) {
      await this.toggleRecPlaying(rec, soundAsync, x);
    } else {
      await this.showRecActionModal(rec);
    }
  }

  onSpectroLongPress = (rec: Rec) => async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    if (state === Gesture.State.ACTIVE) {
      await this.showRecActionModal(rec);
    }
  }

  FiltersModal = () => {
    const Separator = () => (
      <View style={{height: 5}}/>
    );
    const buttonStyle: StyleProp<ViewStyle> = {
      width:          40,
      height:         40,
      justifyContent: 'center', // Main axis
      alignItems:     'center', // Cross axis (unless wrapped)
    };
    return (
      <this.GenericModal>
        <this.GenericModalTitle title='Filters' />

        <Separator/>
        <View style={{
          flexDirection:  'column',
          justifyContent: 'center',     // Main axis
          alignItems:     'flex-start', // Cross axis (unless wrapped)
          alignContent:   'flex-start', // Cross axis (if wrapped)
        }}>
          {typed<Array<[number, (sign: Sign) => Promise<void>, string]>>([
            [this.props.n_recs,   this.edit_n_recs,   'Results'],
            [this.props.n_per_sp, this.edit_n_per_sp, 'Results per species'],
          ]).map(([n, edit_n, label], i) => (
            <View
              key={i}
              style={{
                flexDirection:  'row',
                justifyContent: 'center', // Main axis
                alignItems:     'center', // Cross axis (unless wrapped)
                alignContent:   'center', // Cross axis (if wrapped)
              }}
            >
              <Separator/>
              <this.ActionModalButton
                iconName={'minus'}
                textColor={iOSColors.black}
                buttonColor={iOSColors.customGray}
                buttonStyle={buttonStyle}
                dismiss={false}
                onPress={() => edit_n(-1)}
              />
              <View style={{
                width: 30,
                alignItems: 'center', // Cross axis (unless wrapped)
              }}>
                <Text>{n}</Text>
              </View>
              <this.ActionModalButton
                iconName={'plus'}
                textColor={iOSColors.black}
                buttonColor={iOSColors.customGray}
                buttonStyle={buttonStyle}
                dismiss={false}
                onPress={() => edit_n(+1)}
              />
              <Text>{label}</Text>
            </View>
          ))}
        </View>

        <Separator/>
        <Separator/>
        <View style={{
          flexDirection:  'row',
          justifyContent: 'flex-start', // Main axis
          alignItems:     'center',     // Cross axis (unless wrapped)
          alignContent:   'center',     // Cross axis (if wrapped)
        }}>
          <Text>Quality: {}</Text>
          {Quality.values.map(quality => (
            <this.ActionModalButton
              key={quality}
              label={match<Quality, string>(quality,
                ['no score',    () => 'no'], // Short label to fit inside small square button
                [match.default, () => quality],
              )}
              textColor={iOSColors.black}
              buttonColor={this.props.filterQuality.has(quality) ? iOSColors.tealBlue : iOSColors.customGray}
              buttonStyle={{
                ...buttonStyle,
                marginHorizontal: 1,
              }}
              dismiss={false}
              onPress={() => this.props.settings.set(settings => ({
                filterQuality: setToggle(settings.filterQuality, quality),
              }))}
            />
          ))}
        </View>

        {/* XXX Example of <TextInput>
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
        */}

      </this.GenericModal>
    );
  }

  SortModal = () => {
    return (
      <this.ActionModal title='Sort' actions={local(() => {
        const buttonProps = (active: boolean) => ({
          textColor:   iOSColors.black,
          buttonColor: active ? iOSColors.orange : iOSColors.customGray,
        });
        return matchNull(this.state.query, {
          null: ()    => [],
          x:    query => matchQuery(query, {
            none: () => [],
            random: () => [
              {
                ...buttonProps(this.props.sortListResults === 'species_then_random'),
                onPress: () => this.props.settings.set({sortListResults: 'species_then_random'}),
                label: 'Species, then random',
              }, {
                ...buttonProps(this.props.sortListResults === 'random'),
                onPress: () => this.props.settings.set({sortListResults: 'random'}),
                label: 'Random',
              }, {
                ...buttonProps(this.props.sortListResults === 'xc_id'),
                onPress: () => this.props.settings.set({sortListResults: 'xc_id'}),
                label: 'XC ID (new→old)',
              }, {
                ...buttonProps(this.props.sortListResults === 'month_day'),
                onPress: () => this.props.settings.set({sortListResults: 'month_day'}),
                label: 'Season (Jan→Dec)',
              }, {
                ...buttonProps(this.props.sortListResults === 'date'),
                onPress: () => this.props.settings.set({sortListResults: 'date'}),
                label: 'Date (new→old)',
              }, {
                ...buttonProps(this.props.sortListResults === 'lat'),
                onPress: () => this.props.settings.set({sortListResults: 'lat'}),
                label: 'Latitude (N→S)',
              }, {
                ...buttonProps(this.props.sortListResults === 'lng'),
                onPress: () => this.props.settings.set({sortListResults: 'lng'}),
                label: 'Longitude (W→E)',
              }, {
                ...buttonProps(this.props.sortListResults === 'country__state'),
                onPress: () => this.props.settings.set({sortListResults: 'country__state'}),
                label: 'Country/state (alphabetical)',
              }, {
                ...buttonProps(this.props.sortListResults === 'quality'),
                onPress: () => this.props.settings.set({sortListResults: 'quality'}),
                label: 'Quality (good→bad)',
              },
            ],
            species_group: () => [
              {
                ...buttonProps(this.props.sortListResults === 'species_then_random'),
                onPress: () => this.props.settings.set({sortListResults: 'species_then_random'}),
                label: 'Species, then random',
              }, {
                ...buttonProps(this.props.sortListResults === 'random'),
                onPress: () => this.props.settings.set({sortListResults: 'random'}),
                label: 'Random',
              }, {
                ...buttonProps(this.props.sortListResults === 'xc_id'),
                onPress: () => this.props.settings.set({sortListResults: 'xc_id'}),
                label: 'XC ID (new→old)',
              }, {
                ...buttonProps(this.props.sortListResults === 'month_day'),
                onPress: () => this.props.settings.set({sortListResults: 'month_day'}),
                label: 'Season (Jan→Dec)',
              }, {
                ...buttonProps(this.props.sortListResults === 'date'),
                onPress: () => this.props.settings.set({sortListResults: 'date'}),
                label: 'Date (new→old)',
              }, {
                ...buttonProps(this.props.sortListResults === 'lat'),
                onPress: () => this.props.settings.set({sortListResults: 'lat'}),
                label: 'Latitude (N→S)',
              }, {
                ...buttonProps(this.props.sortListResults === 'lng'),
                onPress: () => this.props.settings.set({sortListResults: 'lng'}),
                label: 'Longitude (W→E)',
              }, {
                ...buttonProps(this.props.sortListResults === 'country__state'),
                onPress: () => this.props.settings.set({sortListResults: 'country__state'}),
                label: 'Country/state (alphabetical)',
              }, {
                ...buttonProps(this.props.sortListResults === 'quality'),
                onPress: () => this.props.settings.set({sortListResults: 'quality'}),
                label: 'Quality (good→bad)',
              },
            ],
            species: () => [
              {
                ...buttonProps(this.props.sortListResults === 'random'),
                onPress: () => this.props.settings.set({sortListResults: 'random'}),
                label: 'Random',
              }, {
                ...buttonProps(this.props.sortListResults === 'xc_id'),
                onPress: () => this.props.settings.set({sortListResults: 'xc_id'}),
                label: 'XC ID (new→old)',
              }, {
                ...buttonProps(this.props.sortListResults === 'month_day'),
                onPress: () => this.props.settings.set({sortListResults: 'month_day'}),
                label: 'Season (Jan→Dec)',
              }, {
                ...buttonProps(this.props.sortListResults === 'date'),
                onPress: () => this.props.settings.set({sortListResults: 'date'}),
                label: 'Date (new→old)',
              }, {
                ...buttonProps(this.props.sortListResults === 'lat'),
                onPress: () => this.props.settings.set({sortListResults: 'lat'}),
                label: 'Latitude (N→S)',
              }, {
                ...buttonProps(this.props.sortListResults === 'lng'),
                onPress: () => this.props.settings.set({sortListResults: 'lng'}),
                label: 'Longitude (W→E)',
              }, {
                ...buttonProps(this.props.sortListResults === 'country__state'),
                onPress: () => this.props.settings.set({sortListResults: 'country__state'}),
                label: 'Country/state (alphabetical)',
              }, {
                ...buttonProps(this.props.sortListResults === 'quality'),
                onPress: () => this.props.settings.set({sortListResults: 'quality'}),
                label: 'Quality (good→bad)',
              },
            ],
            rec: () => [
              {
                ...buttonProps(this.props.sortSearchResults === 'slp__d_pc'),
                onPress: () => this.props.settings.set({sortSearchResults: 'slp__d_pc'}),
                label: 'Species match, then similar recs',
              }, {
                ...buttonProps(this.props.sortSearchResults === 'd_pc'),
                onPress: () => this.props.settings.set({sortSearchResults: 'd_pc'}),
                label: 'Similar recs only (ignore species match)',
              },
            ],
            compare: ({queries}) => [
              // TODO How to present sorting options for compare view?
              {
                ...buttonProps(false),
                onPress: () => {},
                label: '(Disabled for compare view)',
              },
            ],
          }),
        });
      })} />
    );
  }

  showRecActionModal = async (rec: Rec) => {
    this.setState({
      sourceIdForActionModal: rec.source_id,
      showGenericModal: () => (
        this.RecActionModal(rec)
      ),
    });
  }

  RecActionModal = (rec: Rec) => {
    const Separator = () => (
      <View style={{height: 5}}/>
    );
    const styleLikeMetadataLabel: TextStyle = {
      ...material.captionObject,
      fontWeight: 'bold',
    };
    const defaults = {
      buttonStyle: {
        marginVertical: 1,
        marginHorizontal: 5,
        paddingVertical: 2,
        paddingHorizontal: 5,
      },
    };
    return (
      <this.GenericModal style={{
        width: '100%',
        marginTop: 40, marginBottom: 40, // Enough margin so the user can tap outside the modal to close
        padding: 0, // i/o default 15
      }}>
        <ScrollView style={{
          padding: 15, // i/o 15 in GenericModal
        }}>

          {/* Rec title + Edit button */}
          <View style={{
            flexDirection: 'row',
            justifyContent: 'space-between', // (horizontal) Push Edit button out to the right
            alignItems: 'center', // (vertical)
          }}>
            <Text style={{...material.titleObject}}>
              {Source.show(Rec.source(rec), {
                species: this.props.xc,
              })}
            </Text>
            {this.ActionModalButtons({actions: [
              {
                ...defaults,
                label: 'Edit',
                iconName: 'edit',
                buttonColor: iOSColors.purple,
                buttonStyle: {
                  // No margins
                  marginHorizontal:  0, // i/o default 10
                  marginVertical:    0, // i/o default 2
                  // HACK Why must we override padding:10 to make it look like other buttons?
                  paddingHorizontal: 5,
                  paddingVertical:   2,
                },
                onPress: () => this.props.go('record', {path: `/edit/${encodeURIComponent(rec.source_id)}`}),
              }
            ]})}
          </View>

          {/* Spectro */}
          <Separator/>
          <Animated.Image
            // TODO Wrap spectro (e.g. over exactly 2 rows) instead of squishing horizontally
            style={{
              // ...this.spectroDimImage(rec), // XXX Bad(info_modal)
              height: this.spectroDimImage(rec).height,
              width: '100%',
            }}
            foo
            resizeMode='stretch' // TODO(info_modal) Wrap to show whole spectro i/o stretching
            source={{uri: Rec.spectroPath(rec, this.spectroPathOpts)}}
          />

          {/* 'search' buttons */}
          <Separator/>
          <Text style={styleLikeMetadataLabel}>
            Find recordings similar to:
          </Text>
          <Separator/>
          <View style={{flexDirection: 'row'}}>
            {this.ActionModalButtons({actions: [
              {
                ...defaults,
                label: Source.show(Rec.source(rec), {
                  species: this.props.xc,
                }),
                // iconName: 'search',
                iconName: this.props.iconForTab['search'],
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
                // label: rec.species,
                label: rec.species_com_name,
                // iconName: 'search',
                iconName: this.props.iconForTab['search'],
                buttonColor: iOSColors.blue,
                onPress: () => this.props.go('search', {path: `/species/${encodeURIComponent(rec.species)}`}),
              },
            ]})}
          </View>
          <Separator/>
          <View style={{flexDirection: 'row'}}>
            {this.ActionModalButtons({actions: [
              {
                ...defaults,
                label: rec.species_species_group,
                // iconName: 'search',
                iconName: this.props.iconForTab['search'],
                buttonColor: iOSColors.blue,
                onPress: () => this.props.go('search', {path: `/species_group/${encodeURIComponent(rec.species_species_group)}`}),
              }
            ]})}
          </View>

          {/* 'eye' buttons */}
          <Separator/>
          <Text style={styleLikeMetadataLabel}>
            Filter results to only:
          </Text>
          <Separator/>
          <View style={{flexDirection: 'row'}}>
            {this.ActionModalButtons({actions: [
              {
                ...defaults,
                // label: rec.species,
                label: rec.species_com_name,
                // iconName: 'maximize-2',
                iconName: 'eye',
                buttonColor: iOSColors.orange,
                onPress: () => {
                  const {ebird} = this.props;
                  const species = rec.species;
                  // TODO(exclude_invariants): Dedupe with BrowseSectionHeader/BrowseItem
                  //  - (We're always in the !exG case b/c otherwise this rec wouldn't be in the results)
                  this.props.settings.set(({excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS}) => {
                    const s = species;
                    // HACK(exclude_invariants): Smash through existing state [is this good?]
                    exS = setDiff(new Set(ebird.allSpecies), s);
                    exG = new Set();
                    unS = new Set();
                    return {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS};
                  });
                },
              }
            ]})}
          </View>
          <Separator/>
          <View style={{flexDirection: 'row'}}>
            {this.ActionModalButtons({actions: [
              {
                ...defaults,
                label: rec.species_species_group,
                // iconName: 'maximize-2',
                iconName: 'eye',
                buttonColor: iOSColors.orange,
                onPress: () => {
                  const {ebird} = this.props;
                  const species_group = rec.species_species_group;
                  // TODO(exclude_invariants): Dedupe with BrowseSectionHeader/BrowseItem
                  //  - (We're always in the !exG case b/c otherwise this rec wouldn't be in the results)
                  this.props.settings.set(({excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS}) => {
                    const g = species_group;
                    // HACK(exclude_invariants): Smash through existing state [is this good?]
                    exS = new Set();
                    exG = setDiff(new Set(ebird.allSpeciesGroups), g);
                    unS = new Set();
                    return {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS};
                  });
                },
              }
            ]})}
          </View>

          {/* 'eye-off' buttons */}
          <Separator/>
          <Text style={styleLikeMetadataLabel}>
            Exclude from results:
          </Text>
          <Separator/>
          <View style={{flexDirection: 'row'}}>
            {this.ActionModalButtons({actions: [
              {
                ...defaults,
                label: Source.show(Rec.source(rec), {
                  species: this.props.xc,
                }),
                iconName: 'eye-off',
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
                // label: rec.species,
                label: rec.species_com_name,
                iconName: 'eye-off',
                buttonColor: iOSColors.red,
                onPress: () => {
                  const species       = rec.species;
                  const species_group = rec.species_species_group;
                  // TODO(exclude_invariants): Dedupe with BrowseSectionHeader/BrowseItem
                  this.props.settings.set(({excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS}) => {
                    const s = species;
                    const g = species_group;
                    if      (!exG.has(g) && !exS.has(s)) { exS = setAdd  (exS, s); } // !exG, !exS -> exS+s
                    else if (!exG.has(g) &&  exS.has(s)) { exS = setDiff (exS, s); } // !exG,  exS -> exS-s
                    else if ( exG.has(g) && !unS.has(s)) { unS = setAdd  (unS, s); } //  exG, !unS -> unS+s
                    else if ( exG.has(g) &&  unS.has(s)) { unS = setDiff (unS, s); } //  exG,  unS -> unS-s
                    return {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS};
                  });
                },
              }
            ]})}
          </View>
          <Separator/>
          <View style={{flexDirection: 'row'}}>
            {this.ActionModalButtons({actions: [
              {
                ...defaults,
                label: rec.species_species_group,
                iconName: 'eye-off',
                buttonColor: iOSColors.red,
                onPress: () => {
                  const {ebird} = this.props;
                  const species_group = rec.species_species_group;
                  // TODO(exclude_invariants): Dedupe with BrowseScreen
                  const unexcludedAny = (unS: Set<Species>) => (
                    Array.from(unS)
                    .map(x => ebird.speciesGroupFromSpecies(x))
                    .includes(species_group)
                  );
                  // TODO(exclude_invariants): Dedupe with BrowseSectionHeader/BrowseItem
                  //  - (We're always in the !exG case b/c otherwise this rec wouldn't be in the results)
                  this.props.settings.set(({excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS}) => {
                    const unAny = unexcludedAny(unS);
                    const g     = species_group;
                    const ss    = ebird.speciesForSpeciesGroup.get(g) || []; // (Degrade gracefully if g is somehow unknown)
                    if      (!exG.has(g)          ) { exG = setAdd  (exG, g); exS = setDiff (exS, ss); } // !exG         -> exG+g, exS-ss
                    else if ( exG.has(g) && !unAny) { exG = setDiff (exG, g);                          } //  exG, !unAny -> exG-g
                    else if ( exG.has(g) &&  unAny) { unS = setDiff (unS, ss);                         } //  exG,  unAny -> unS-ss
                    return {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS};
                  });
                },
              }
            ]})}
          </View>

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

        </ScrollView>
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

      {/* Random recs */}
      <this.BottomControlsButton
        help='Shuffle'
        iconProps={{name: 'shuffle'}}
        onPress={() => this.props.go('search', {path: this.shufflePath()})}
      />

      {/* Reset filters */}
      <this.BottomControlsButton
        help='Reset'
        // iconProps={{name: 'refresh-ccw'}}
        iconProps={{name: 'rotate-ccw'}}
        disabled={true
          // If no active filters
          && this.props.excludeSpecies.size       === 0
          && this.props.excludeSpeciesGroups.size === 0
          && this.props.unexcludeSpecies.size     === 0
          && !this.state.filterQueryText
          && this.state.excludeRecs.size === 0
        }
        onPress={() => {
          log.info('Reset filters');
          this.props.settings.set({
            excludeSpecies:       new Set(),
            excludeSpeciesGroups: new Set(),
            unexcludeSpecies:     new Set(),
          });
          this.setState({
            filterQueryText:      undefined,
            excludeRecs:          new Set(),
          });
        }}
      />

      {/* Filters */}
      <this.BottomControlsButton
        help='Filters'
        iconProps={{name: 'filter'}}
        onPress={() => this.setState({
          showGenericModal: () => this.FiltersModal(),
        })}
      />

      {/* XXX Dev: Query that returns no results */}
      {/* <this.BottomControlsButton
        help='Blank'
        iconProps={{name: 'power'}}
        onPress={() => this.props.go('search', {path: `/species/${encodeURIComponent('_BLANK')}`})} // HACK No results via junk species
      /> */}

      {/* Toggle sort */}
      <this.BottomControlsButton
        help='Sort'
        iconProps={{name: 'chevrons-down'}}
        // iconProps={{name: 'chevron-down'}}
        // iconProps={{name: 'arrow-down'}}
        // iconProps={{name: 'arrow-down-circle'}}
        onPress={() => this.setState({
          showGenericModal: () => this.SortModal(),
        })}
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
          ),
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
          ),
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

            // XXX Red !playOnTap is too complex, too confusing for UX
            // playOnTap && seekOnPlay  ? [true,  false] : // blue  -> black
            // playOnTap && !seekOnPlay ? [false, false] : // black -> red
            //                            [true,  true]    // red   -> blue

            // Simpler (skip red !playOnTap)
            playOnTap && seekOnPlay  ? [true,  false] : // blue  -> black
                                       [true,  true]    // black -> blue

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
          event.nativeEvent.state === Gesture.State.ACTIVE && props.onLongPress && (
            props.onLongPress()
          )
        )}
      >
        <BorderlessButton
          style={styles.bottomControlsButton}
          onPress={props.disabled ? undefined : props.onPress}
        >
          {/* XXX Subsumed by TitleBarWithHelp */}
          {/* {this.props.showHelp && (
            <Text style={styles.bottomControlsButtonHelp}>{props.help}</Text>
          )} */}
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
    children?: ReactNode,
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
      {props.children}
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
      buttonStyle?: StyleProp<ViewStyle>,
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
      label?: string,
      iconName?: string,
      buttonColor?: string,
      textColor?: string,
      buttonStyle?: StyleProp<ViewStyle>,
      dismiss?: boolean,
      onPress: () => void,
    }>,
  }) => props.actions.map((props, i) => (
    <this.ActionModalButton
      key={i}
      {...props}
    />
  ));

  ActionModalButton = (props: {
    label?: string,
    iconName?: string,
    buttonColor?: string,
    textColor?: string,
    buttonStyle?: StyleProp<ViewStyle>,
    dismiss?: boolean,
    onPress: () => void,
  }) => (
    <RectButton
      style={{
        // flex:             1, // Makes everything big
        flexDirection:    'row',
        justifyContent:   'flex-start', // Main axis
        alignItems:       'center',     // Cross axis (unless wrapped)
        padding:          10,
        marginHorizontal: 10,
        marginVertical:   2,
        backgroundColor:  _.defaultTo(props.buttonColor, iOSColors.customGray),
        ..._.defaultTo(props.buttonStyle, {}),
      }}
      onPress={() => {
        if (_.defaultTo(props.dismiss, true)) {
          this.setState({
            showGenericModal: null, // Dismiss modal
          });
        }
        props.onPress();
      }}
    >
      {props.iconName && (
        <Feather
          style={{
            // ...material.headlineObject,
            ...material.buttonObject,
            color: _.defaultTo(props.textColor, iOSColors.white),
          }}
          name={props.iconName}
        />
      )}
      {props.iconName && props.label && (
        <View style={{width: 5}} />
      )}
      {props.label && (
        <Text
          style={{
            // ...material.buttonObject,
            ...material.body2Object,
            color: _.defaultTo(props.textColor, iOSColors.white),
          }}
          children={props.label}
        />
      )}
    </RectButton>
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

    // Assign styles to all species across displayed recs
    const styleForSpecies = this.stylesForSpecies(
      // Compute unique species across displayed recs
      _(recursively<StateRecs, Array<Rec>>(this.state.recs, (recs, recur) => (
        matchStateRecs(recs, {
          loading:  ()        => [],
          notfound: ()        => [],
          recs:     ({recs})  => recs,
          compare:  ({recss}) => _.flatten(recss.map(recur)),
        })
      )))
      .map(rec => rec.species)
      .uniq()
      .value()
    );

    return (
      <View style={{
        flex: 1,
      }}>

        {/* Redirect: '/' -> default */}
        <Route exact path='/' render={() => (
          <Redirect to={this.defaultPath()} />
        )}/>

        <TitleBarWithHelp

          // TODO TODO compare_view
          // title='Result recordings'
          title={
            // TODO Dedupe with SavedScreen.render + RecentScreen.render
            recursively({query: this.state.query, verbose: true}, ({query, verbose}, recur) => (
              !query ? (
                'Loading...'
              ) : (
                matchQuery(query, {
                  none: () => (
                    '[None]' // [Does this ever happen?]
                  ),
                  random: ({filters, seed}) => (
                    `Random`
                  ),
                  species_group: ({filters, species_group}) => (
                    species_group
                  ),
                  species: ({filters, species}) => (
                    species === '_BLANK' ? '[BLANK]' :
                    !verbose ? species :
                    matchUndefined(this.props.ebird.speciesMetadataFromSpecies.get(species), {
                      undefined: () => `? (${species})`,
                      x:         x  => `${x.com_name} (${species})`,
                    })
                  ),
                  rec: ({filters, source}) => (
                    Source.show(source, {
                      species:  this.props.xc,
                      long:     true, // e.g. 'User recording: ...' / 'XC recording: ...'
                      showDate: x => showDate(x),
                      // showDate: x => showDate(x).replace(`${section.title} `, ''), // HACK showTime if date = section date
                    })
                  ),
                  compare: ({filters, queries}) => (
                    `Compare: ${_.join(queries.map(query => recur({query, verbose: false})), ' | ')}`
                  ),
                })
              )
            ))
          }

          settings={this.props.settings}
          showHelp={this.props.showHelp}
          help={(
            // TODO TODO Add a headline ("These are search results" or whatever)
            <HelpText>
              • Tap a recording to play it{'\n'}
              • Long-press a recording to see more actions{'\n'}
              • Tap <Feather name='shuffle'/> to show a random species{'\n'}
              • Tap <Feather name='filter'/> to change how many results are shown{'\n'}
              • Tap <Feather name='rotate-ccw'/> to reset filters and exclusions{'\n'}
              • Tap <Feather name='chevrons-down'/> to change sorting{'\n'}
              • Tap <Feather name='sidebar'/> to show/hide metadata on the left{'\n'}
              • Tap <Feather name='credit-card' style={Styles.flipVertical}/> to show/hide metadata inline{'\n'}
              • Long-press <Feather name='sidebar'/>/<Feather name='credit-card' style={Styles.flipVertical}/> to
                change which metadata is shown{'\n'}
              • Tap <Feather name='crosshair'/> to toggle playback: start from where you tap or start from the beginning{'\n'}
              • Tap <Feather name='zoom-out'/>/<Feather name='zoom-in'/> to resize the spectrograms{'\n'}
              • Pinch in/out to resize the spectrograms
            </HelpText>
          )}
        />

        {/* TODO Extract this as a named function for easier readability */}
        {recursively(this.state.recs, (recs, recur) => (

          // Loading spinner
          recs.kind === 'loading' ? (
            <View style={{
              flex: 1,
              justifyContent: 'center',
            }}>
              <ActivityIndicator size='large' />
            </View>

          // Compare view
          //  - NOTE Must recur outside of the ScrollView (below) else the nested ScrollViews will behave like vanilla
          //    Views (no scrolling)
          ) : recs.kind === 'compare' ? (
            recs.recss.map((recs, i) => (
              <View key={i} style={{
                flex: 1,
                borderTopWidth: i === 0 ? 0 : 5,
                borderColor: iOSColors.gray,
              }}>
                {recur(recs)}
              </View>
            ))

          ) : (
            // Recs list (with pan/pinch)
            //  - We use ScrollView instead of SectionList to avoid _lots_ of opaque pinch-to-zoom bugs
            //  - We use ScrollView instead of manual gestures (react-native-gesture-handler) to avoid _lots_ of opaque animation bugs
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

            >
              {/* Mimic a FlatList */}

              {/*
              // (Unused, keeping for reference)
              // _.flatten(this.sectionsForRecs(recs).map(({
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
                {matchStateRecs<ReactNode>(recs, {
                  loading: () => {

                    // Dispatched above
                    throw new Error('Unreachable');

                  },
                  notfound: () => (

                    <View style={[Styles.center, {padding: 30,
                      width: Dimensions.get('window').width, // HACK Fix width else we drift right with scrollViewContentWidth
                    }]}>
                      <Text style={material.subheading}>
                        Recording not found
                      </Text>
                    </View>

                  ),
                  recs: ({recs}) => (
                    fastIsEqual(recs, []) ? (

                      <View style={[Styles.center, {padding: 30,
                        width: Dimensions.get('window').width, // HACK Fix width else we drift right with scrollViewContentWidth
                      }]}>
                        <Text style={material.subheading}>
                          No results
                        </Text>
                      </View>

                    ) : (
                      recs.map((rec, recIndex) => [

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
                                <TapGestureHandler onHandlerStateChange={this.onSpectroHandlerStateChange(rec)}>
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
                    )

                  ),
                  compare: ({recss}) => {

                    // Dispatched above
                    throw new Error('Unreachable');

                  },
                })}
              </View>

            </ScrollView>
          )
        ))}

        {/* Debug info */}
        <this.DebugView>
          <this.DebugText>queryDesc: {this.queryDesc}</this.DebugText>
          <this.DebugText>
            Recs: {recursively(this.state.recs, (recs, recur) => {
              matchStateRecs(recs, {
                loading:  ()        => `.../${this.state.totalRecs || '?'}`,
                notfound: ()        => `.../${this.state.totalRecs || '?'}`,
                recs:     ({recs})  => `${recs.length}/${this.state.totalRecs || '?'} (${sprintf('%.3f', this.state.recsQueryTime)}s)`,
                compare:  ({recss}) => _.join(_.flatten(recss.map(recs => recur(recs))), ' | '),
              })
            })}
          </this.DebugText>
          <this.DebugText>
            Filters: {yaml({
              filterQuality:   Array.from(this.props.filterQuality),
              filterQueryText: this.state.filterQueryText,
            })}
          </this.DebugText>
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

  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
    this.log.info('componentDidMount');
    this._keyboardDidShowListener = Keyboard.addListener('keyboardDidShow', this.keyboardDidShow);
    this._keyboardDidHideListener = Keyboard.addListener('keyboardDidHide', this.keyboardDidHide);
  });

  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    this.log.info('componentWillUnmount');
    this._keyboardDidShowListener!.remove();
    this._keyboardDidHideListener!.remove();
  });

  componentDidUpdate = async (
    prevProps: KeyboardDismissingViewProps,
    prevState: KeyboardDismissingViewState,
  ) => logErrorsAsync('componentDidUpdate', async () => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  });

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
