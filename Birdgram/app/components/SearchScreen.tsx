import * as d3sc from 'd3-scale-chromatic';
import { Location, MemoryHistory } from 'history';
import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import RN, {
  ActivityIndicator, Animated, Dimensions, FlatList, FlexStyle, GestureResponderEvent, Image, ImageStyle, Keyboard,
  KeyboardAvoidingView, LayoutChangeEvent, Modal, Platform, RegisteredStyle, ScrollView, SectionList, SectionListData,
  SectionListStatic, StyleProp, Text, TextInput, TextStyle, TouchableHighlight, View, ViewStyle, WebView,
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
import SQLite from 'react-native-sqlite-storage';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
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

import { ActionSheetBasic } from './ActionSheets';
import {
  MetadataColumnBelow, MetadataColumnsBelow, MetadataColumnLeft, MetadataColumnsLeft, MetadataText,
} from './MetadataColumns';
import { CCIcon, LicenseTypeIcons } from './Misc';
import { TabBarStyle } from './TabRoutes';
import { config } from '../config';
import {
  ModelsSearch, matchSearchPathParams, matchSourceId, Place, Quality, Rec, rec_f_preds, Rec_f_preds, SearchPathParams,
  searchPathParamsFromLocation, SearchRecs, ServerConfig, showSourceId, SourceId, Species, SpeciesMetadata, UserRec,
  XCRec,
} from '../datatypes';
import { Ebird } from '../ebird';
import { Log, puts, rich, tap } from '../log';
import { NativeSearch } from '../native/Search';
import { Go } from '../router';
import { SettingsWrites } from '../settings';
import Sound from '../sound';
import { queryPlanFromRows, queryPlanPretty, querySql, SQL, sqlf } from '../sql';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, LabelStyle, labelStyles, Styles } from '../styles';
import {
  all, any, chance, Clamp, deepEqual, Dim, finallyAsync, getOrSet, global, json, mapMapValues, match, matchNull,
  matchUndefined, noawait, objectKeysTyped, Omit, Point, pretty, QueryString, round, shallowDiffPropsState, Style,
  Timer, yaml, yamlPretty, zipSame,
} from '../utils';

const log = new Log('SearchScreen');

const sidewaysTextWidth = 14;
const recEditingButtonWidth = 30;

interface ScrollViewState {
  contentOffset: Point;
  // (More fields available in NativeScrollEvent)
}

type Query = QueryNone | QueryRandom | QuerySpecies | QueryRec;
type QueryNone    = {kind: 'none'}; // e.g. so we can show nothing on redirect from '/'
type QueryRandom  = {kind: 'random',  filters: Filters, seed: number};
type QuerySpecies = {kind: 'species', filters: Filters, species: string};
type QueryRec     = {kind: 'rec',     filters: Filters, sourceId: SourceId};
function matchQuery<X>(query: Query, cases: {
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

interface Props {
  // App globals
  serverConfig:            ServerConfig;
  modelsSearch:            ModelsSearch;
  location:                Location;
  history:                 MemoryHistory;
  go:                      Go;
  ebird:                   Ebird;
  // Settings
  settings:                SettingsWrites;
  showDebug:               boolean;
  showMetadataLeft:        boolean;
  showMetadataBelow:       boolean;
  metadataColumnsLeft:     Array<MetadataColumnLeft>;
  metadataColumnsBelow:    Array<MetadataColumnBelow>;
  editing:                 boolean;
  seekOnPlay:              boolean;
  playingProgressEnable:   boolean;
  playingProgressInterval: number;
  spectroScale:            number;
  place:                   null | Place;
  places:                  Array<Place>;
  // SearchScreen
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
  showHelp: boolean;
  totalRecs?: number;
  f_preds_cols?: Array<string>;
  refreshQuery: boolean; // TODO(put_all_query_state_in_location)
  // TODO Persist filters with settings
  //  - Top-level fields instead of nested object so we can use state merging when updating them in isolation
  filterQueryText?: string;
  filterQuality: Array<Quality>;
  n_recs: number;   // For non-rec queries
  n_sp: number;     // For rec queries
  n_per_sp: number; // For rec queries
  excludeSpecies: Array<string>;
  excludeRecIds: Array<string>;
  recs: 'loading' | Array<Rec>;
  recsQueryInProgress?: Query,
  recsQueryShown?: Query;
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
    scrollViewKey:    '',
    scrollViewState:  this._scrollViewState,
    showGenericModal: null,
    showHelp:         false,
    refreshQuery:     false,
    filterQuality:    ['A', 'B'],
    n_recs:           this.props.default_n_recs,
    n_sp:             this.props.default_n_sp,
    n_per_sp:         this.props.default_n_per_sp,
    excludeSpecies:   [],
    excludeRecIds:    [],
    recs:             'loading',
    _spectroScale:    this.props.spectroScale, // Sync from/to Settings (2/3)
  };

  // Getters for prevProps
  _pathParams = (props?: Props): SearchPathParams => {
    return searchPathParamsFromLocation((props || this.props).location);
  }

  // Getters for props
  get pathParams (): SearchPathParams { return this._pathParams(); }

  // Getters for state
  get filters(): object { return _.pickBy(this.state, (v, k) => k.startsWith('filter')); }
  get recsOrEmpty(): Array<Rec> { return this.state.recs === 'loading' ? [] : this.state.recs; }
  get query_rec(): null | Rec { return this.recsOrEmpty[0] || null; }

  // Private attrs
  db?: SQLiteDatabase;
  soundsCache: Map<SourceId, Promise<Sound> | Sound> = new Map();

  // (Unused, kept for reference)
  // sortActionSheet: RefObject<ActionSheet> = React.createRef();

  // Refs
  scrollViewRef: RefObject<SectionListStatic<Rec>> = React.createRef();

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
      'Default', // "The default audio session mode"
    );

    // Tell other apps we're using the audio device
    Sound.setActive(true);

    // Open db conn
    const dbFilename = SearchRecs.dbPath;
    const dbExists = await fs.exists(`${fs.dirs.MainBundleDir}/${dbFilename}`);
    if (!dbExists) {
      log.error(`componentDidMount: DB file not found: ${dbFilename}`);
    } else {
      const dbLocation = `~/${dbFilename}`; // Relative to app bundle (copied into the bundle root by react-native-asset)
      this.db = await SQLite.openDatabase({
        name: dbFilename,               // Just for SQLite bookkeeping, I think
        readOnly: true,                 // Else it will copy the (huge!) db file from the app bundle to the documents dir
        createFromLocation: dbLocation, // Else readOnly will silently not work
      });
    }

    // Query db size (once)
    log.info('componentDidMount: Querying db size');
    await querySql<{totalRecs: number}>(this.db!, `
      select count(*) as totalRecs
      from search_recs
    `)(async results => {
      log.info('componentDidMount: state.totalRecs');
      const [{totalRecs}] = results.rows.raw();
      this.setState({
        totalRecs,
      });
    });

    // Query f_preds_* cols (once)
    log.info('componentDidMount: Querying f_preds_* cols');
    await querySql<Rec>(this.db!, `
      select *
      from search_recs
      limit 1
    `)(async results => {
      log.info('componentDidMount: state.f_preds_cols');
      const [rec] = results.rows.raw();
      const n = Object.keys(rec).filter(k => k.startsWith('f_preds_')).length;
      // Reconstruct strings from .length to enforce ordering
      const f_preds_cols = _.range(n).map(i => `f_preds_${i}`);
      this.setState({
        f_preds_cols,
      });
    });

    // Query recs (from navParams.species)
    // log.debug('componentDidMount: loadRecsFromQuery()');
    await this.loadRecsFromQuery();

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

    // Reset view state if query changed
    //  - TODO Pass props.key to reset _all_ state? [https://reactjs.org/blog/2018/06/07/you-probably-dont-need-derived-state.html#recap]
    if (!deepEqual(this.query, this._query(prevProps))) {
      log.info('componentDidUpdate: Reset view state');
      this.setState({
        filterQueryText: undefined,
        n_recs:          this.props.default_n_recs,
        n_sp:            this.props.default_n_sp,
        n_per_sp:        this.props.default_n_per_sp,
        excludeSpecies:  [],
        excludeRecIds:   [],
      });
    }

    // Else _scrollViewState falls behind on non-scroll/non-zoom events (e.g. +/- buttons)
    this._scrollViewState = this.state.scrollViewState;

    // Sync from/to Settings (3/3)
    //  - These aren't typical: we only use this for (global) settings keys that we also keep locally in state so we can
    //    batch-update them with other local state keys (e.g. global spectroScale + local scrollViewKey)
    //  - QUESTION What's a better pattern for "batch setState(x,y,z) locally + persist settings.set(x) globally"?
    if (this.state._spectroScale !== prevState._spectroScale) {
      noawait(this.props.settings.set('spectroScale', this.state._spectroScale));
    }

    // Query recs (from updated navParams.species)
    //  - (Will noop if deepEqual(query, state.recsQueryShown))
    // log.debug('componentDidUpdate: loadRecsFromQuery()');
    await this.loadRecsFromQuery();

  }

  randomPath = (seed?: number): string => {
    seed = seed !== undefined ? seed : chance.natural({max: 1e6});
    return `/random/${seed}`;
  }

  spectroDim = (duration_s: number): Dim<number> => {
    return {
      height: this.props.spectroBase.height * this.state._spectroScale,
      width:  this.scrollViewContentWidths.image / this.props.searchRecsMaxDurationS * duration_s,
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

  get query(): Query { return this._query(); }
  _query = (props?: Props): Query => {
    return matchSearchPathParams<Query>(this._pathParams(props), {
      none:    ()                    => ({kind: 'species', filters: {}, species: ''}),
      random:  ({filters, seed})     => ({kind: 'random',  filters, seed}),
      species: ({filters, species})  => ({kind: 'species', filters, species}),
      rec:     ({filters, sourceId}) => ({kind: 'rec',     filters, sourceId}),
    });
    // We ignore state.filterQueryText b/c TextInput.onSubmitEditing -> history.push -> navParams.species
  };

  get queryDesc(): string {
    return matchQuery(this.query, {
      none:    ()                    => 'none',
      random:  ({filters, seed})     => `random/${seed}`,
      species: ({filters, species})  => species,
      rec:     ({filters, sourceId}) => showSourceId(sourceId),
    });
  }

  loadRecsFromQuery = async () => {
    if (
      (
        // Noop if we don't know f_preds_cols yet (assume we'll be called again)
        !this.state.f_preds_cols ||
        // Noop if this.query is already shown
        deepEqual(this.query, this.state.recsQueryShown) ||
        // Noop if this.query is already in progress
        deepEqual(this.query, this.state.recsQueryInProgress) ||
        // Noop if this.query isn't valid
        matchQuery(this.query, {
          none:    ()                    => false,
          random:  ({filters, seed})     => false,
          species: ({filters, species})  => species  === '',
          rec:     ({filters, sourceId}) => sourceId === '',
        })
      ) && (
        // But don't noop if filters/limits have changed
        //  - XXX(put_all_query_state_in_location)
        !this.state.refreshQuery
      )
      // TODO(put_all_query_state_in_location): Refresh on place change
    ) {
      log.info('loadRecsFromQuery: Skipping');
    } else {
      log.info('loadRecsFromQuery', () => pretty({query: this.query}));

      // Set loading state
      //  - TODO Fade previous recs instead of showing a blank screen while loading
      log.info("loadRecsFromQuery: state.recs = 'loading'");
      this.setState({
        recs: 'loading',
        recsQueryInProgress: this.query,
        refreshQuery: false,
      });
      await this.releaseSounds(); // (Safe to do after clearing state.recs, since it uses this.soundsCache)

      // Can't use window functions until sqlite ≥3.25.x
      //  - TODO Waiting on: https://github.com/litehelpers/Cordova-sqlite-storage/issues/828

      const timer = new Timer();
      const _setRecs = ({recs}: {recs: 'loading' | Array<Rec>}): void => {
        log.info(`loadRecsFromQuery: state.recs = ${recs === 'loading' ? recs : `(${recs.length} recs)`}`);
        this.setState({
          recs,
          recsQueryShown: this.query,
          recsQueryTime: timer.time(),
        });
      };

      // Global filters
      //  - TODO(put_all_query_state_in_location)
      const qualityFilter = (table: string) => (
        sqlf`and ${SQL.raw(table)}.quality in (${this.state.filterQuality})`
      );
      const placeFilter   = (table: string) => matchNull(this.props.place, {
        null: ()    => '',
        x:    place => sqlf`and ${SQL.raw(table)}.species in (${place.species})`,
      });

      await matchQuery(this.query, {

        none: async () => {
          log.info(`loadRecsFromQuery: Got QueryNone, staying in 'loading'...`);
          _setRecs({recs: 'loading'});
        },

        // TODO Weight species uniformly (e.g. select random species, then select random recs)
        // TODO Get deterministic results from seed [how? sqlite doesn't support random(seed) or hash()]
        random: async ({filters, seed}) => {
          log.info(`loadRecsFromQuery: Querying random recs`, {seed});
          await querySql<Rec>(this.db!, sqlf`
            select *
            from (
              select
                *,
                cast(taxon_order as real) as taxon_order_num
              from search_recs S
              where true
                ${SQL.raw(placeFilter('S'))}
                ${SQL.raw(qualityFilter('S'))}
              order by
                random()
              limit ${this.state.n_recs}
            )
            order by
              taxon_order_num asc,
              source_id desc
          `)(async results => {
            const recs = results.rows.raw();
            _setRecs({recs});
          });
        },

        species: async ({filters, species}) => {
          log.info('loadRecsFromQuery: Querying recs for species', {species});
          await querySql<Rec>(this.db!, sqlf`
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
          `)(async results => {
            const recs = results.rows.raw();
            _setRecs({recs});
          });
        },

        rec: async ({filters, sourceId}) => {
          log.info('loadRecsFromQuery: Loading recs for query_rec', {sourceId});

          // Compute top n_per_sp recs per species by d_pc (cosine_distance)
          //  - TODO Replace with window functions after sqlite upgrade
          //    - https://github.com/litehelpers/Cordova-sqlite-storage/issues/828
          //  - Alternative approach w/o window functions:
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
          const query_rec = await this.loadRec(sourceId);
          // log.debug('loadRecsFromQuery: query_rec', rich(query_rec)); // XXX Debug

          // Bail if sourceId not found (e.g. from persisted history)
          if (!query_rec) {
            log.warn(`loadRecsFromQuery: sourceId not found: ${sourceId}`);
            _setRecs({recs: 'loading'});
            return;
          }

          // Read sp_p's (species probs) from query_rec.f_preds_*
          //  - Filter by place.species (else too few results downstream after filtering in sql)
          const sp_ps: Map<string, number> = new Map(zipSame(
            this.props.modelsSearch.classes_,
            f_preds_cols.map(c => rec_f_preds(query_rec)[c]),
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

          // Rank species by slp (slp asc b/c sgn(slp) ~ -sgn(sp_p))
          const includeSpecies: (species: Species) => boolean = matchNull(this.props.place, {
            null: ()    => (species: Species) => true, // No place selected -> include all species
            x:    place => _.bind(Set.prototype.has, new Set(place.species)),
          });
          const topSlps: Array<{species: string, slp: number}> = (
            _(Array.from(slps.entries()))
            .map(([species, slp]) => ({species, slp}))
            .filter(({species}) => includeSpecies(species))
            .sortBy(({slp}) => slp)
            .slice(0, n_sp + 1) // FIXME +1 else we get n_sp-1 species -- why?
            .value()
          );

          // Inject sql table: slps -> (species, slp)
          //  - FIXME sql syntax error if topSlps is empty
          const tableSlp = sqlf`
            select column1 as species, column2 as slp from (values ${SQL.raw(topSlps
              // .slice(0, 2) // XXX Debug: smaller query
              .map(({species, slp}) => sqlf`(${species}, ${slp})`)
              .join(', ')
            )})
          `;

          // Construct queries for each species
          //  - TODO Shorter query: refactor sqlCosineDist expr (1 per topSlps) into a shared `with` table (1 total)
          const sqlPerSpecies = (topSlps
            // .slice(0, 2) // XXX Debug: smaller query
            .map(({species, slp}) => sqlf`
              select
                S.*,
                ${SQL.raw(sqlCosineDist)} as d_pc
              from search_recs S
                left join (select * from search_recs where source_id = ${sourceId}) Q on true -- Only 1 row in Q
              where true
                and S.species = ${species}
                ${SQL.raw(placeFilter('S'))} -- Empty subquery for species outside of placeFilter
                ${SQL.raw(qualityFilter('S'))}
                and S.source_id != ${sourceId} -- Exclude query_rec from results
              order by
                d_pc asc
              limit ${n_per_sp}
            `)
          );

          // Construct query
          const sql = sqlf`
            select
              S.*,
              coalesce(slp.slp, 1e38) as slp
            -- Must select * from (...) else union complains about nested order by
            from (${SQL.raw(sqlPerSpecies.map(x => `select * from (${x})`).join(' union all '))}) S
              left join (${SQL.raw(tableSlp)}) slp on S.species = slp.species
            order by
              slp asc,
              d_pc asc
            limit ${n_recs}
          `;

          // Run query
          log.info('loadRecsFromQuery: Querying recs for query_rec', {sourceId});
          await querySql<Rec>(this.db!, sql, {
            // logTruncate: null, // XXX Debug
          })(async results => {
            const recs = results.rows.raw();

            // HACK Inject query_rec as first result so it's visible at top
            //  - TODO Replace this with a proper display of query_rec at the top
            _setRecs({recs: [query_rec, ...recs]});

          });

        },

      });

    }
  }

  loadRec = async (sourceId: SourceId): Promise<Rec> => {
    log.info('loadRec', {sourceId});
    return await matchSourceId(sourceId, {
      xc: async ({xc_id}) => {
        return await querySql<XCRec>(this.db!, sqlf`
          select *
          from search_recs
          where source_id = ${sourceId}
        `)(async results => {
          const [rec] = results.rows.raw();
          return rec; // TODO XCRec
        });
      },
      user: async ({name, clip}) => {
        // Predict f_preds from audio
        //  - Audio not spectro: model uses its own f_bins=40, separate from this.props.f_bins=80 that we use to draw while recording
        const f_preds = await log.timedAsync('loadRec: f_preds', async () => {
          return await NativeSearch.f_preds(UserRec.audioPath(name));
        });
        if (f_preds === null) {
          throw `loadRec: Unexpected null f_preds (audio < nperseg), for sourceId[${sourceId}]`;
        } else {
          let userRec: UserRec = {
            source_id:  sourceId,
            duration_s: NaN, // TODO(stretch_user_rec)
            f_preds,
            // Mock the xc fields
            //  - TODO Clean up junk fields after splitting subtypes XCRec, UserRec <: Rec
            xc_id:               -1,
            species:             'unknown',
            species_taxon_order: '_UNK',
            species_com_name:    'unknown',
            species_sci_name:    'unknown',
            recs_for_sp:         -1,
            quality:             'no score',
            month_day:           '',
            place:               '',
            place_only:          '',
            state:               '',
            state_only:          '',
            recordist:           '',
            license_type:        '',
            remarks:             '',
          };
          // HACK Read duration_s from audio file
          //  - (Probably simpler for NativeSpectro to return this from .stop)
          userRec.duration_s = (await this.getOrAllocateSoundAsync(userRec)).getDuration();
          return userRec
        }
      },
    });
  }

  releaseSounds = async () => {
    log.info(`releaseSounds: Releasing ${this.soundsCache.size} cached sounds`);
    await Promise.all(
      Array.from(this.soundsCache).map(async ([sourceId, soundAsync]) => {
        log.debug('releaseSounds: Releasing sound',
          showSourceId(sourceId), // Noisy (but these log lines don't de-dupe anyway when rndebugger timestamps are shown)
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
        showSourceId(rec.source_id), // Noisy (but these log lines don't de-dupe anyway when rndebugger timestamps are shown)
      );
      // Allocate + cache sound resource
      //  - Cache the promise so that get+set is atomic, else we race and allocate multiple sounds per rec.source_id
      //  - (Observable via log counts in the console: if num alloc > num release, then we're racing)
      this.soundsCache.set(rec.source_id, Sound.newAsync(
        Rec.audioPath(rec),
        Sound.MAIN_BUNDLE, // TODO(asset_main_bundle): Why implicitly rel to MAIN_BUNDLE?
      ));
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
            startTime = this.spectroTimeFromX(sound, x, absoluteX);
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

  spectroTimeFromX = (sound: Sound, x: number, absoluteX: number): number => {
    const {contentOffset} = this._scrollViewState;
    const duration_s = sound.getDuration();
    const {width} = this.spectroDim(duration_s);
    const {audio_s} = this.props.serverConfig.api.recs.search_recs.params;
    const time = x / width * audio_s;
    // log.debug('spectroTimeFromX', () => pretty({time, x, absoluteX, contentOffset, width, audio_s, duration_s}));
    return time;
  }

  spectroXFromTime = (sound: Sound, time: number): number => {
    const {contentOffset} = this._scrollViewState;
    const duration_s = sound.getDuration();
    const {width} = this.spectroDim(duration_s);
    const {audio_s} = this.props.serverConfig.api.recs.search_recs.params;
    const x = time / audio_s * width;
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
      <this.GenericModal>
        {/* TODO(browse_species): Add title with place name */}
        <FlatList <SpeciesMetadata>
          style={{
            flex: 1,
            height: '100%',
            width: 300, // HACK(browse_species): ~95%
          }}
          data={_.sortBy(
            matchNull(this.props.place, {
              null: () => [],
              x: ({species}) => _.flatMap(species, x => (
                matchUndefined(this.props.ebird.speciesMetadataFromSpecies.get(x), {
                  undefined: () => [],
                  x:         x  => [x],
                })
              )),
            }),
            x => parseFloat(x.taxon_order),
          )}
          keyExtractor={x => x.shorthand}
          renderItem={({item: speciesMetadata, index}) => (
            <RectButton
              style={{
                width: '100%',
                paddingVertical: 3,
              }}
              onPress={() => {
                this.setState({
                  showGenericModal: null, // Dismiss modal
                });
                // Show species
                this.props.go('search', {path: `/species/${speciesMetadata.shorthand}`});
              }}
            >
              {/* TODO(browse_species): Better formatting */}
              <Text style={[material.captionObject, {color: 'black'}]}>
                {speciesMetadata.com_name}
              </Text>
              <Text style={[material.captionObject, {fontSize: 8}]}>
                {speciesMetadata.species_group}
              </Text>
            </RectButton>
          )}
        />
      </this.GenericModal>
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
          title={`${rec.species}/${showSourceId(rec.source_id)}`}
        />

        {/* Spectro */}
        <Animated.Image
          style={{
            // ...this.spectroDim(rec.duration_s), // XXX Bad(info_modal)
            height: this.spectroDim(rec.duration_s).height,
            width: '100%',
          }}
          foo
          resizeMode='stretch' // TODO(info_modal) Wrap to show whole spectro i/o stretching
          source={{uri: Rec.spectroPath(rec)}}
        />

        <Separator/>
        <View style={{flexDirection: 'row'}}>
          {this.ActionModalButtons({actions: [
            {
              ...defaults,
              label: `${rec.species}`,
              iconName: 'search',
              buttonColor: iOSColors.blue,
              onPress: () => this.props.go('search', {path: `/species/${encodeURIComponent(rec.species)}`}),
            }, {
              ...defaults,
              label: `${showSourceId(rec.source_id)}`,
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
              label: `${rec.species}`,
              iconName: 'x',
              buttonColor: iOSColors.red,
              onPress: () => this.setState((state: State, props: Props) => ({
                excludeSpecies: [...state.excludeSpecies, rec.species],
              })),
            }, {
              ...defaults,
              label: `${showSourceId(rec.source_id)}`,
              iconName: 'x',
              buttonColor: iOSColors.red,
              onPress: () => this.setState((state: State, props: Props) => ({
                excludeRecIds: [...state.excludeRecIds, rec.source_id],
              })),
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
            label: `Save to list (${showSourceId(rec.source_id)})`,
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
              <Text style={{
                ...material.captionObject,
                fontWeight: 'bold',
              }}>{c}:</Text> {MetadataColumnsBelow[c](rec)}
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
          showGenericModal: () => this.BrowseModal()
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
                buttonColor: iOSColors.orange,
                onPress: () => {},
              }, {
                label: 'Similar recs only (ignoring species match)',
                iconName: 'chevrons-down',
                buttonColor: iOSColors.orange,
                onPress: () => {},
              },
            ]} />
          )
        })}
      />
      {/* Query that returns no results [XXX For dev] */}
      <this.BottomControlsButton
        help='Blank'
        iconProps={{name: 'power'}}
        onPress={() => this.props.go('search', {path: `/species/_BLANK`})} // HACK No results via non-existent species
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
        active={this.props.showMetadataLeft}
        // iconProps={{name: 'file-minus'}}
        iconProps={{name: 'sidebar'}}
        onPress={() => this.props.settings.toggle('showMetadataLeft')}
        onLongPress={() => this.setState({
          showGenericModal: () => (
            <this.ActionModal title='Show columns' actions={
              objectKeysTyped(MetadataColumnsLeft).map(c => ({
                label: c,
                textColor: iOSColors.black,
                buttonColor: this.props.metadataColumnsLeft.includes(c) ? iOSColors.tealBlue : iOSColors.customGray,
                buttonStyle: {
                  marginVertical: 2,
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
                label: c,
                textColor: iOSColors.black,
                buttonColor: this.props.metadataColumnsBelow.includes(c) ? iOSColors.tealBlue : iOSColors.customGray,
                buttonStyle: {
                  marginVertical: 2,
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
        active={this.props.seekOnPlay}
        iconProps={{name: 'crosshair'}}
        onPress={() => this.props.settings.toggle('seekOnPlay')}
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
    style?: TextStyle,
  }) => (
    <Text style={{
      alignSelf: 'center', // (horizontal)
      marginBottom: 5,
      ...material.titleObject,
      ...props.style,
    }}>
      {props.title}
    </Text>
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
      <this.GenericModalTitle style={props.titleStyle} title={props.title} />
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
            // @ts-ignore [Why doesn't this typecheck?]
            ref={this.scrollViewRef as RefObject<Component<SectionListStatic<Rec>, any, any>>}

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
              {this.recsOrEmpty.map((rec, recIndex) => [

                // Rec row (with editing buttons)
                <Animated.View
                  key={`row-${recIndex}-${rec.source_id}`}
                  style={{
                    flex: 1, flexDirection: 'row',
                    // Alternating row colors
                    // backgroundColor: recIndex % 2 == 0 ? iOSColors.white : iOSColors.lightGray,
                    // Compact controls/labels when zoom makes image smaller than controls/labels
                    ...(this.props.showMetadataBelow ? {} : {
                      height: this.spectroDim(rec.duration_s).height,
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
                            height: this.spectroDim(rec.duration_s).height,
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
                        <TapGestureHandler onHandlerStateChange={this.toggleRecPlaying(rec)}>
                          <Animated.View>

                            {/* Image */}
                            <Animated.Image
                              style={this.spectroDim(rec.duration_s)}
                              resizeMode='stretch'
                              source={{uri: Rec.spectroPath(rec)}}
                            />

                            {/* Start time cursor (if playing + startTime) */}
                            {this.recIsPlaying(rec.source_id, this.state.playing) && (
                              this.state.playing!.startTime && (
                                <View style={{
                                  position: 'absolute', width: 1, height: '100%',
                                  left: this.spectroXFromTime(this.state.playing!.sound, this.state.playing!.startTime!),
                                  backgroundColor: iOSColors.gray,
                                }}/>
                              )
                            )}

                            {/* Progress time cursor (if playing + playingCurrentTime) */}
                            {this.recIsPlaying(rec.source_id, this.state.playing) && (
                              this.state.playing!.startTime && this.state.playingCurrentTime !== undefined && (
                                <View style={{
                                  position: 'absolute', width: 1, height: '100%',
                                  left: this.spectroXFromTime(this.state.playing!.sound, this.state.playingCurrentTime),
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
                              <Text style={{
                                ...material.captionObject,
                                fontWeight: 'bold',
                              }}>{c}:</Text> {MetadataColumnsBelow[c](rec)}
                            </MetadataText>
                          ))}
                        </View>
                      )}

                    </Animated.View>
                  </LongPressGestureHandler>

                </Animated.View>

              ])}
            </View>

            {/* Footer */}
            <View style={{
              ...Styles.center,
              width: Dimensions.get('window').width,
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
            Recs: {this.state.recs === 'loading'
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

// TODO Why is this slow to respond after keyboard shows? -- adding logging to find the bottleneck
interface KeyboardDismissingViewState {
  isKeyboardShown: boolean;
}
export class KeyboardDismissingView extends PureComponent<RN.ViewProps, KeyboardDismissingViewState> {
  log = new Log('KeyboardDismissingView');
  state = {
    isKeyboardShown: false,
  };
  _keyboardDidShowListener?: {remove: () => void};
  _keyboardDidHideListener?: {remove: () => void};
  componentDidMount = () => {
    this._keyboardDidShowListener = Keyboard.addListener('keyboardDidShow', this.keyboardDidShow);
    this._keyboardDidHideListener = Keyboard.addListener('keyboardDidHide', this.keyboardDidHide);
  };
  componentWillUnmount = () => {
    this._keyboardDidShowListener!.remove();
    this._keyboardDidHideListener!.remove();
  };
  componentDidUpdate = (prevProps: RN.ViewProps, prevState: KeyboardDismissingViewState) => {
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
