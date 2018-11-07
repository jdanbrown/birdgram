import * as d3sc from 'd3-scale-chromatic';
import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import RN, {
  ActivityIndicator, Animated, Dimensions, FlatList, GestureResponderEvent, Image, ImageStyle, Keyboard,
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
import { NavigationScreenProps } from 'react-navigation';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
import stringHash from "string-hash";
const fs = RNFB.fs;

import { ActionSheetBasic } from './ActionSheets';
import { Settings, ShowMetadata } from '../settings';
import { config } from '../config';
import {
  InlineMetadataColumn, InlineMetadataColumns, ModelsSearch, Quality, Rec, rec_f_preds, Rec_f_preds, RecId, SearchRecs,
  ServerConfig, shortRecId,
} from '../datatypes';
import { log, puts, tap } from '../log';
import { Nav, navigate, NavParams, NavParamsSearch, ScreenProps } from '../nav';
import Sound from '../sound';
import { querySql, SQL, sqlf } from '../sql';
import { StyleSheet } from '../stylesheet';
import { debugStyle, LabelStyle, labelStyles } from '../styles';
import {
  all, any, chance, Clamp, deepEqual, Dim, finallyAsync, getOrSet, global, json, mapMapValues, match, noawait,
  objectKeysTyped, Point, pretty, round, setStateAsync, Style, Styles, TabBarBottomConstants, zipSame,
} from '../utils';

const sidewaysTextWidth = 14;
const recEditingButtonWidth = 35;
const someExampleSpecies = 'GREG,LASP,HOFI,NOFL,GTGR,SWTH,GHOW' // XXX Dev

interface ScrollViewState {
  contentOffset: Point;
  // (More fields available in NativeScrollEvent)
}

type Query = QuerySpecies | QueryRecId;
type QuerySpecies = {kind: 'species', species: string};
type QueryRecId   = {kind: 'recId',   recId: string};
function matchQuery<X>(query: Query, cases: {
  species: (query: QuerySpecies) => X,
  recId:   (query: QueryRecId)   => X,
}): X {
  switch (query.kind) {
    case 'species': return cases.species(query);
    case 'recId':   return cases.recId(query);
  }
}

interface Props extends NavigationScreenProps<NavParams, any> {
  navigation:        Nav;
  screenProps:       ScreenProps;
  spectroBase:       Dim<number>;
  spectroScaleClamp: Clamp<number>;
}

interface State {
  scrollViewKey: string;
  scrollViewState: ScrollViewState;
  showGenericModal: null | (() => Element);
  showFilters: boolean;
  showHelp: boolean;
  totalRecs?: number;
  f_preds_cols?: Array<string>;
  // TODO Persist filters with settings
  //  - Top-level fields instead of nested object so we can use state merging when updating them in isolation
  filterQueryText?: string;
  filterQuality: Array<Quality>;
  filterLimit: number;
  excludeSpecies: Array<string>;
  excludeRecIds: Array<string>;
  status: {
    loading: boolean,
    queryInProgress?: Query,
    queryShown?: Query;
  },
  recs: Array<Rec>;
  playing?: {
    rec: Rec,
    sound: Sound,
    startTime?: number,
    // progressTime?: number,
  };
  playingCurrentTime?: number;
  // Sync from/to Settings (1/3)
  spectroScale: number;
};

export class SearchScreen extends Component<Props, State> {

  // Getters for prevProps
  _serverConfig = (props?: Props): ServerConfig    => { return (props || this.props).screenProps.serverConfig; }
  _modelsSearch = (props?: Props): ModelsSearch    => { return (props || this.props).screenProps.modelsSearch; }
  _settings     = (props?: Props): Settings        => { return (props || this.props).screenProps.settings; }
  _nav          = (props?: Props): Nav             => { return (props || this.props).navigation; }
  _navParamsAll = (props?: Props): NavParams       => { return this._nav(props).state.params || {}; }
  _navParams    = (props?: Props): NavParamsSearch => { return this._navParamsAll(props).search || {}; }

  // Getters for this.props
  get serverConfig (): ServerConfig    { return this._serverConfig(); }
  get modelsSearch (): ModelsSearch    { return this._modelsSearch(); }
  get settings     (): Settings        { return this._settings(); }
  get nav          (): Nav             { return this._nav(); }
  get navParamsAll (): NavParams       { return this._navParamsAll(); }
  get navParams    (): NavParamsSearch { return this._navParams(); }

  // Getters for this.state
  get filters(): object { return _.pickBy(this.state, (v, k) => k.startsWith('filter')); }

  static defaultProps = {
    spectroBase:       {height: 20, width: Dimensions.get('window').width},
    spectroScaleClamp: {min: 1, max: 8},
  };

  // Else we have to do too many setState's, which makes animations jump (e.g. ScrollView momentum)
  _scrollViewState: ScrollViewState = {
    contentOffset: {x: 0, y: 0},
  };

  state: State = {
    scrollViewKey: '',
    scrollViewState: this._scrollViewState,
    showGenericModal: null,
    showFilters: false,
    showHelp: false,
    filterQuality: ['A', 'B'],
    filterLimit: 30, // TODO How big vs. fast? (-> Settings with sane default)
    excludeSpecies: [],
    excludeRecIds: [],
    status: {loading: true},
    recs: [],
    // Sync from/to Settings (2/3)
    spectroScale: this.settings.spectroScale,
  };

  db?: SQLiteDatabase;
  soundsCache: Map<RecId, Promise<Sound> | Sound> = new Map();

  saveActionSheet: RefObject<ActionSheet> = React.createRef();
  addActionSheet:  RefObject<ActionSheet> = React.createRef();
  sortActionSheet: RefObject<ActionSheet> = React.createRef();

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
    log.info('SearchScreen.componentDidMount');
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
      log.error(`DB file not found: ${dbFilename}`);
    } else {
      const dbLocation = `~/${dbFilename}`; // Relative to app bundle (copied into the bundle root by react-native-asset)
      this.db = await SQLite.openDatabase({
        name: dbFilename,               // Just for SQLite bookkeeping, I think
        readOnly: true,                 // Else it will copy the (huge!) db file from the app bundle to the documents dir
        createFromLocation: dbLocation, // Else readOnly will silently not work
      });
    }

    // Query db size (once)
    await querySql<{totalRecs: number}>(this.db!, `
      select count(*) as totalRecs
      from search_recs
    `)(async results => {
      log.info('SearchScreen.componentDidMount: state.totalRecs');
      const [{totalRecs}] = results.rows.raw();
      await setStateAsync(this, {
        totalRecs,
      });
    });

    // Query f_preds_* cols (once)
    await querySql<Rec>(this.db!, `
      select *
      from search_recs
      limit 1
    `)(async results => {
      log.info('SearchScreen.componentDidMount: state.f_preds_cols');
      const [rec] = results.rows.raw();
      const n = Object.keys(rec).filter(k => k.startsWith('f_preds_')).length;
      // Reconstruct strings from .length to enforce ordering
      const f_preds_cols = _.range(n).map(i => `f_preds_${i}`);
      await setStateAsync(this, {
        f_preds_cols,
      });
    });

    // Query recs (from navParams.species)
    log.debug('componentDidMount -> loadRecsFromQuery');
    await this.loadRecsFromQuery();

  }

  // Before a component is removed from the DOM and destroyed
  //  - Commit phase (impure, may read/write DOM, called once per commit)
  //  - Best practices
  //    - Do unsubscribe listeners / cancel timers (created in componentDidMount)
  //    - Don't setState(), since no more render() will happen for this instance
  componentWillUnmount = async () => {
    log.info('SearchScreen.componentWillUnmount');

    // Tell other apps we're no longer using the audio device
    Sound.setActive(false);

    // Release cached sound resources
    await this.releaseSounds();

    // Clear timers
    timer.clearTimeout(this);

  }

  // Update on !deepEqual instead of default (any setState() / props change, even if data is the same) else any setState
  // in componentDidUpdate would trigger an infinite update loop
  shouldComponentUpdate(nextProps: Props, nextState: State): boolean {
    const ret = !deepEqual(this.props, nextProps) || !deepEqual(this.state, nextState);
    // log.debug('shouldComponentUpdate', {ret});
    return ret;
  }

  // After props/state change; not called for the initial render()
  //  - Commit phase (impure, may read/write DOM, called once per commit)
  //  - Best practices
  //    - Do operate on DOM in response to changed props/state
  //    - Do fetch data, conditioned on changed props/state (else update loops)
  //    - Do setState(), conditioned on changed props (else update loops)
  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('SearchScreen.componentDidUpdate', this.props, this.state);

    // Reset view state if query (= props.navigation.state.params.{search,recId}) changed
    //  - TODO Pass props.key to reset _all_ state? [https://reactjs.org/blog/2018/06/07/you-probably-dont-need-derived-state.html#recap]
    if (!deepEqual(this.query, this._query(prevProps))) {
      log.info('SearchScreen.componentDidUpdate: Reset view state');
      await setStateAsync(this, {
        filterQueryText: undefined,
        excludeSpecies: [],
        excludeRecIds: [],
      });
    }

    // Else _scrollViewState falls behind on non-scroll/non-zoom events (e.g. +/- buttons)
    this._scrollViewState = this.state.scrollViewState;

    // Sync from/to Settings (3/3)
    //  - These aren't typical: we only use this for (global) settings keys that we also keep locally in state so we can
    //    batch-update them with other local state keys (e.g. global spectroScale + local scrollViewKey)
    //  - TODO Is this a good pattern for "setState(x,y,z) locally + settings.set(x) globally"?
    if (this.state.spectroScale !== prevState.spectroScale) {
      noawait(this.settings.set('spectroScale', this.state.spectroScale));
    }

    // Query recs (from updated navParams.species)
    //  - (Will noop if deepEqual(query, state.status.queryShown))
    log.debug('componentDidUpdate -> loadRecsFromQuery');
    await this.loadRecsFromQuery();

  }

  get spectroDim(): Dim<number> {
    return {
      height: this.props.spectroBase.height * this.state.spectroScale,
      width:  this.scrollViewContentWidths.image,
    };
  }

  // Manually specify widths of all components that add up to the ScrollView content width so we can explicitly compute
  // and set it, else the ScrollView won't scroll horizontally (the overflow direction)
  //  - https://github.com/facebook/react-native/issues/8579#issuecomment-233162695
  //  - I also tried using onLayout to automatically get subcomponent widths from the DOM instead of manually
  //    maintaining them all here, but that got gnarly and I bailed (bad onLayout/setState interactions causing infinite
  //    update loops, probably because I'm missing conditions in lifecycle methods like componentDidUpdate)
  get scrollViewContentWidth() { return _.sum(_.values(this.scrollViewContentWidths)); }
  get scrollViewContentWidths() {
    return {
      // NOTE Conditions duplicated elsewhere (render, spectroDim, ...)
      recEditing:     !this.settings.editing ? 0 : (
        recEditingButtonWidth * this._recEditingButtons.length
      ),
      sidewaysText:   this.settings.showMetadata === 'full' ? 0 : (
        sidewaysTextWidth
      ),
      debugInfo:      !this.settings.showDebug || this.settings.showMetadata !== 'inline' ? 0 : 70,
      inlineMetadata: this.settings.showMetadata !== 'inline' ? 0 : 50,
      image:          (
        this.props.spectroBase.width * this.state.spectroScale - (
          this.settings.showMetadata === 'full' ? 0 : sidewaysTextWidth
        )
      ),
    };
  }

  get query(): Query { return this._query(); }
  _query = (props?: Props): Query => {
    const {species, recId} = this._navParams(props || this.props);
    return (
      species ? {kind: 'species', species} :
      recId ? {kind: 'recId', recId} :
      {kind: 'species', species: someExampleSpecies}
    );
    // We ignore state.filterQueryText b/c TextInput.onSubmitEditing -> navigate.search -> navParams.species
  };

  get queryDesc(): string {
    // TODO Make match() typecheck for this use case
    switch (this.query.kind) {
      case 'species': return this.query.species;
      case 'recId':   return shortRecId(this.query.recId);
    }
  }

  loadRecsFromQuery = async () => {
    if (
      // Noop if we don't know f_preds_cols yet (assume we'll be called again)
      this.state.f_preds_cols &&
      // Noop if this.query is already shown
      !deepEqual(this.query, this.state.status.queryShown) &&
      // Noop if this.query is already in progress
      !deepEqual(this.query, this.state.status.queryInProgress) &&
      // Noop if this.query isn't valid
      matchQuery(this.query, {
        species: ({species}) => species !== '',
        recId:   ({recId})   => recId   !== '',
      })
    ) {
      log.info('loadRecsFromQuery', pretty({query: this.query, status: this.state.status}));

      // Set loading state
      //  - TODO Fade previous recs instead of showing a blank screen while loading
      log.info('loadRecsFromQuery: state.status.loading = true');
      await setStateAsync(this, {
        status: {
          loading: true,
          queryInProgress: this.query,
        },
        recs: [],
      });
      await this.releaseSounds(); // (Safe to do after clearing state.recs, since it uses this.soundsCache)

      // Can't use window functions until sqlite ≥3.25.x
      //  - TODO Waiting on: https://github.com/litehelpers/Cordova-sqlite-storage/issues/828

      const _setRecs = async (recs: Array<Rec>): Promise<void> => {
        log.info('loadRecsFromQuery: state.status.loading = false');
        await setStateAsync(this, {
          status: {
            loading: false,
            queryShown: this.query,
          },
          recs,
        });
      };

      await matchQuery(this.query, {

        species: async ({species}) => {
          await querySql<Rec>(this.db!, sqlf`
            select *
            from (
              select
                *,
                cast(taxon_order as real) as taxon_order_num
              from search_recs
              where
                species in (${species.split(',').map(x => _.trim(x).toUpperCase())}) and
                quality in (${this.state.filterQuality})
              order by
                xc_id desc
              limit ${this.state.filterLimit}
            )
            order by
              taxon_order_num asc,
              xc_id desc
          `)(async results => {
            const recs = results.rows.raw();
            await _setRecs(recs);
          });
        },

        recId: async ({recId}) => {

          // TODO(nav_rec_id)
          //  - TODO Top k by sp (w/o window functions)
          //    - [x] Query query_rec from db.search_recs
          //      - query_rec.preds is query_sp_p (preds from search model)
          //    - [ ] Sort .preds and pick top n species (to query for)
          //    - [ ] Query results from db.search_recs:
          //      - union all for each species:
          //        - select
          //        - top k s

          const f_preds_cols: string[] = this.state.f_preds_cols || [];

          // Load query_rec from db
          const query_rec = await this.loadRec(recId);

          // sp_p: "species prob"
          const sp_ps: Map<string, number> = new Map(zipSame(
            this.modelsSearch.classes_,
            f_preds_cols.map(c => rec_f_preds(query_rec)[c]),
          ));

          // slp: "species log prob"
          const slp = (sp_p: number): number => Math.abs(-Math.log(sp_p)) // (abs for 1->0 i/o -0)
          const slps: Map<string, number> = mapMapValues(sp_ps, slp);

          const sqlSlp = SQL.raw(
            Array.from(slps)
            .map(([species, slp]) => sqlf`select ${species} as species, ${slp} as slp`)
            .join(' union all ')
          );

          // Manually compute dot(S.f_preds_*, Q.f_preds_*)
          const sqlDot = SQL.raw(
            f_preds_cols.map(c => `S.${c}*Q.${c}`).join(' + ') || '0'
          );

          await querySql<Rec>(this.db!, sqlf`
            select
              S.*,
              coalesce(slp.slp, 1e38) as slp,
              -- Manually compute cosine_distance(S.f_preds_*, Q.f_preds_*):
              --    cosine_distance(x,y) = 1 - dot(x,y) / norm(x) / norm(y)
              1 - (${sqlDot}) / S.norm_f_preds / Q.norm_f_preds as d_pc
            from search_recs S
              left join (select * from search_recs where id = ${recId}) Q on true -- Only 1 row in Q
              left join (${sqlSlp}) slp on S.species = slp.species
            where true
              and S.quality in (${this.state.filterQuality})
              and S.id != ${recId} -- Exclude query_rec from results
            order by
              slp asc,
              d_pc asc
            limit ${this.state.filterLimit}
          `)(async results => {
            const recs = results.rows.raw();

            // Add query_rec as first result
            //  - TODO Replace this with a proper display of query_rec at the top
            await _setRecs([
              query_rec,
              ...recs,
            ]);

          });

        },

      });

    }
  }

  loadRec = async (id: string): Promise<Rec> => {
    return await querySql<Rec>(this.db!, sqlf`
      select *
      from search_recs
      where id = ${id}
    `)(async results => {
      const [rec] = results.rows.raw();
      return rec;
    });
  }

  releaseSounds = async () => {
    log.info(`Releasing ${this.soundsCache.size} cached sounds`);
    await Promise.all(
      Array.from(this.soundsCache).map(async ([recId, soundAsync]) => {
        log.debug('Releasing sound',
          shortRecId(recId), // Noisy (unless rndebugger timestamps are shown, in which case these log lines don't de-dupe anyway)
        );
        (await soundAsync).release();
      }),
    );
    this.soundsCache = new Map();
  }

  getOrAllocateSoundAsync = async (rec: Rec): Promise<Sound> => {
    // Is sound already allocated (in the cache)?
    let soundAsync = this.soundsCache.get(rec.id);
    if (!soundAsync) {
      log.debug('Allocating sound',
        shortRecId(rec.id), // Noisy (unless rndebugger timestamps are shown, in which case these log lines don't de-dupe anyway)
      );
      // Allocate + cache sound resource
      //  - Cache the promise so that get+set is atomic, else we race and allocate multiple sounds per rec.id
      //  - (Observable via log counts in the console: if num alloc > num release, then we're racing)
      this.soundsCache.set(rec.id, Sound.newAsync(
        Rec.audioPath(rec),
        Sound.MAIN_BUNDLE,
      ));
      soundAsync = this.soundsCache.get(rec.id);
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
        log.debug('toggleRecPlaying', pretty({x, recId: rec.id,
          playing: this.state.playing && {recId: this.state.playing.rec.id},
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
          log.info('Done: rec', rec.id);
          timer.clearInterval(this, 'playingCurrentTime');
          await setStateAsync(this, {
            playing: undefined,
          });
        };

        // Stop any recs that are currently playing
        if (playing) {
          const {rec, sound} = playing;
          global.sound = sound; // XXX Debug

          // Stop sound playback
          log.info('Stopping: rec', rec.id);
          await sound.stopAsync();
          await onDone();

        }

        // If touched rec was the currently playing rec, then we're done (it's stopped)
        // Else, play the (new) touched rec
        if (!this.recIsPlaying(rec.id, playing)) {
          const sound = await soundAsync;
          global.sound = sound; // XXX Debug

          // Compute startTime to seek rec (if enabled)
          let startTime;
          if (this.settings.seekOnPlay) {
            startTime = this.spectroTimeFromX(sound, x, absoluteX);
          } else {
            // startTime = 0; // TODO Show some kind of visual feedback when not seekOnPlay
          }

          // Play rec (if startTime is valid)
          if (!startTime || startTime < sound.getDuration()) {
            log.info('Playing: rec', rec.id);

            // setState
            await setStateAsync(this, {
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
            //  - WARNING Don't separate timers per rec.id until we resolve "FIXME Race conditions" above ("tap many")
            if (this.settings.playingProgressEnable && this.settings.playingProgressInterval !== 0) {
              timer.setInterval(this, 'playingCurrentTime',
                async () => {
                  const {seconds, isPlaying} = await sound.getCurrentTimeAsync();
                  if (isPlaying) {
                    await setStateAsync(this, {
                      playingCurrentTime: seconds,
                    });
                  }
                },
                this.settings.playingProgressInterval,
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
    const {spectroScale} = this.state;
    const {width} = this.spectroDim;
    const {audio_s} = this.serverConfig.api.recs.search_recs.params;
    const duration = sound.getDuration();
    const time = x / width * audio_s;
    // log.debug('spectroTimeFromX', pretty({time, x, absoluteX, contentOffset, width, spectroScale, audio_s, duration}));
    return time;
  }

  spectroXFromTime = (sound: Sound, time: number): number => {
    const {contentOffset} = this._scrollViewState;
    const {spectroScale} = this.state;
    const {width} = this.spectroDim;
    const {audio_s} = this.serverConfig.api.recs.search_recs.params;
    const duration = sound.getDuration();
    const x = time / audio_s * width;
    // log.debug('spectroXFromTime', pretty({x, time, contentOffset, width, spectroScale, audio_s, duration}));
    return x;
  }

  recIsPlaying = (recId: RecId, playing: undefined | {rec: Rec}): boolean => {
    return !playing ? false : playing.rec.id === recId;
  }

  onLongPress = (rec: Rec) => async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    if (state === Gesture.State.ACTIVE) {
      log.debug('onLongPress');
    }
  }

  onBottomControlsLongPress = async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    await match(state,
      [Gesture.State.ACTIVE, async () => await setStateAsync(this, {showHelp: true})],
      [Gesture.State.END,    async () => await setStateAsync(this, {showHelp: false})],
    )();
  }

  onMockPress = (rec: Rec) => async () => {
    console.log('renderLeftAction.onMockPress');
  }

  Filters = () => (
    <KeyboardDismissingView style={{width: '100%', height: '100%'}}>
      <View style={[
        styles.filtersModal,
        {marginBottom: TabBarBottomConstants.DEFAULT_HEIGHT},
      ]}>
        <TextInput
          style={styles.queryInput}
          value={this.state.filterQueryText}
          onChangeText={async x => await setStateAsync(this, {filterQueryText: x})}
          onSubmitEditing={() => navigate.search(this.nav, {species: this.state.filterQueryText})}
          autoCorrect={false}
          autoCapitalize='characters'
          enablesReturnKeyAutomatically={true}
          placeholder={this.queryDesc}
          returnKeyType='search'
        />
        <Text>Filters</Text>
        <Text>- quality</Text>
        <Text>- month</Text>
        <Text>- species likelihood [bucketed ebird priors]</Text>
        <Text>- rec text search [conflate fields]</Text>
        <RectButton onPress={async () => await setStateAsync(this, {showFilters: false})}>
          <View style={{padding: 10, backgroundColor: iOSColors.blue}}>
            <Text>Done</Text>
          </View>
        </RectButton>
      </View>
    </KeyboardDismissingView>
  );

  cycleMetadataFull = async () => {
    const next = (showMetadata: ShowMetadata) => match<ShowMetadata, ShowMetadata, ShowMetadata>(showMetadata,
      ['none',   'full'],
      ['inline', 'full'],
      ['full',   'none'],
    );
    await this.settings.set('showMetadata', next(this.settings.showMetadata));
  }

  cycleMetadataInline = async () => {
    const next = (showMetadata: ShowMetadata) => match<ShowMetadata, ShowMetadata, ShowMetadata>(showMetadata,
      ['none',   'inline'],
      ['inline', 'none'],
      ['full',   'inline'],
    );
    await this.settings.set('showMetadata', next(this.settings.showMetadata));
  }

  cycleMetadata = async () => {
    const next = (showMetadata: ShowMetadata) => match<ShowMetadata, ShowMetadata, ShowMetadata>(showMetadata,
      ['none',   'inline'],
      ['inline', 'full'],
      ['full',   'none'],
    );
    await this.settings.set('showMetadata', next(this.settings.showMetadata));

    // Scroll SectionList so that same ~top recs are showing after drawing with new item/section heights
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

  }

  scaleSpectros = async (delta: number) => {
    await setStateAsync(this, (state, props) => {
      // Round current spectroScale so that +1/-1 deltas snap back to non-fractional scales (e.g. after pinch zooms)
      const spectroScale = this.clampSpectroScaleY(Math.round(state.spectroScale) + delta);
      return {
        spectroScale,
        scrollViewState: {
          // FIXME Zoom in -> scroll far down+right -> use '-' button to zoom out -> scroll view clipped b/c contentOffset nonzero
          contentOffset: {
            x: this._scrollViewState.contentOffset.x * spectroScale / state.spectroScale,
            y: this._scrollViewState.contentOffset.y * spectroScale / state.spectroScale,
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
      {/* Back */}
      <this.BottomControlsButton
        help='Back'
        // TODO(nav_back): Add onLongPress to undo nav.goBack() [how?]
        // TODO(nav_back): Disable when no screen to go back to
        //  - How to detect whether nav.goBack() will do anything?
        //  - e.g. https://stackoverflow.com/questions/47915572/how-to-check-if-i-can-go-back-use-pop-on-react-native-navigation
        // disable={...}
        onPress={() => this.nav.goBack()}
        iconProps={{name: 'chevron-left'}}
      />
      {/* Filters */}
      <this.BottomControlsButton
        help='Filters'
        onPress={async () => await setStateAsync(this, {showFilters: true})}
        iconProps={{name: 'filter'}}
      />
      {/* Save as new list / add all to saved list / share list */}
      <this.BottomControlsButton
        help='Save'
        onPress={() => this.saveActionSheet.current!.show()}
        // iconProps={{name: 'share'}}
        // iconProps={{name: 'star'}}
        iconProps={{name: 'bookmark'}}
      />
      {/* Add species (select species manually) */}
      <this.BottomControlsButton
        help='Add'
        onPress={() => this.addActionSheet.current!.show()}
        // iconProps={{name: 'user-plus'}}
        // iconProps={{name: 'file-plus'}}
        // iconProps={{name: 'folder-plus'}}
        // iconProps={{name: 'plus-circle'}}
        iconProps={{name: 'plus'}}
      />
      {/* Toggle sort: species probs / rec dist / manual list */}
      <this.BottomControlsButton
        help='Sort'
        onPress={() => this.sortActionSheet.current!.show()}
        iconProps={{name: 'chevrons-down'}}
        // iconProps={{name: 'chevron-down'}}
        // iconProps={{name: 'arrow-down'}}
        // iconProps={{name: 'arrow-down-circle'}}
      />
      {/* XXX Dev: a temporary way to reset to recs from >1 species */}
      <this.BottomControlsButton
        help='Reset'
        onPress={() => navigate.search(this.nav, {})}
        // iconProps={{name: 'refresh-ccw'}}
        iconProps={{name: 'home'}}
      />
      {/* Toggle editing controls for rec/species */}
      <this.BottomControlsButton
        help='Edit'
        onPress={() => this.settings.toggle('editing')}
        active={this.settings.editing}
        iconProps={{name: 'sliders'}}
        // iconProps={{name: 'edit'}}
        // iconProps={{name: 'edit-2'}}
        // iconProps={{name: 'edit-3'}}
        // iconProps={{name: 'layout', style: Styles.flipBoth}}
      />
      {/* Cycle metadata: none / inline */}
      <this.BottomControlsButton
        help='Info'
        onPress={() => this.cycleMetadataInline()}
        onLongPress={() => setStateAsync(this, {
          showGenericModal: () => (
            <this.GenericModal title='Show columns' actions={
              objectKeysTyped(InlineMetadataColumns).map(c => ({
                label: c,
                textColor: iOSColors.black,
                buttonColor: this.settings.inlineMetadataColumns.includes(c) ? iOSColors.tealBlue : iOSColors.customGray,
                marginVertical: 2,
                dismiss: false,
                onPress: () => this.settings.update('inlineMetadataColumns', cs => (
                  cs.includes(c) ? _.without(cs, c) : [...cs, c]
                )),
              }))
            } />
          )
        })}
        active={this.settings.showMetadata === 'inline'}
        iconProps={{name: 'file-minus'}}
      />
      {/* Cycle metadata: none / full */}
      <this.BottomControlsButton
        help='Info'
        onPress={() => this.cycleMetadataFull()}
        active={this.settings.showMetadata === 'full'}
        iconProps={{name: 'file-text'}}
        // iconProps={{name: 'credit-card', style: Styles.flipVertical}}
        // iconProps={{name: 'sidebar', style: Styles.rotate270}}
      />
      {/* Toggle seekOnPlay crosshairs */}
      <this.BottomControlsButton
        help='Seek'
        onPress={() => this.settings.toggle('seekOnPlay')}
        active={this.settings.seekOnPlay}
        iconProps={{name: 'crosshair'}}
      />
      {/* Zoom more/fewer recs (spectro height) */}
      {/* - TODO Disable when spectroScale is min/max */}
      <this.BottomControlsButton
        help='Dense'
        disabled={this.state.spectroScale === this.props.spectroScaleClamp.min}
        onPress={async () => await this.scaleSpectros(-1)}
        // iconProps={{name: 'align-justify'}} // 4 horizontal lines
        iconProps={{name: 'zoom-out'}}
      />
      <this.BottomControlsButton
        help='Tall'
        disabled={this.state.spectroScale === this.props.spectroScaleClamp.max}
        onPress={async () => await this.scaleSpectros(+1)}
        // iconProps={{name: 'menu'}}          // 3 horizontal lines
        iconProps={{name: 'zoom-in'}}
      />
    </View>
  );

  BottomControlsButton = (props: {
    help: string,
    iconProps: IconProps,
    onPress?: (pointerInside: boolean) => void,
    onLongPress?: () => Promise<void>,
    active?: boolean,
    disabled?: boolean,
  }) => {
    const {style: iconStyle, ...iconProps} = props.iconProps;
    return (
      <LongPressGestureHandler
        onHandlerStateChange={async event => (
          props.onLongPress
            ? event.nativeEvent.state === Gesture.State.ACTIVE && await props.onLongPress()
            : await this.onBottomControlsLongPress(event)
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

  sectionsForRecs = (recs: Array<Rec>): Array<SectionListData<Rec>> => {
    const sections = [];
    let section;
    for (let rec of recs) {
      const title = rec.species;
      if (!section || title !== section.title) {
        if (section) sections.push(section);
        section = {
          title,
          data: [] as Rec[],
          species: rec.species,
          species_taxon_order: rec.species_taxon_order,
          species_com_name: rec.species_com_name,
          species_sci_name: rec.species_sci_name,
          recs_for_sp: rec.recs_for_sp,
        };
      }
      section.data.push(rec);
    }
    if (section) sections.push(section);
    return sections;
  }

  RecEditingButtons = (props: {rec: Rec}) => (
    <View style={styles.recEditingView}>
      {this._recEditingButtons.map((f, i) => f(i, props.rec))}
    </View>
  );

  get _recEditingButtons() {
    return [

      // (i: number, rec: Rec) => (
      //   <this.RecEditingButton
      //     key={i}
      //     iconName='move'
      //     onPress={() => {}}
      //   />
      // ),

      (i: number, rec: Rec) => (
        <this.RecEditingButton
          key={i}
          iconName='search'
          // onPress={() => navigate.search(this.nav, {recId: rec.id})}
          onPress={() => setStateAsync(this, {
            showGenericModal: () => (
              <this.GenericModal title='Search for' actions={[
                {
                  label: rec.species,
                  iconName: 'search',
                  buttonColor: iOSColors.blue,
                  onPress: async () => { navigate.search(this.nav, {species: rec.species}); },
                }, {
                  label: shortRecId(rec.id),
                  iconName: 'search',
                  buttonColor: iOSColors.blue,
                  onPress: async () => { navigate.search(this.nav, {recId: rec.id}); },
                },
              ]} />
            )
          })}
        />
      ),

      (i: number, rec: Rec) => (
        <this.RecEditingButton
          key={i}
          iconName='x'
          // iconName='slash'
          onPress={() => setStateAsync(this, {
            showGenericModal: () => (
              <this.GenericModal title='Remove results' actions={[
                {
                  label: rec.species,
                  iconName: 'x',
                  buttonColor: iOSColors.red,
                  onPress: async () => await setStateAsync(this, (state: State, props: Props) => ({
                    excludeSpecies: [...state.excludeSpecies, rec.species],
                  })),
                }, {
                  label: shortRecId(rec.id),
                  iconName: 'x',
                  buttonColor: iOSColors.red,
                  onPress: async () => await setStateAsync(this, (state: State, props: Props) => ({
                    excludeRecIds: [...state.excludeRecIds, rec.id],
                  })),
                },
              ]} />
            )
          })}
        />
      ),

      (i: number, rec: Rec) => (
        <this.RecEditingButton
          key={i}
          // iconName='star'
          iconName='bookmark'
          onPress={() => setStateAsync(this, {
            showGenericModal: () => (
              <this.GenericModal title='Save results' actions={[
                {
                  label: rec.species,
                  iconName: 'bookmark',
                  buttonColor: iOSColors.purple,
                  onPress: async () => {},
                }, {
                  label: shortRecId(rec.id),
                  iconName: 'bookmark',
                  buttonColor: iOSColors.purple,
                  onPress: async () => {},
                },
              ]} />
            )
          })}
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
          this.state.spectroScale >= 2 ? {} : {
            fontSize: this.state.spectroScale / 2 * material.headlineObject.fontSize!,
            lineHeight: this.state.spectroScale / 2 * material.headlineObject.lineHeight!,
          },
        ]}
        name={props.iconName}
      />
    </BorderlessButton>
  );

  GenericModal = (props: {
    title: string,
    actions: Array<{
      label: string,
      iconName?: string,
      buttonColor?: string,
      textColor?: string,
      marginVertical?: number,
      dismiss?: boolean,
      onPress: () => Promise<void>,
    }>,
  }) => (
    // Background overlay: semi-transparent background + tap outside modal to dismiss
    <BaseButton
      onPress={() => setStateAsync(this, {
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
        <Text style={{
          alignSelf: 'center', // (horizontal)
          marginBottom: 5,
          ...material.titleObject,
        }}>
          {props.title}
        </Text>
        {props.actions.map(({
          label,
          iconName,
          buttonColor,
          textColor,
          marginVertical,
          dismiss,
          onPress,
        }, i) => (
          <RectButton
            key={i}
            style={{
              flexDirection: 'row',
              alignItems: 'center',
              padding: 10, marginHorizontal: 10,
              marginVertical: marginVertical !== undefined ? marginVertical : 10,
              backgroundColor: buttonColor || iOSColors.customGray,
            }}
            onPress={async () => {
              await Promise.all([
                ...(!(dismiss !== undefined ? dismiss : true) ? [] : [
                  setStateAsync(this, {
                    showGenericModal: null, // Dismiss modal
                  }),
                ]),
                onPress(),
              ]);
            }}
          >
            {iconName && (
              <Feather
                style={{
                  ...material.headlineObject,
                  marginRight: 5,
                  color: textColor || iOSColors.white,
                }}
                name={iconName}
              />
            )}
            <Text
              style={{
                ...material.buttonObject,
                color: textColor || iOSColors.white,
              }}
              children={label}
            />
          </RectButton>
        ))}
      </View>
    </BaseButton>
  );

  RecText = <X extends {children: any, flex?: number}>(props: X) => {
    const flex = props.flex || 1;
    return (<Text
      style={[styles.recText, {flex}]}
      numberOfLines={1}
      ellipsizeMode='tail'
      {...props}
    />);
  }

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
      <Modal
        visible={this.state.showFilters}
        animationType='none' // 'none' | 'slide' | 'fade'
        transparent={true}
        children={this.Filters()}
      />
      <ActionSheetBasic
        innerRef={this.saveActionSheet}
        options={[
          ['Save as new list',      () => {}],
          ['Add all to saved list', () => {}],
          ['Share list',            () => {}],
        ]}
      />
      <ActionSheetBasic
        innerRef={this.addActionSheet}
        options={[
          ['+ num species',          () => {}],
          ['– num species',          () => {}],
          ['+ num recs per species', () => {}],
          ['– num recs per species', () => {}],
          ['Add a species manually', () => {}],
        ]}
      />
      <ActionSheetBasic
        innerRef={this.sortActionSheet}
        options={
          // this.state.queryRec ? [ // TODO queryRec
          true ? [
            ['Sort by species, then by recs', () => {}],
            ['Sort by recs only',             () => {}],
            ['Order manually',                () => {}],
          ] : [
            ['Sort recs by similarity',       () => {}],
            ['Order manually',                () => {}],
          ]
        }
      />
    </View>
  );

  // Map props/state to a DOM node
  //  - Render phase (pure, no read/write DOM, may be called multiple times per commit or interrupted)
  render = () => {
    const styleForSpecies = this.stylesForSpecies(_.uniq(this.state.recs.map(rec => rec.species)));
    return (
      <View style={styles.container}>

        {/* Loading spinner */}
        {this.state.status.loading && (
          <View style={styles.loadingView}>
            <ActivityIndicator size='large' />
          </View>
        )}

        {/* Recs list (with pan/pinch) */}
        {/* - We use ScrollView instead of SectionList to avoid _lots_ of opaque pinch-to-zoom bugs */}
        {/* - We use ScrollView instead of manual gestures (react-native-gesture-handler) to avoid _lots_ of opaque animation bugs */}
        {this.state.status.loading || (
          <ScrollView
            // @ts-ignore [Why doesn't this typecheck?]
            ref={this.scrollViewRef as RefObject<Component<SectionListStatic<Rec>, any, any>>}
            style={styles.recList}

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
              // log.debug('render.contentOffset', json(x)); // XXX Debug
            })}
            bounces={false}
            bouncesZoom={false}
            directionalLockEnabled={true} // Don't scroll vertical and horizontal at the same time (ios only)
            minimumZoomScale={this.props.spectroScaleClamp.min / this.state.spectroScale}
            maximumZoomScale={this.props.spectroScaleClamp.max / this.state.spectroScale}
            onScrollEndDrag={async ({nativeEvent}) => {
              // log.debug('onScrollEndDrag', json(nativeEvent)); // XXX Debug
              const {contentOffset, zoomScale, velocity} = nativeEvent;
              this._scrollViewState = {contentOffset};
              if (
                zoomScale !== 1              // Don't trigger zoom if no zooming happened (e.g. only scrolling)
                // && velocity !== undefined // [XXX Unreliable] Don't trigger zoom on 1/2 fingers released, wait for 2/2
              ) {
                const scale = zoomScale * this.state.spectroScale;
                // log.debug('ZOOM', json(nativeEvent)); // XXX Debug
                // Trigger re-layout so non-image components (e.g. text) redraw at non-zoomed size
                await setStateAsync(this, {
                  scrollViewState: this._scrollViewState,
                  spectroScale: this.clampSpectroScaleY(scale),
                  scrollViewKey: chance.hash(), // Else bad things (that I don't understand)
                });
              }
            }}

            // Mimic a SectionList
            children={
              _.flatten(this.sectionsForRecs(this.state.recs).map(({
                title,
                data: recs,
                species,
                species_taxon_order,
                species_com_name,
                species_sci_name,
                recs_for_sp,
              }) => [

                // Species header
                this.settings.showMetadata === 'full' && (
                  <View
                    key={`section-${title}`}
                    style={styles.sectionSpecies}
                  >
                    {/* Species name */}
                    <Text numberOfLines={1} style={styles.sectionSpeciesText}>
                      {species_com_name}
                    </Text>
                    {/* Debug info */}
                    {this.settings.showDebug && (
                      // FIXME Off screen unless zoom=1
                      <this.DebugText numberOfLines={1} style={[{marginLeft: 'auto', alignSelf: 'center'}]}>
                        ({recs_for_sp} recs)
                      </this.DebugText>
                    )}
                  </View>
                ),

                // Rec rows
                ...recs.map(rec => [

                  // Rec row
                  <Animated.View
                    key={`row-${rec.id.toString()}`}
                    style={[styles.recRow, {
                      ...(this.settings.showMetadata === 'full' ? {} : {
                        height: this.spectroDim.height, // Compact controls/labels when zoom makes image smaller than controls/labels
                      }),
                    }]}
                  >

                    {/* Rec editing buttons */}
                    {/* - NOTE Condition duplicated in scrollViewContentWidths */}
                    {this.settings.editing && (
                      <this.RecEditingButtons rec={rec} />
                    )}

                    {/* Rec region without the editing buttons  */}
                    <Animated.View style={styles.recRowInner}>

                      {/* Rec row */}
                      <View
                        style={{
                          flexDirection: 'row',
                          ...(this.settings.showMetadata === 'full' ? {} : {
                            height: this.spectroDim.height, // Compact controls/labels when zoom makes image smaller than controls/labels
                          }),
                        }}
                      >

                        {/* Rec debug info */}
                        {this.settings.showMetadata === 'inline' && (
                          <this.DebugView style={{
                            padding: 0, // Reset padding:3 from settings.debugView
                            width: this.scrollViewContentWidths.debugInfo,
                          }}>
                            {/* TODO(nav_rec_id) */}
                            {/* <this.RecText style={this.recDebugText} children={rec.xc_id} /> */}
                            <this.RecText style={this.recDebugText}>slp: {rec.slp && round(rec.slp, 2)}</this.RecText>
                            <this.RecText style={this.recDebugText}>d_pc: {rec.d_pc && round(rec.d_pc, 2)}</this.RecText>
                          </this.DebugView>
                        )}

                        {/* Rec inline metadata */}
                        {this.settings.showMetadata === 'inline' && (
                          <View style={[styles.recMetadataInlineLeft, {
                            width: this.scrollViewContentWidths.inlineMetadata,
                          }]}>
                            {this.settings.inlineMetadataColumns.map(c => (
                              <this.RecText key={c} children={InlineMetadataColumns[c](rec)} />
                            ))}
                          </View>
                        )}

                        {/* Sideways species label */}
                        {/* - After controls/metadata so that label+spectro always abut (e.g. if scrolled all the way to the right) */}
                        {/* - NOTE Keep outside of TapGestureHandler else spectroTimeFromX/spectroXFromTime have to adjust */}
                        {this.settings.showMetadata !== 'full' && (
                          <View style={[styles.recSpeciesSidewaysView, {
                            backgroundColor: styleForSpecies.get(rec.species)!.backgroundColor,
                          }]}>
                            <View style={styles.recSpeciesSidewaysViewInner}>
                              <Text numberOfLines={1} style={[styles.recSpeciesSidewaysText, {
                                fontSize: this.state.spectroScale >= 2 ? 11 : 6, // Compact species label to fit within tiny rows
                                color: styleForSpecies.get(rec.species)!.color,
                              }]}>
                                {rec.species}
                              </Text>
                            </View>
                          </View>
                        )}

                        {/* Spectro (tap) */}
                        <LongPressGestureHandler onHandlerStateChange={this.onLongPress(rec)}>
                          <TapGestureHandler onHandlerStateChange={this.toggleRecPlaying(rec)}>
                            <Animated.View>

                              {/* Image */}
                              <Animated.Image
                                style={this.spectroDim}
                                resizeMode='stretch'
                                source={{uri: Rec.spectroPath(rec)}}
                              />

                              {/* Start time cursor (if playing + startTime) */}
                              {this.recIsPlaying(rec.id, this.state.playing) && (
                                this.state.playing!.startTime && (
                                  <View style={{
                                    position: 'absolute', width: 1, height: '100%',
                                    left: this.spectroXFromTime(this.state.playing!.sound, this.state.playing!.startTime!),
                                    backgroundColor: iOSColors.gray,
                                  }}/>
                                )
                              )}

                              {/* Progress time cursor (if playing + playingCurrentTime) */}
                              {this.recIsPlaying(rec.id, this.state.playing) && (
                                this.state.playing!.startTime && this.state.playingCurrentTime !== undefined && (
                                  <View style={{
                                    position: 'absolute', width: 1, height: '100%',
                                    left: this.spectroXFromTime(this.state.playing!.sound, this.state.playingCurrentTime),
                                    backgroundColor: iOSColors.black,
                                  }}/>
                                )
                              )}

                              {/* Visual feedback for playing rec [XXX after adding progress bar by default] */}
                              {this.recIsPlaying(rec.id, this.state.playing) && (
                                <View style={{
                                  position: 'absolute', width: 2, height: '100%',
                                  left: 0,
                                  backgroundColor: iOSColors.red,
                                }}/>
                              )}

                            </Animated.View>
                          </TapGestureHandler>
                        </LongPressGestureHandler>

                      </View>

                      {/* Rec metadata */}
                      {/* {this.settings.showMetadata === 'inline' && (
                        <View style={styles.recMetadataInlineBelow}>
                          <this.RecText flex={3} children={rec.xc_id} />
                          <this.RecText flex={1} children={rec.quality} />
                          <this.RecText flex={2} children={rec.month_day} />
                          <this.RecText flex={4} children={Rec.placeNorm(rec.place)} />
                          {ccIcon({style: styles.recTextFont})}
                          <this.RecText flex={4} children={` ${rec.recordist}`} />
                        </View>
                      )} */}
                      {this.settings.showMetadata === 'full' && (
                        <View style={styles.recMetadataFull}>
                          <this.RecText flex={3} children={rec.xc_id} />
                          <this.RecText flex={1} children={rec.quality} />
                          <this.RecText flex={2} children={rec.month_day} />
                          <this.RecText flex={4} children={Rec.placeNorm(rec.place)} />
                          {ccIcon({style: styles.recTextFont})}
                          <this.RecText flex={4} children={` ${rec.recordist}`} />
                        </View>
                      )}

                    </Animated.View>

                  </Animated.View>

                ]),

              ]))
            }

          />
        )}

        {/* Debug info */}
        <this.DebugView>
          <this.DebugText>queryDesc: {this.queryDesc}</this.DebugText>
          <this.DebugText>
            Recs: {this.state.status.loading ? '...' : this.state.recs.length}/{this.state.totalRecs || '?'}
          </this.DebugText>
          <this.DebugText>Filters: {json(this.filters)}</this.DebugText>
          {/* <this.DebugText>NavParams: {json(this.navParamsAll)}</this.DebugText> */}
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
  DebugView = (props: RN.ViewProps & {children: any}) => {
    props = {...props, style: [this.settings.debugView, ...sanitizeStyle(props.style)]};
    return (<View {...props} />);
  };
  DebugText = (props: RN.TextProps & {children: any}) => {
    props = {...props, style: [this.settings.debugText, ...sanitizeStyle(props.style)]};
    return (<Text {...props} />);
  };
  get recDebugText() { return [styles.recText, this.settings.debugText]; }

}

// (Not sure about this type)
function sanitizeStyle<X extends {}>(style: undefined | null | X | Array<X>): Array<X> {
  return (
    !style ? [] :
    style instanceof Array ? style :
    [style]
  );
}

function ccIcon(props?: object): Element {
  const [icon] = licenseTypeIcons('cc', props);
  return icon;
}

function licenseTypeIcons(license_type: string, props?: object): Array<Element> {
  license_type = `cc-${license_type}`;
  return license_type.split('-').map(k => (<FontAwesome5
    key={k}
    name={k === 'cc' ? 'creative-commons' : `creative-commons-${k}`}
    {...props}
  />));
}

// TODO Why is this slow to respond after keyboard shows? -- adding logging to find the bottleneck
interface KeyboardDismissingViewState {
  isKeyboardShown: boolean;
}
export class KeyboardDismissingView extends Component<RN.ViewProps, KeyboardDismissingViewState> {
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
  keyboardDidShow = async () => {
    await setStateAsync(this, {isKeyboardShown: true});
  };
  keyboardDidHide = async () => {
    await setStateAsync(this, {isKeyboardShown: false});
  };
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
  container: {
    flex: 1,
  },
  loadingView: {
    flex: 1,
    justifyContent: 'center',
  },
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
    alignItems: 'center',
    paddingVertical: 5,
    backgroundColor: iOSColors.midGray,
  },
  bottomControlsButton: {
    flex: 1,
    alignItems: 'center',
  },
  bottomControlsButtonHelp: {
    ...material.captionObject,
  },
  bottomControlsButtonIcon: {
    ...material.headlineObject,
  },
  queryInput: {
    borderWidth: 1, borderColor: 'gray',
    ...material.display1Object,
  },
  summaryText: {
    ...material.captionObject,
  },
  recList: {
    // borderWidth: 1, borderColor: 'gray',
  },
  sectionSpecies: {
    flexDirection: 'row',
    // ...material.body1Object, backgroundColor: iOSColors.customGray, // Black on white
    ...material.body1WhiteObject, backgroundColor: iOSColors.gray, // White on black
  },
  sectionSpeciesText: {
    alignSelf: 'center', // Align text vertically
  },
  recRow: {
    flex: 1, flexDirection: 'row',
  },
  recRowInner: {
    flex: 1, flexDirection: 'column',
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
  recMetadataInlineLeft: {
    flexDirection: 'column',
  },
  recMetadataInlineBelow: {
    flex: 2, flexDirection: 'row', // TODO Eh...
  },
  recMetadataFull: {
    flex: 1,
    flexDirection: 'column',
    height: 100, // TODO(fix_rec_metadata)
  },
  recText: {
    ...material.captionObject,
  },
  recTextFont: {
    ...material.captionObject,
  },
  recSpectro: {
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
  swipeButtons: {
    flexDirection: 'row',
  },
  swipeButton: {
    flex: 1,
    alignItems: 'center',
  },
  swipeButtonText: {
  },
});
