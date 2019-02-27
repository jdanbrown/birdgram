import cheerio from 'cheerio-without-node-native';
import _ from 'lodash';
import React, { Component, ComponentType, PureComponent, ReactNode, RefObject } from 'React';
import {
  ActivityIndicator, Alert, Animated, AsyncStorage, Dimensions, Linking, NativeModules, Platform, SafeAreaView,
  StatusBar, Text, View,
} from 'react-native';
import DeviceInfo from 'react-native-device-info';
import KeepAwake from 'react-native-keep-awake';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import { BackButton, Link, matchPath, Redirect, Route, Switch } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { DeepLinking } from 'app/components/DeepLinking';
import { HelpScreen } from 'app/components/HelpScreen';
import { Geo, geolocation } from 'app/components/Geo';
import { PlacesScreen } from 'app/components/PlacesScreen';
import { RecentScreen } from 'app/components/RecentScreen';
import { RecordScreen } from 'app/components/RecordScreen';
import { SavedScreen } from 'app/components/SavedScreen';
import { SearchScreen } from 'app/components/SearchScreen';
import { SettingsScreen } from 'app/components/SettingsScreen';
import { TabRoute, TabRouteKey, TabRoutes, TabLink } from 'app/components/TabRoutes';
import * as Colors from 'app/colors';
import { config } from 'app/config';
import {
  MetadataSpecies, Models, ModelsSearch, Rec, SearchRecs, ServerConfig, UserRec, XCRec,
} from 'app/datatypes';
import { DB } from 'app/db';
import { Ebird } from 'app/ebird';
import { debug_print, Log, puts, rich } from 'app/log';
import { NativeHttp } from 'app/native/Http';
import { NativeSearch } from 'app/native/Search';
import { NativeSpectro } from 'app/native/Spectro';
import { NativeTagLib } from 'app/native/TagLib';
import { getOrientation, matchOrientation, Orientation } from 'app/orientation';
import {
  createDefaultHistories, getTabLocations, Go, GoTo, Histories, HistoryConsumer, loadHistories, locationPathIsEqual,
  ObserveHistory, RouterWithHistory, saveHistories, TabHistories, TabLocations, TabName,
} from 'app/router';
import { Settings, SettingsProxy, SettingsWrites } from 'app/settings';
import { querySql } from 'app/sql';
import { StyleSheet } from 'app/stylesheet';
import { urlpack } from 'app/urlpack';
import {
  assert, dirname, fastIsEqual, global, Interval, json, local, mapNull, match, matchNil, Omit, pretty, qsSane,
  readJsonFile, shallowDiff, shallowDiffPropsState, Style, Timer, yaml,
} from 'app/utils';
import { XC } from 'app/xc';

const log = new Log('App');

log.info('config', pretty(config));

// // XXX Debug: log bridge msgs
// //  - https://blog.callstack.io/reactnative-how-to-check-what-passes-through-your-bridge-e435571ffd85
// //  - https://github.com/jondot/rn-snoopy
// // @ts-ignore (no d.ts file)
// import MessageQueue from 'react-native/Libraries/BatchedBridge/MessageQueue.js';
// MessageQueue.spy((x: any) => {
//   const msg       = `${json(x)}`;
//   const taggedMsg = `[bridge] ${msg}`;
//   log.debug('', taggedMsg);
//   if (!msg.includes('[bridge]')) Spectro.debugPrintNative(taggedMsg); // Avoid infinite recursion
// });

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
Object.assign(global,
  require('./datatypes'),
  require('./db'),
  require('./router'),
  require('./sql'),
  require('./utils'),
);
global.Alert = Alert;
global.Animated = Animated;
global.AsyncStorage = AsyncStorage;
global.cheerio = cheerio;
global.Colors = Colors;
global.config = config;
global.DeviceInfo = DeviceInfo;
global.Dimensions = Dimensions;
global.Ebird = Ebird;
global.fastIsEqual = fastIsEqual;
global.Geo = Geo;
global.geolocation = geolocation;
global.Linking = Linking;
global.Platform = Platform;
global.iOSColors = iOSColors;
global.Interval = Interval;
global.matchPath = matchPath;
global.material = material;
global.materialColors = materialColors;
global.NativeModules = NativeModules;
global.NativeHttp = NativeHttp;
global.NativeSearch = NativeSearch;
global.NativeSpectro = NativeSpectro;
global.NativeTagLib = NativeTagLib;
global.querySql = querySql;
global.qsSane = qsSane;
global.Rec = Rec;
global.shallowDiff = shallowDiff;
global.Settings = Settings;
global.systemWeights = systemWeights;
global.Timer = Timer;
global.urlpack = urlpack;
global.UserRec = UserRec;
global.XC = XC;
global.XCRec = XCRec;
const timed = (desc: string, f: () => void) => { log.time(desc); f(); log.timeEnd(desc); };
global.sj = {};
timed('AudioUtils',         () => global.AudioUtils      = require('../third-party/magenta/music/transcription/audio_utils'));
timed('AudioRecord',        () => global.AudioRecord     = require('react-native-audio-record').default); // ?
timed('base64-js',          () => global.base64js        = require('base64-js'));               // ?
// timed('d3',              () => global.d3              = require('d3'));                      // 50ms [heavy, don't need full d3]
timed('d3-color',           () => global.d3c             = require('d3-color'));                // 2ms
timed('d3-scale-chromatic', () => global.d3sc            = require('d3-scale-chromatic'));      // 6ms
// timed('jimp',               () => global.Jimp            = require('jimp'));                    // 170ms // XXX Unused
timed('lodash',             () => global._               = require('lodash'));                  // 0ms
timed('path-to-regexp',     () => global.pathToRegexp    = require('path-to-regexp'));          // ?
timed('ndarray',            () => global.ndarray         = require('ndarray'));                 // 1ms
timed('nj',                 () => global.nj              = require('../third-party/numjs/dist/numjs.min')); // 130ms
timed('sj.ops',             () => global.sj.ops          = require('ndarray-ops'));             // 50ms
timed('query-string',       () => global.queryString     = require('query-string'));            // ?
timed('React',              () => global.R               = require('React'));                   // 0ms
timed('ReactNative',        () => global.RN              = require('ReactNative'));             // 13ms
timed('rn-fetch-blob',      () => global.RNFB            = require('rn-fetch-blob').default);   // 0ms
timed('Gesture',            () => global.Gesture         = require('react-native-gesture-handler')); // ?
timed('./sound',            () => global.Sound           = require('./sound').default);         // 1ms
timed('SQLite',             () => global.SQLite          = require('react-native-sqlite-storage')); // 0ms
timed('traverse',           () => global.traverse        = require('traverse'));                // ?
timed('typography',         () => global.typography      = require('react-native-typography')); // 27ms
timed('wavefile',           () => global.WaveFile        = require('wavefile/dist/wavefile'));  // ?
timed('sj.zeros',           () => global.sj.zeros        = require('zeros'));                   // 0ms
// timed('sj.getPixels',    () => global.sj.getPixels    = require('get-pixels'));              // 10ms // XXX Doesn't work in RN
// timed('sj.savePixels',   () => global.sj.savePixels   = require('save-pixels'));             // 30ms // XXX Doesn't work in RN
timed('url-parse',          () => global.urlParse        = require('url-parse'));               // ?
global.base64 = global.RNFB.base64;
global.fs = global.RNFB.fs;

interface Props {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
}

interface State {
  tabIndex: number;
  orientation: 'portrait' | 'landscape';
  loading: boolean;
  // Loaded async (e.g. from fs, storage)
  histories?: Histories;       // FIXME Why doesn't TabRoutes trigger componentDidUpdate on sub-components when .histories changes?
  tabLocations?: TabLocations; // HACK Track separately from .histories so sub-components can get componentDidUpdate's
  serverConfig?: ServerConfig;
  metadataSpecies?: MetadataSpecies;
  xc?: XC;
  ebird?: Ebird;
  modelsSearch?: ModelsSearch;
  settings?: Settings;
  settingsWrites?: SettingsWrites;
  db?: DB;
  nSpecies?: number;
  geo?: Geo;
  appContext?: AppContext;
}

interface AppContext {
  // TODO Use context for anything?
}

const AppContext = React.createContext(
  // HACK No default: can't provide a real AppContext here, and we can ensure we always provide one [until tests]
  undefined as unknown as AppContext,
);

const iconForTab: {[key in TabName]: string} = {
  record:   'activity',
  search:   'search',
  // recent:   'list',
  recent:   'clock',
  saved:    'bookmark',
  places:   'map-pin',
  settings: 'settings',
  help:     'help-circle',
};

export default class App extends PureComponent<Props, State> {

  // Many of these are hardcoded to match Bubo/Models.swift:Features (which is in turn hardcoded to match py Features config)
  static defaultProps: Partial<Props> = {
    sampleRate:    22050,
    channels:      1,
    bitsPerSample: 16,
  };

  state: State = {
    tabIndex: 2, // TODO
    orientation: getOrientation(),
    loading: true,
  };

  componentDidMount = async () => {
    log.info('componentDidMount');
    global.App = this; // XXX Debug
    await log.timedAsync('componentDidMount [total]', async () => {

      // Back compat: migrate old-style {user-recs-v0,edits-v0}/ dirs to new-style Recordings/ dir
      //  - XXX(unify_edit_user_recs): After all (three) active users have migrated
      if (await fs.isDir(Rec.old_editDir))    await fs.mv(Rec.old_editDir, Rec.trash_editDir); // edits-v0/     -> _trash_edits-v0/
      if (await fs.isDir(Rec.old_userRecDir)) await fs.mv(Rec.old_userRecDir, Rec.userRecDir); // user-recs-v0/ -> Recordings/

      // Load/create histories
      const histories = (
        await log.timedAsync(`Load histories`, async () => {
          const histories = await loadHistories() || createDefaultHistories();
          Object.values(histories).forEach(history => {
            // On history change
            history.listen(() => {

              // Trim history.entries (else we're stuck with the default unbounded growth)
              //  - HACK MemoryHistory.setState isn't exposed, so we have to mutate the props manually
              const maxHistory = this.state.settings && this.state.settings.maxHistory;
              if (maxHistory && maxHistory > 0) {
                const {entries} = history;
                const trimmed   = entries.slice(-maxHistory); // Most recent last
                const diff      = entries.length - trimmed.length
                history.entries = trimmed;
                history.length  = history.length - diff;
                history.index   = Math.max(0, history.index - diff);
                assert(history.length >= 0 && history.index >= 0);
              }

              // Save all histories (on any history change)
              saveHistories(histories);

              // Update tabLocations (on any history change)
              this.setState((state, props) => {
                if (!state.histories) return null;
                const tabLocations = getTabLocations(state.histories);
                if (fastIsEqual(tabLocations, state.tabLocations)) return null;
                return {
                  tabLocations,
                };
              });

            });
          });
          return histories;
        })
      );

      // Set initial tabLocations
      const tabLocations = getTabLocations(histories);

      // Load settings (async) on app startup
      const settings = (
        await log.timedAsync(`Load Settings`, async () => {
          return await Settings.load(
            settings => this.setState({settings}), // Callback for when Settings updates
          );
        })
      );

      // A prop to pass to children components that won't change when Settings props change
      //  - We pass Settings props separately, for more fine-grained control over when to trigger updates
      //  - this.state.settings will change on each prop update, but this.state.settingsWrites won't
      const settingsWrites: SettingsWrites = new SettingsProxy(() => {
        return this.state.settings!
      });

      // Load db (async) on app startup
      const db = await DB.newAsync();

      // XXX Debug: log sqlite version
      // log.timedAsync('select sqlite_version()', async () => {
      //   return await db.query<{sqlite_version: string}>(`
      //     select sqlite_version() as sqlite_version
      //   `)(async results => {
      //     const [{sqlite_version}] = results.rows.raw();
      //     log.info('componentDidMount: sqlite_version', {sqlite_version});
      //   });
      // });

      // Count species (from db)
      //  - TODO Write this into a payload .json so we don't have to spend the ~100ms[?] querying it
      const nSpecies: number = await log.timedAsync('select count(distinct species)', async () => {
        return await db.query<{n: number}>(`
          select count(distinct species) as n
          from search_recs
        `)(async results => {
          const [{n}] = results.rows.raw();
          return n;
        });
      });

      // Load serverConfig (async) on app startup
      const serverConfigPath = `${fs.dirs.MainBundleDir}/${SearchRecs.serverConfigPath}`;
      const serverConfig = (
        await log.timedAsync(`Load serverConfig ${json({serverConfigPath})}`, async () => {
          return await readJsonFile<ServerConfig>(serverConfigPath);
        })
      );

      // Load metadataSpecies (async) on app startup
      const metadataSpeciesPath = `${fs.dirs.MainBundleDir}/${SearchRecs.metadataSpeciesPath}`;
      const metadataSpecies = (
        await log.timedAsync(`Load metadataSpecies ${json({metadataSpeciesPath})}`, async () => {
          return await readJsonFile<MetadataSpecies>(metadataSpeciesPath);
        })
      );

      // Load xc
      const xc = await XC.newAsync(db);

      // Load ebird (not much to it, just needs metadataSpecies)
      const ebird = new Ebird(metadataSpecies);

      // Load modelsSearch (async) on app startup
      const modelsSearchPath = `${fs.dirs.MainBundleDir}/${Models.search.path}`;
      const modelsSearch: ModelsSearch = (
        await log.timedAsync(`Load modelsSearch ${json({modelsSearchPath})}`, async () => ({
          ...(await readJsonFile<Omit<ModelsSearch, '_path'>>(modelsSearchPath)),
          _path: modelsSearchPath,
        }))
      );

      // Track geo coords (while app is running)
      //  - TODO(geo_reload): Trigger reload when settings.geoHighAccuracy changes (used to be a react component)
      const geo = new Geo({
        enableHighAccuracy: settings.geoHighAccuracy,
      });
      geo.start();

      // Load native models (async) on app startup
      //  - TODO(refactor_native_deps) Refactor so that all native singletons are created together at App init, so deps can be passed in
      //    - Search is currently created at App init [here]
      //    - Spectro is currently re-created on each startRecording(), and needs Search as a dep
      await log.timedAsync(`Load NativeSearch`, async () => {
        await NativeSearch.create(modelsSearch);
      });

      // Load native spectro module (global singleton)
      //  - After NativeSearch
      await log.timedAsync(`Load NativeSpectro`, async () => {
        await NativeSpectro.create({
          f_bins:           settings.f_bins, // NOTE Requires restart
          sampleRate:       this.props.sampleRate,
          bitsPerChannel:   this.props.bitsPerSample,
          channelsPerFrame: this.props.channels,
        });
      });

      const appContext = {
      };

      // TODO Show loading screen until loads complete
      this.setState({
        loading: false,
        histories,
        tabLocations,
        serverConfig,
        metadataSpecies,
        xc,
        ebird,
        modelsSearch,
        settings,
        settingsWrites,
        geo,
        db,
        nSpecies,
        appContext,
      });

    });
  }

  componentDidUpdate = (prevProps: Props, prevState: State) => {
    // log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state))); // Noisy in xcode
    global.histories = this.state.histories; // XXX Debug
    global.settings = this.state.settings; // XXX Debug
  }

  render = () => {
    log.info('render');
    return (

      // Avoid rounded corners and camera notches (ios ≥11)
      //  - https://facebook.github.io/react-native/docs/safeareaview
      <SafeAreaView style={{
        flex: 1,
        backgroundColor: '#ffffff',
      }}>

        {/* For onLayout -> orientation */}
        <View style={{flex: 1}} onLayout={this.onLayout}>

          {/* Show loading spinner */}
          {this.state.loading ? (
            <View style={{
              flex: 1,
              justifyContent: 'center',
            }}>
              <ActivityIndicator size='large' />
            </View>
          ) : (

            // Provide app-global stuff via context
            <AppContext.Provider value={this.state.appContext!}>

              {/* Keep screen awake (in dev) */}
              {__DEV__ && <KeepAwake/>}

              {/* Hide status bar on all screens [I tried to toggle it on/off on different screens and got weird behaviors] */}
              <StatusBar hidden />

              {/* Top-level tab router (nested stacks will have their own router) */}
              <RouterWithHistory history={this.state.histories!.tabs}>
                <View style={styles.fill}>

                  {/* Route the back button (android) */}
                  {/* - FIXME Update for multiple histories (will require moving into TabRoutes since it owns tab index state) */}
                  <BackButton/>

                  {/* Route deep links */}
                  {/* - TODO(nav_router): Do Linking.openURL('birdgram-us://open/search/species/HOWR') twice -> /search/search/HOWR */}
                  <DeepLinking
                    prefix='birdgram-us://open'
                    onUrl={this.onDeepLinkingUrl}
                  />

                  {/* Tabs + screen pager */}
                  {/* - NOTE Avoid history.location [https://reacttraining.com/react-router/native/api/history/history-is-mutable] */}
                  <HistoryConsumer children={({location: locationTabs, history: historyTabs}) => (
                    <Route children={({location}) => (
                      <TabRoutes
                        tabLocation={location}
                        histories={this.state.histories!}
                        routes={this.makeRoutes()}
                        defaultPath={this.defaultPath}
                        priorityTabs={this.priorityTabs}
                      />
                    )}/>
                  )}/>

                  {/* [Examples of] Global redirects */}
                  <Route exact path='/test-redir-fixed' render={() => (
                    <Redirect to='/settings' />
                  )}/>
                  <Route exact path='/test-redir-part/:part' render={({match: {params: {part}}}) => (
                    <Redirect to={`/${part}`} />
                  )}/>
                  <Route exact path='/test-redir-path/*' render={({match: {params: {0: path}}}) => (
                    <Redirect to={`/${path}`} />
                  )}/>

                </View>
              </RouterWithHistory>

            </AppContext.Provider>

          )}

        </View>

      </SafeAreaView>

    );
  }

  defaultPath:  string             = '/help';    // Which tab to open on first app launch
  priorityTabs: Array<TabRouteKey> = ['record']; // If one of these tabs opens at launch, lazy-load other tabs
  makeRoutes = (): Array<TabRoute> => [          // Must be a function, else screens won't update on App props/state change
    {
      key: 'record', route: {path: '/record'}, label: 'Record', iconName: iconForTab['record'],
      render: props => (
        <RecordScreen {...props}
          // App globals
          modelsSearch            = {this.state.modelsSearch!}
          go                      = {this.go}
          geo                     = {this.state.geo!}
          // Settings
          settings                = {this.state.settingsWrites!}
          db                      = {this.state.db!}
          showDebug               = {this.state.settings!.showDebug}
          refreshRate             = {this.state.settings!.refreshRate}
          doneSpectroChunkWidth   = {this.state.settings!.doneSpectroChunkWidth}
          spectroChunkLimit       = {this.state.settings!.spectroChunkLimit}
          geoWarnIfNoCoords       = {this.state.settings!.geoWarnIfNoCoords}
          // RecordScreen
          f_bins                  = {this.state.settings!.f_bins}
          sampleRate              = {this.props.sampleRate}
          channels                = {this.props.channels}
          bitsPerSample           = {this.props.bitsPerSample}
        />
      ),
    }, {
      key: 'search', route: {path: '/search'}, label: 'Search', iconName: iconForTab['search'],
      render: props => (
        <SearchScreen {...props}
          // App globals
          serverConfig            = {this.state.serverConfig!}
          modelsSearch            = {this.state.modelsSearch!}
          go                      = {this.go}
          xc                      = {this.state.xc!}
          ebird                   = {this.state.ebird!}
          // Settings
          settings                = {this.state.settingsWrites!}
          db                      = {this.state.db!}
          showDebug               = {this.state.settings!.showDebug}
          showMetadataLeft        = {this.state.settings!.showMetadataLeft}
          showMetadataBelow       = {this.state.settings!.showMetadataBelow}
          metadataColumnsLeft     = {this.state.settings!.metadataColumnsLeft}
          metadataColumnsBelow    = {this.state.settings!.metadataColumnsBelow}
          editing                 = {this.state.settings!.editing}
          seekOnPlay              = {this.state.settings!.seekOnPlay}
          playingProgressEnable   = {this.state.settings!.playingProgressEnable}
          playingProgressInterval = {this.state.settings!.playingProgressInterval}
          spectroScale            = {this.state.settings!.spectroScale}
          place                   = {this.state.settings!.place}
          places                  = {this.state.settings!.places}
          // SearchScreen
          f_bins                  = {this.state.settings!.f_bins}
        />
      ),
    }, {
      key: 'recent', route: {path: '/recent'}, label: 'Recent', iconName: iconForTab['recent'],
      render: props => (
        <RecentScreen {...props}
          // App globals
          go                      = {this.go}
          tabLocations            = {this.state.tabLocations!}
          xc                      = {this.state.xc!}
          ebird                   = {this.state.ebird!}
          // Settings
          showDebug               = {this.state.settings!.showDebug}
          maxHistory              = {this.state.settings!.maxHistory}
          // RecentScreen
          iconForTab              = {iconForTab}
        />
      ),
    }, {
      key: 'saved', route: {path: '/saved'}, label: 'Saved', iconName: iconForTab['saved'],
      render: props => (
        <SavedScreen {...props}
          // App globals
          go                      = {this.go}
          tabLocations            = {this.state.tabLocations!}
          xc                      = {this.state.xc!}
          ebird                   = {this.state.ebird!}
          // SavedScreen
          iconForTab              = {iconForTab}
        />
      ),
    }, {
      key: 'places', route: {path: '/places'}, label: 'Places', iconName: iconForTab['places'],
      render: props => (
        <PlacesScreen {...props}
          // App globals
          go                      = {this.go}
          metadataSpecies         = {this.state.metadataSpecies!}
          ebird                   = {this.state.ebird!}
          geo                     = {this.state.geo!}
          nSpecies                = {this.state.nSpecies!}
          // Settings
          settings                = {this.state.settingsWrites!}
          showDebug               = {this.state.settings!.showDebug}
          place                   = {this.state.settings!.place}
          places                  = {this.state.settings!.places}
        />
      ),
    }, {
      key: 'settings', route: {path: '/settings'}, label: 'Settings', iconName: iconForTab['settings'],
      render: props => (
        <SettingsScreen {...props}
          // Settings
          serverConfig            = {this.state.serverConfig!}
          settings                = {this.state.settingsWrites!}
          // Global
          showDebug               = {this.state.settings!.showDebug}
          allowUploads            = {this.state.settings!.allowUploads}
          geoHighAccuracy         = {this.state.settings!.geoHighAccuracy}
          geoWarnIfNoCoords       = {this.state.settings!.geoWarnIfNoCoords}
          maxHistory              = {this.state.settings!.maxHistory}
          f_bins                  = {this.state.settings!.f_bins}
          // RecordScreen
          refreshRate             = {this.state.settings!.refreshRate}
          doneSpectroChunkWidth   = {this.state.settings!.doneSpectroChunkWidth}
          spectroChunkLimit       = {this.state.settings!.spectroChunkLimit}
          // SearchScreen
          playingProgressEnable   = {this.state.settings!.playingProgressEnable}
          playingProgressInterval = {this.state.settings!.playingProgressInterval}
        />
      ),
    }, {
      key: 'help', route: {path: '/help'}, label: 'Help', iconName: iconForTab['help'],
      render: props => (
        <HelpScreen {...props}
          // App globals
          go = {this.go}
        />
      ),
    },
  ];

  onLayout = () => {
    // log.info('onLayout');
    this.setState({
      orientation: getOrientation(),
    });
  }

  onDeepLinkingUrl = ({path}: {path: string}) => {
    const match = matchPath(path, '/:tab/:tabPath*');
    if (match) {
      const {tab, tabPath} = match.params as {tab: TabName, tabPath: string};
      this.go(tab, {path: '/' + (tabPath || '')}); // (Leading '/' for absolute i/o relative)
    }
  }

  go = (tab: TabName, to: GoTo) => {
    log.info('go', {tab, to});
    // Update tab location (async, can't await)
    const history = this.state.histories![tab];
    if (to.path !== undefined) {
      // Push new location on top of most recent location (else we lose items)
      //  - HACK Mutate history.index i/o calling history.go() to avoid an unnecessary update
      //    - Ref: https://github.com/ReactTraining/history/blob/v4.7.2/modules/createMemoryHistory.js#L61
      history.index = history.length - 1;
      // Dedupe contiguous history.entries: don't push if .pathname isn't changing
      //  - NOTE .state isn't compared; it should be determined by .pathname (modulo important, e.g. timestamp isn't)
      //  - NOTE Compare using locationPathIsEqual b/c === on .pathname isn't reliable (see locationPathIsEqual)
      if (locationPathIsEqual({pathname: to.path}, history.entries[history.index])) {
        // log.debug('go: Dedupe', {to, historyLocation: history.location}); // XXX Debug
      } else {
        // log.debug('go: Push', {to, historyLocation: history.location}); // XXX Debug
        history.push(to.path, {
          timestamp: new Date(),
        });
      }
    } else if (to.index !== undefined) {
      history.go(to.index - history.index);
    }
    // Switch to tab
    this.state.histories!.tabs.replace('/' + tab); // (Leading '/' for absolute i/o relative)
  }

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  fill: {
    width: '100%',
    height: '100%',
  },
});
