import _ from 'lodash';
import { createMemoryHistory, Location, MemoryHistory } from 'history';
import React, { Component, ComponentType, PureComponent, ReactNode, RefObject } from 'React';
import {
  ActivityIndicator, Animated, AsyncStorage, Dimensions, Linking, Platform, SafeAreaView, StatusBar, Text, View,
} from 'react-native';
import KeepAwake from 'react-native-keep-awake';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import { BackButton, Link, matchPath, Redirect, Route, Switch } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { DeepLinking } from './components/DeepLinking';
import { RecentScreen } from './components/RecentScreen';
import { RecordScreen } from './components/RecordScreen';
import { SavedScreen } from './components/SavedScreen';
import { SearchScreen } from './components/SearchScreen';
import { SettingsScreen } from './components/SettingsScreen';
import { TabRoutes, TabLink } from './components/TabRoutes';
import { config } from './config';
import { Models, ModelsSearch, SearchRecs, ServerConfig } from './datatypes';
import { log } from './log';
import { getOrientation, matchOrientation, Orientation } from './orientation';
import {
  createDefaultHistories, Go, Histories, HistoryConsumer, loadHistories, ObserveHistory, RouterWithHistory,
  saveHistories, TabHistories, TabName,
} from './router';
import { Settings, SettingsProxy, SettingsWrites } from './settings';
import { StyleSheet } from './stylesheet';
import { urlpack } from './urlpack';
import {
  deepEqual, global, json, match, Omit, pretty, readJsonFile, shallowDiff, shallowDiffPropsState, Style,
} from './utils';

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.Animated = Animated;
global.AsyncStorage = AsyncStorage;
global.deepEqual = deepEqual;
global.Dimensions = Dimensions;
global.Linking = Linking;
global.Platform = Platform;
global.iOSColors = iOSColors;
global.matchPath = matchPath;
global.material = material;
global.materialColors = materialColors;
global.shallowDiff = shallowDiff;
global.Settings = Settings;
global.systemWeights = systemWeights;
global.urlpack = urlpack;
const timed = (desc: string, f: () => void) => { log.time(desc); f(); log.timeEnd(desc); };
global.sj = {};
timed('AudioUtils',         () => global.AudioUtils      = require('../third-party/magenta/music/transcription/audio_utils'));
// timed('d3',              () => global.d3              = require('d3'));                      // 50ms [heavy, don't need full d3]
timed('d3-color',           () => global.d3c             = require('d3-color'));                // 2ms
timed('d3-scale-chromatic', () => global.d3sc            = require('d3-scale-chromatic'));      // 6ms
timed('jimp',               () => global.Jimp            = require('jimp'));                    // 170ms
timed('lodash',             () => global._               = require('lodash'));                  // 0ms
// timed('path-to-regexp',     () => global.pathToRegexp    = require('path-to-regexp'));          // ?ms
timed('ndarray',            () => global.ndarray         = require('ndarray'));                 // 1ms
timed('nj',                 () => global.nj              = require('../third-party/numjs/dist/numjs.min')); // 130ms
timed('sj.ops',             () => global.sj.ops          = require('ndarray-ops'));             // 50ms
timed('React',              () => global.R               = require('React'));                   // 0ms
timed('ReactNative',        () => global.RN              = require('ReactNative'));             // 13ms
timed('rn-fetch-blob',      () => global.RNFB            = require('rn-fetch-blob').default);   // 0ms
timed('Gesture',            () => global.Gesture         = require('react-native-gesture-handler')); // ?
timed('react-native-sound', () => global.Sound           = require('react-native-sound'));      // 1ms
timed('SQLite',             () => global.SQLite          = require('react-native-sqlite-storage')); // 0ms
timed('typography',         () => global.typography      = require('react-native-typography')); // 27ms
timed('sj.zeros',           () => global.sj.zeros        = require('zeros'));                   // 0ms
// timed('sj.getPixels',    () => global.sj.getPixels    = require('get-pixels'));              // 10ms // XXX Doesn't work in RN
// timed('sj.savePixels',   () => global.sj.savePixels   = require('save-pixels'));             // 30ms // XXX Doesn't work in RN
global.fs = global.RNFB.fs;

interface Props {}

interface State {
  tabIndex: number;
  orientation: 'portrait' | 'landscape';
  loading: boolean;
  // Loaded async (e.g. from storage)
  histories?: Histories;
  serverConfig?: ServerConfig;
  modelsSearch?: ModelsSearch;
  settings?: Settings;
  settingsWrites?: SettingsWrites;
  appContext?: AppContext;
}

interface AppContext {
  // TODO Use context for anything?
}

const AppContext = React.createContext(
  // HACK No default: can't provide a real AppContext here, and we can ensure we always provide one [until tests]
  undefined as unknown as AppContext,
);

export default class App extends PureComponent<Props, State> {

  constructor(props: Props) {
    super(props);
    this.state = {
      tabIndex: 2, // TODO
      orientation: getOrientation(),
      loading: true,
    };
    global.App = this; // XXX Debug
  }

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);

    // Load/create histories
    //  - Save histories on change
    const histories = await loadHistories() || createDefaultHistories();
    Object.values(histories).forEach(history => {
      history.listen(() => saveHistories(histories));
    });

    // Load serverConfig (async) on app startup
    const serverConfig = await readJsonFile<ServerConfig>(`${fs.dirs.MainBundleDir}/${SearchRecs.serverConfigPath}`);

    // Load modelsSearch (async) on app startup
    const modelsSearch = await readJsonFile<ModelsSearch>(`${fs.dirs.MainBundleDir}/${Models.search.path}`);

    // Load settings (async) on app startup
    const settings = await Settings.load(
      settings => this.setState({settings}), // Callback for when Settings updates
    );

    // A prop to pass to children components that won't change when Settings props change
    //  - We pass Settings props separately, for more fine-grained control over when to trigger updates
    //  - this.state.settings will change on each prop update, but this.state.settingsWrites won't
    const settingsWrites: SettingsWrites = new SettingsProxy(() => {
      return this.state.settings!
    });

    const appContext = {
    };

    // TODO Show loading screen until loads complete
    this.setState({
      loading: false,
      histories,
      serverConfig,
      modelsSearch,
      settings,
      settingsWrites,
      appContext,
    });

  }

  componentDidUpdate = (prevProps: Props, prevState: State) => {
    log.info('App.componentDidUpdate', shallowDiffPropsState(prevProps, prevState, this.props, this.state));
    global.histories = this.state.histories; // XXX Debug
    global.settings = this.state.settings; // XXX Debug
  }

  render = () => (

    // Avoid rounded corners and camera notches (ios ≥11)
    //  - https://facebook.github.io/react-native/docs/safeareaview
    <SafeAreaView style={{
      flex: 1,
      backgroundColor: '#ffffff',
    }}>

      {/* For onLayout -> orientation */}
      <View onLayout={this.onLayout}>

        {/* Show loading spinner */}
        {this.state.loading ? (
          <View style={styles.loadingView}>
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
                  onUrl={({path}) => {
                    const match = matchPath(path, '/:tab/:tabPath*');
                    if (match) {
                      const {tab, tabPath} = match.params as {tab: TabName, tabPath: string};
                      this.go(tab, '/' + (tabPath || '')); // (Leading '/' for absolute i/o relative)
                    }
                  }}
                />

                {/* Tabs + screen pager */}
                {/* - NOTE Avoid history.location [https://reacttraining.com/react-router/native/api/history/history-is-mutable] */}
                <HistoryConsumer children={({location: locationTabs, history: historyTabs}) => (
                  <TabRoutes
                    defaultPath='/search'
                    histories={this.state.histories!}
                    routes={[
                      {
                        key: 'record', route: {path: '/record'},
                        label: 'Record', iconName: 'activity',
                        render: props => (
                          <RecordScreen {...props} />
                        ),
                      }, {
                        key: 'search', route: {path: '/search'},
                        label: 'Search', iconName: 'search',
                        render: props => (
                          <SearchScreen {...props}
                            // App globals
                            serverConfig            = {this.state.serverConfig!}
                            modelsSearch            = {this.state.modelsSearch!}
                            go                      = {this.go}
                            // Settings
                            settings                = {this.state.settingsWrites!}
                            showDebug               = {this.state.settings!.showDebug}
                            showMetadata            = {this.state.settings!.showMetadata}
                            inlineMetadataColumns   = {this.state.settings!.inlineMetadataColumns}
                            editing                 = {this.state.settings!.editing}
                            seekOnPlay              = {this.state.settings!.seekOnPlay}
                            playingProgressEnable   = {this.state.settings!.playingProgressEnable}
                            playingProgressInterval = {this.state.settings!.playingProgressInterval}
                            spectroScale            = {this.state.settings!.spectroScale}
                          />
                        ),
                      }, {
                        key: 'recent', route: {path: '/recent'},
                        label: 'Recent', iconName: 'list',
                        render: props => (
                          <RecentScreen {...props}
                            // App globals
                            go        = {this.go}
                            // Settings
                            showDebug = {this.state.settings!.showDebug}
                          />
                        ),
                      }, {
                        key: 'saved', route: {path: '/saved'},
                        label: 'Saved', iconName: 'bookmark',
                        render: props => (
                          <SavedScreen {...props} />
                        ),
                      }, {
                        key: 'settings', route: {path: '/settings'},
                        label: 'Settings', iconName: 'settings',
                        render: props => (
                          <SettingsScreen {...props}
                            // Settings
                            settings                = {this.state.settingsWrites!}
                            showDebug               = {this.state.settings!.showDebug}
                            allowUploads            = {this.state.settings!.allowUploads}
                            playingProgressEnable   = {this.state.settings!.playingProgressEnable}
                            playingProgressInterval = {this.state.settings!.playingProgressInterval}
                          />
                        ),
                      },
                    ]}
                  />
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

  onLayout = () => {
    // log.info('App.onLayout');
    this.setState({
      orientation: getOrientation(),
    });
  }

  go = (tab: TabName, path: string) => {
    log.info('App.go', json({tab, path}));
    if (tab) {
      // Show tab
      this.state.histories!.tabs.replace('/' + tab); // (Leading '/' for absolute i/o relative)
      if (path) {
        // Push new location, unless it's the current location
        const history = this.state.histories![tab];
        if (path !== history.location.pathname) {
          history.push(path);
        }
      }
    }
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
  loadingView: {
    flex: 1,
    justifyContent: 'center',
  },
});
