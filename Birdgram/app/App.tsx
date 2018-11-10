import { Location, MemoryHistory } from 'history';
import React, { Component, ComponentType, RefObject } from 'React';
import {
  ActivityIndicator, Animated, AsyncStorage, Dimensions, Linking, Platform, SafeAreaView, StatusBar, Text, View,
} from 'react-native';
import KeepAwake from 'react-native-keep-awake';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import { BackButton, Link, NativeRouter, Redirect, Route, Switch } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { DeepLinking } from './components/DeepLinking';
import { RecentScreen } from './components/RecentScreen';
import { RecordScreen } from './components/RecordScreen';
import { SavedScreen } from './components/SavedScreen';
import { SearchScreen } from './components/SearchScreen';
import { SettingsScreen } from './components/SettingsScreen';
import { TabRoutes, TabLink } from './components/TabRoutes';
import { Settings } from './settings';
import { config } from './config';
import { Models, ModelsSearch, ScreenProps, SearchRecs, ServerConfig } from './datatypes';
import { log } from './log';
import { getOrientation, matchOrientation, Orientation } from './orientation';
import { StyleSheet } from './stylesheet';
import { urlpack } from './urlpack';
import { deepEqual, global, json, match, pretty, readJsonFile, Style } from './utils';

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.Animated = Animated;
global.AsyncStorage = AsyncStorage;
global.deepEqual = deepEqual;
global.Dimensions = Dimensions;
global.Linking = Linking;
global.Platform = Platform;
global.iOSColors = iOSColors;
global.material = material;
global.materialColors = materialColors;
// global.navigate = navigate; // XXX(nav_router)
global.systemWeights = systemWeights;
global.Settings = Settings;
global.urlpack = urlpack;
const timed = (desc: string, f: () => void) => { log.time(desc); f(); log.timeEnd(desc); };
global.sj = {};
timed('AudioUtils',         () => global.AudioUtils      = require('../third-party/magenta/music/transcription/audio_utils'));
// timed('d3',              () => global.d3              = require('d3'));                      // 50ms [heavy, don't need full d3]
timed('d3-color',           () => global.d3c             = require('d3-color'));                // 2ms
timed('d3-scale-chromatic', () => global.d3sc            = require('d3-scale-chromatic'));      // 6ms
timed('jimp',               () => global.Jimp            = require('jimp'));                    // 170ms
timed('lodash',             () => global._               = require('lodash'));                  // 0ms
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

type Props = {};

type State = {
  tabIndex: number;
  orientation: 'portrait' | 'landscape';
  loading: boolean;
  serverConfig?: ServerConfig;
  modelsSearch?: ModelsSearch;
  settings?: Settings;
};

export default class App extends Component<Props, State> {

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

    // Load serverConfig (async) on app startup
    const serverConfig = await readJsonFile<ServerConfig>(`${fs.dirs.MainBundleDir}/${SearchRecs.serverConfigPath}`);

    // Load modelsSearch (async) on app startup
    const modelsSearch = await readJsonFile<ModelsSearch>(`${fs.dirs.MainBundleDir}/${Models.search.path}`);

    // Load settings (async) on app startup
    const settings = await Settings.load(
      settings => this.setState({settings}), // Callback for when Settings updates
    );

    // TODO Show loading screen until loads complete
    this.setState({
      loading: false,
      serverConfig,
      modelsSearch,
      settings,
    });

  }

  // Update on !deepEqual instead of default (any setState() / props change, even if data is the same) else any setState
  // in componentDidUpdate would trigger an infinite update loop
  shouldComponentUpdate(nextProps: Props, nextState: State): boolean {
    const ret = !deepEqual(this.props, nextProps) || !deepEqual(this.state, nextState);
    // log.debug('App.shouldComponentUpdate', {ret});
    return ret;
  }

  componentDidUpdate = (prevProps: Props, prevState: State) => {
    log.info('App.componentDidUpdate', this.state.settings);
    global.settings = this.state.settings; // XXX Debug
  }

  render = () => (

    // TODO(nav_router) -> Context?
    // Pass props to screens (as props.screenProps)
    // screenProps={{
    //   serverConfig: this.state.serverConfig,
    //   modelsSearch: this.state.modelsSearch,
    //   settings: this.state.settings,
    // } as ScreenProps}

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

          // Provide Settings via context
          <Settings.Context.Provider value={this.state.settings!}>

            {/* Keep screen awake (in dev) */}
            {__DEV__ && <KeepAwake/>}

            {/* Hide status bar on all screens [I tried to toggle it on/off on different screens and got weird behaviors] */}
            <StatusBar hidden />

            {/* Top-level tab router (nested stacks will have their own router) */}
            <NativeRouter
              // Reference: props passed through to createMemoryHistory
              //  - TODO How to bound history length?
              // getUserConfirmation // () => boolean
              // initialEntries      // string[]
              // initialIndex        // number
              // keyLength           // number
            >
              <View style={styles.fill}>

                {/* HACK Globals for dev (rely on type checking to catch improper uses of these in real code) */}
                <Route children={({location, history}: {location: Location, history: MemoryHistory}) => {
                  global.routeProps = {location, history}; // (Exclude match since it's meaningless)
                  log.info('App.Route: location', json(location));
                  // log.debug('App.Route: history', history.entries.map((entry, i) => {
                  //   const current = history.index === i;
                  //   return `\n${current ? ' -> ' : '    '}${json(entry)}`;
                  // }).join());
                  return null;
                }} />

                {/* Route the back button (android) */}
                <BackButton/>

                {/* Route deep links */}
                <DeepLinking
                  prefix='birdgram-us://open'
                  action='replace'
                />

                {/* Tabs + screen pager */}
                <TabRoutes
                  defaultPath='/recent'
                  routes={[
                    {route: {path: '/record'},   label: 'Record',   iconName: 'activity', component: RecordScreen,   },
                    {route: {path: '/search'},   label: 'Search',   iconName: 'search',   component: SearchScreen,   },
                    {route: {path: '/recent'},   label: 'Recent',   iconName: 'list',     component: RecentScreen,   },
                    {route: {path: '/saved'},    label: 'Saved',    iconName: 'bookmark', component: SavedScreen,    },
                    {route: {path: '/settings'}, label: 'Settings', iconName: 'settings', component: SettingsScreen, },
                  ]}
                />

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
            </NativeRouter>

          </Settings.Context.Provider>

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
