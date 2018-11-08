import React, { Component, ComponentType, RefObject } from 'React';
import {
  Animated, AsyncStorage, Dimensions, Linking, Platform, StatusBar, Text, View,
} from 'react-native';
import KeepAwake from 'react-native-keep-awake';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import {
  createBottomTabNavigator, createStackNavigator, NavigationContainer, NavigationParams, NavigationRoute,
  NavigationScreenProp, NavigationScreenProps,
 } from 'react-navigation';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { RecentScreen } from './components/RecentScreen';
import { RecordScreen } from './components/RecordScreen';
import { SavedScreen } from './components/SavedScreen';
import { SearchScreen } from './components/SearchScreen';
import { SettingsScreen } from './components/SettingsScreen';
import { Settings } from './settings';
import { config } from './config';
import { Models, ModelsSearch, SearchRecs, ServerConfig } from './datatypes';
import { navigate, NavParams, ScreenProps } from './nav';
import { log } from './log';
import { urlpack } from './urlpack';
import { deepEqual, global, json, match, pretty, readJsonFile, setStateAsync } from './utils';

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
global.navigate = navigate;
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

// Tab nav at top level
//  - https://reactnavigation.org/docs/en/tab-navigator.html
//  - QUESTION Add a createAppContainer? [https://reactnavigation.org/docs/en/app-containers.html]
//    - I tried briefly and gave up when `createAppContainer` wasn't declared in @types...
const TabNav = createBottomTabNavigator(
  {
    // NOTE Must bump the TabNav persistenceKey version when changing these keys (below)

    Record:   {path: 'open/Record',   screen: RecordScreen,   params: {}},

    // Stack nav for Search tab (for push/back)
    //  - https://reactnavigation.org/docs/en/stack-navigator.html
    //  - https://reactnavigation.org/docs/en/tab-based-navigation.html#a-stack-navigator-for-each-tab
    Search:   {path: 'open/Search', screen: createStackNavigator(
      {
        Search: {
          screen: SearchScreen,
          // TODO(deep_link)
          // For deep links
          //  - https://reactnavigation.org/docs/en/deep-linking.html
          // path: 'some-params', // XXX Debug
          path: ':urlParams',
        }
      },
      {
        navigationOptions: {
          header: null, // Hide header bar
        },
      },
    )},

    Recent:   {path: 'open/Recent',   screen: RecentScreen,   params: {}},
    Saved:    {path: 'open/Saved',    screen: SavedScreen,    params: {}},
    Settings: {path: 'open/Settings', screen: SettingsScreen, params: {}},

    // To pass props down [https://github.com/zeit/next.js/blob/7.0.0/lib/dynamic.js#L55]:
    //  screen: (props: object) => (<SearchScreen {...props} />),

  },
  {
    initialRouteName: 'Recent', // TODO
    navigationOptions: ({navigation}) => ({
      tabBarIcon: ({focused, horizontal, tintColor}) => {
        const size = horizontal ? 20 : 25;
        return match(navigation.state.key,
          ['Record',   (<Feather name='activity'  size={size} color={tintColor || undefined} />)],
          ['Search',   (<Feather name='search'    size={size} color={tintColor || undefined} />)],
          ['Recent',   (<Feather name='list'      size={size} color={tintColor || undefined} />)],
          ['Saved',    (<Feather name='bookmark'  size={size} color={tintColor || undefined} />)],
          ['Settings', (<Feather name='settings'  size={size} color={tintColor || undefined} />)],
        );
      },
    }),
    tabBarOptions: {
      showLabel: false,
      // activeTintColor: 'tomato',
      // inactiveTintColor: 'gray',
    },
  },
);

type Props = {};

type State = {
  loading: boolean;
  serverConfig?: ServerConfig;
  modelsSearch?: ModelsSearch;
  settings?: Settings;
};

class App extends Component<Props, State> {

  appNavRef: RefObject<NavigationContainer> = React.createRef();

  listeners: {[key: string]: any} = {};

  constructor(props: Props) {
    super(props);
    this.state = {
      loading: true,
    };
    global.App = this; // XXX Debug
  }

  componentDidMount = async () => {
    log.info('App.componentDidMount');

    // Load serverConfig (async) on app startup
    const serverConfig = await readJsonFile<ServerConfig>(`${fs.dirs.MainBundleDir}/${SearchRecs.serverConfigPath}`);

    // Load modelsSearch (async) on app startup
    const modelsSearch = await readJsonFile<ModelsSearch>(`${fs.dirs.MainBundleDir}/${Models.search.path}`);

    // Load settings (async) on app startup
    const settings = await Settings.load(
      settings => setStateAsync(this, {settings}), // Callback for when Settings updates
    );

    // TODO Show loading screen until loads complete
    await setStateAsync(this, {
      loading: false,
      serverConfig,
      modelsSearch,
      settings,
    });

    // Open app urls (/ app links / deep links)
    //  - e.g. 'birdgram-us://open/...'
    //  - Docs
    //    - https://facebook.github.io/react-native/docs/linking -- simpler approach [using this one]
    //    - https://reactnavigation.org/docs/en/deep-linking.html -- coupled with react-navigation [avoid]
    //  - Setup
    //    - ios
    //      - AppDelegate.m -> https://facebook.github.io/react-native/docs/linking#basic-usage
    //      - Xcode -> https://reactnavigation.org/docs/en/deep-linking.html#ios
    //    - TODO android
    //      - https://reactnavigation.org/docs/en/deep-linking.html#android
    //      - https://facebook.github.io/react-native/docs/linking#basic-usage
    this.listeners.url = Linking.addEventListener('url', async ({url}) => {
      // User opened an app url while app is already running
      log.info('App.listeners.url: Opening url', {url});
      await this.openUrl(url);
    });
    const initialUrl = await Linking.getInitialURL();
    if (initialUrl) {
      // User opened an app url when app was not running, and it caused the app to launch
      log.info('App.componentDidMount: Opening initialUrl', {initialUrl});
      await this.openUrl(initialUrl);
    }

  }

  componentWillUnmount = async () => {
    Linking.removeEventListener('url', this.listeners.url);
  }

  // TODO Want? (cf. SearchScreen.shouldComponentUpdate)
  // shouldComponentUpdate(nextProps: Props, nextState: State): boolean {
  //   return !deepEqual(this.props, nextProps) || !deepEqual(this.state, nextState);
  // }

  componentDidUpdate = (prevProps: Props, prevState: State) => {
    log.info('App.componentDidUpdate', this.state.settings);
    global.settings = this.state.settings; // XXX Debug
  }

  openUrl = async (url: string) => {
    log.info('App.openUrl', {url});
    // TODO(deep_link)
    //  - TODO 'birdgram-us://open/u/:tinyid' -> 'https://tinyurl.com/:tinyid' -> 'birdgram-us/open/:screen/:params'
  }

  render = () => (
    this.state.loading ? (
      <View>
        <Text>LOADING HOLD ON</Text>
      </View>
    ) : (
      // https://reactnavigation.org/docs/en/state-persistence.html
      //  - NOTE Must bump this persistenceKey version when changing the TabNav keys
      <Settings.Context.Provider value={this.state.settings!}>

        {/* Hide status bar on all screens [I tried to toggle it on/off on different screens and got weird behaviors] */}
        <StatusBar hidden />

        {/* TODO Does this work at top level? (Used to work when present in each screen) */}
        {__DEV__ && <KeepAwake/>}

        <TabNav

          // @ts-ignore [Why doesn't this typecheck?]
          ref={this.appNavRef}

          // XXX For dev: persist nav state [https://reactnavigation.org/docs/en/state-persistence.html]
          // persistenceKey={__DEV__ ? '_dev_NavigationState_v8' : null}

          // Pass props to screens (as props.screenProps)
          screenProps={{
            serverConfig: this.state.serverConfig,
            modelsSearch: this.state.modelsSearch,
            settings: this.state.settings,
          } as ScreenProps}

          // TODO(deep_link)
          // For deep links
          //  - https://reactnavigation.org/docs/en/deep-linking.html
          uriPrefix={Platform.select({
            ios:     'birdgram-us://',
            android: 'birdgram-us://open/',
          })}

          // XXX Debug
          onNavigationStateChange={(prevState: any, newState: any, action: any) => {
            log.debug('TabNav.onNavigationStateChange', pretty({
              // prevState, newState, action, // XXX Noisy
              action,
              newState: {
                ...newState,
                routes: newState.routes.map((route: {routes?: any[]}) => ({
                  ...route,
                  routes: (route.routes || []).map(x => json(x).replace(/"/g, "'")),
                })),
              },
            }));
          }}

          // TODO TODO(deep_link)
          //  - FIXME nav.state doesn't update when opening url that differs from current state only by a path :param
          /*  - Repros:

          // To observe nav states
          pp(App.appNavRef.current.state) // e.g. .nav.routes
          pp(App.appNavRef.current._navigation._childrenNavigation.Search.state) // != SearchScreen.nav.state
          pp(SearchScreen.nav.state)

          // (More observability)
          App.appNavRef.current.props // The componet props above (screenProps, uriPrefix, onNavigationStateChange)
          App.appNavRef.current.state // e.g. .nav.index = current tab index
          App.appNavRef.current._navigation.router // Lots here...

          // Second /Search has no effect, as if route is ignored if path :params is only change
          Linking.openURL('birdgram-us://open/Settings')
          Linking.openURL('birdgram-us://open/Search/url-params-1')
          Linking.openURL('birdgram-us://open/Search/url-params-2')

          // Adding a nav back to Settings in between seems to work around the issue
          Linking.openURL('birdgram-us://open/Settings')
          Linking.openURL('birdgram-us://open/Search/url-params-1')
          Linking.openURL('birdgram-us://open/Settings')
          Linking.openURL('birdgram-us://open/Search/url-params-2')

          */
          //  - TODO Try mucking with Router.getActionForPathAndParams to see if we can force it
          //    - https://reactnavigation.org/docs/en/routers.html -- getActionForPathAndParams
          //    - https://reactnavigation.org/docs/en/custom-routers.html -- way more detail

        />

      </Settings.Context.Provider>
    )
  );

}

export default App;
