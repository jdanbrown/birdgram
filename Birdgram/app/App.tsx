import React, { Component, ComponentType, RefObject } from 'React';
import { Animated, AsyncStorage, Dimensions, Platform, StatusBar, Text, View } from 'react-native';
import KeepAwake from 'react-native-keep-awake';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import {
  createBottomTabNavigator, NavigationContainer, NavigationParams, NavigationRoute, NavigationScreenProp,
  NavigationScreenProps,
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
import { SearchRecs, ServerConfig } from './datatypes';
import { NavParams, ScreenProps } from './nav';
import { log } from './log';
import { deepEqual, global, match, setStateAsync } from './utils';

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.Animated = Animated;
global.AsyncStorage = AsyncStorage;
global.deepEqual = deepEqual;
global.Dimensions = Dimensions;
global.Platform = Platform;
global.iOSColors = iOSColors;
global.material = material;
global.materialColors = materialColors;
global.systemWeights = systemWeights;
global.Settings = Settings;
const timed = (desc: string, f: () => void) => { log.time(desc); f(); log.timeEnd(desc); };
global.sj = {};
global.d3 = {};
timed('AudioUtils',         () => global.AudioUtils      = require('../third-party/magenta/music/transcription/audio_utils'));
// timed('d3',              () => global.d3              = require('d3'));                      // 50ms [heavy, don't need full d3]
timed('d3-color',           () => Object.assign(global.d3, require('d3-color')));               // 2ms
timed('d3-scale-chromatic', () => Object.assign(global.d3, require('d3-scale-chromatic')));     // 6ms
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
  loading: boolean;
  serverConfig?: ServerConfig;
  settings?: Settings;
};

class App extends Component<Props, State> {

  appNavRef: RefObject<NavigationContainer> = React.createRef();

  constructor(props: Props) {
    super(props);
    this.state = {
      loading: true,
    };
    global.App = this; // XXX Debug
  }

  componentDidMount = async () => {

    // Load serverConfig (async) on app startup
    const serverConfigJson = await fs.readFile(`${fs.dirs.MainBundleDir}/${SearchRecs.serverConfigPath}`, 'utf8');
    const serverConfig = JSON.parse(serverConfigJson);

    // Load settings (async) on app startup
    const settings = await Settings.load(
      settings => setStateAsync(this, {settings}), // Callback for when Settings updates
    );

    // TODO Show loading screen until loads complete
    await setStateAsync(this, {
      loading: false,
      serverConfig,
      settings,
    });

  }

  // TODO Want? (cf. SearchScreen.shouldComponentUpdate)
  // shouldComponentUpdate(nextProps: Props, nextState: State): boolean {
  //   return !deepEqual(this.props, nextProps) || !deepEqual(this.state, nextState);
  // }

  componentDidUpdate = (prevProps: Props, prevState: State) => {
    log.debug('App.componentDidUpdate', this.state.settings);
    global.settings = this.state.settings; // XXX Debug
  }

  render = () => (
    this.state.loading ? (
      <View>
        <Text>LOADING HOLD ON</Text>
      </View>
    ) : (
      // https://reactnavigation.org/docs/en/state-persistence.html
      //  - NOTE Must bump this persistenceKey version when changing the AppNav keys
      <Settings.Context.Provider value={this.state.settings!}>

        {/* Hide status bar on all screens [I tried to toggle it on/off on different screens and got weird behaviors] */}
        <StatusBar hidden />

        {/* TODO Does this work at top level? (Used to work when present in each screen) */}
        {__DEV__ && <KeepAwake/>}

        <AppNav
          // @ts-ignore [Why doesn't this typecheck?]
          ref={this.appNavRef}
          persistenceKey={__DEV__ ? '_dev_NavigationState_v6' : null}
          // Pass props to screens (as props.screenProps)
          screenProps={{
            serverConfig: this.state.serverConfig,
            settings: this.state.settings,
          } as ScreenProps}
        />

      </Settings.Context.Provider>
    )
  );

}

const AppNav = createBottomTabNavigator(
  {
    // NOTE Must bump the AppNav persistenceKey version when changing these keys (below)
    Record:   {screen: RecordScreen,   params: {}},
    Search:   {screen: SearchScreen,   params: {}},
    Recent:   {screen: RecentScreen,   params: {}},
    Saved:    {screen: SavedScreen,    params: {}},
    Settings: {screen: SettingsScreen, params: {}},
    // How to pass props down [https://github.com/zeit/next.js/blob/7.0.0/lib/dynamic.js#L55]:
    //  screen: (props: object) => (<SearchScreen {...props} />),
  },
  {
    navigationOptions: ({navigation}) => ({
      tabBarIcon: ({focused, horizontal, tintColor}) => {
        const size = horizontal ? 20 : 25;
        return match(navigation.state.key,
          ['Record',   (<Feather name='activity'  size={size} color={tintColor || undefined} />)],
          ['Search',   (<Feather name='search'    size={size} color={tintColor || undefined} />)],
          ['Recent',   (<Feather name='list'      size={size} color={tintColor || undefined} />)],
          ['Saved',    (<Feather name='star'      size={size} color={tintColor || undefined} />)],
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

export default App;
