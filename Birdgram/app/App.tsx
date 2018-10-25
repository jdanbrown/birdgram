import React, { Component } from 'React';
import { Animated, AsyncStorage, Dimensions, Platform, Text, View } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import ReactNav from 'react-navigation';

import { RecentScreen } from './components/RecentScreen';
import { RecordScreen } from './components/RecordScreen';
import { SavedScreen } from './components/SavedScreen';
import { SearchScreen } from './components/SearchScreen';
import { SettingsScreen } from './components/SettingsScreen';
import { Settings } from './components/Settings';
import { config } from './config';
import { log } from './log';
import { deepEqual, global, match } from './utils';

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.Animated = Animated;
global.AsyncStorage = AsyncStorage;
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

const Navigator = ReactNav.createBottomTabNavigator(
  {
    // NOTE Must bump the Navigator persistenceKey version when changing these keys (below)
    Record:   { screen: RecordScreen },
    Search:   { screen: SearchScreen },
    Recent:   { screen: RecentScreen },
    Saved:    { screen: SavedScreen },
    Settings: { screen: SettingsScreen },
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

type Props = {};

type State = {
  loading: boolean;
  settings?: Settings;
};

class App extends Component<Props, State> {

  constructor(props: Props) {
    super(props);
    this.state = {
      loading: true,
    };
  }

  componentDidMount = async () => {
    // Load settings (async) on app startup
    //  - TODO Show loading screen until load completes
    const settings = await Settings.load(
      settings => this.setState({settings}), // Callback for when Settings updates
    );
    this.setState({
      loading: false,
      settings,
    });
  }

  shouldComponentUpdate(nextProps: Props, nextState: State): boolean {
    return !(deepEqual(this.props, nextProps) && deepEqual(this.state, nextState));
  }

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
      //  - NOTE Must bump this persistenceKey version when changing the Navigator keys (above)
      <Settings.Context.Provider value={this.state.settings!}>
        <Navigator
          persistenceKey={__DEV__ ? '_dev_NavigationState_v4' : null}
        />
      </Settings.Context.Provider>
    )
  );

}

export default App;
