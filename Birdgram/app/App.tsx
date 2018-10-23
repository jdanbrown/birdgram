import React from 'React';
import { Dimensions, Platform, Text, View } from 'react-native';
import FontAwesome from 'react-native-vector-icons/FontAwesome';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import ReactNav from 'react-navigation';

import { BrowseScreen } from './components/BrowseScreen';
import { SearchScreen } from './components/SearchScreen';
import { SettingsScreen } from './components/SettingsScreen';
import { SpectroScreen } from './components/SpectroScreen';
import { log } from './log';
import { global, match } from './utils';

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.Dimensions = Dimensions;
global.Platform = Platform;
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
timed('SQLite',             () => global.SQLite          = require('react-native-sqlite-storage')); // 0ms
timed('rn-fetch-blob',      () => global.RNFB            = require('rn-fetch-blob').default);   // 0ms
timed('react-native-sound', () => global.Sound           = require('react-native-sound'));      // 1ms
timed('typography',         () => global.typography      = require('react-native-typography')); // 27ms
timed('sj.zeros',           () => global.sj.zeros        = require('zeros'));                   // 0ms
// timed('sj.getPixels',    () => global.sj.getPixels    = require('get-pixels'));              // 10ms // XXX Doesn't work in RN
// timed('sj.savePixels',   () => global.sj.savePixels   = require('save-pixels'));             // 30ms // XXX Doesn't work in RN
global.fs = global.RNFB.fs;

// TODO config -> Settings.state [how to share globally?]
const config = {

  host: 'local',
  // host: 'remote',

  baseUris: {
    'local':  'http://192.168.0.195:8000',
    'remote': 'http://35.230.68.91',
  },

}

const Navigator = ReactNav.createBottomTabNavigator(
  {
    Spectro: { screen: SpectroScreen },
    Browse: { screen: BrowseScreen },
    Search: { screen: SearchScreen },
    Settings: { screen: SettingsScreen },
  },
  {
    navigationOptions: ({navigation}) => ({
      tabBarIcon: ({focused, horizontal, tintColor}) => {
        const size = horizontal ? 20 : 25;
        return match(navigation.state.key,
          ['Spectro',  (<FontAwesome5 name={'signature'} size={size} color={tintColor || undefined} />)],
          ['Browse',   (<FontAwesome5 name={'list-ul'}   size={size} color={tintColor || undefined} />)],
          ['Search',   (<FontAwesome5 name={'search'}    size={size} color={tintColor || undefined} />)],
          ['Settings', (<FontAwesome  name={'gear'}      size={size} color={tintColor || undefined} />)],
        );
      },
    }),
    tabBarOptions: {
      // activeTintColor: 'tomato',
      // inactiveTintColor: 'gray',
    },
  },
);

const App = () => (
  // https://reactnavigation.org/docs/en/state-persistence.html
  <Navigator
    key="a"
    persistenceKey={__DEV__ ? '_dev_NavigationState' : null}
  />
);

export default App;
