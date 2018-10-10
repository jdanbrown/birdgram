// NOTE index.js must be .js (not .ts/.tsx)

import {YellowBox} from 'react-native';
YellowBox.ignoreWarnings([
  'Module AudioRecorderManager requires main queue setup', // https://github.com/jsierles/react-native-audio/issues/283
]);

import {AppRegistry} from 'react-native';
import App from './App';
import {name as appName} from './app.json';

AppRegistry.registerComponent(appName, () => App);
