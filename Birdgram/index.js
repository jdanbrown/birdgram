// This entrypoint file (index.js) must be .js i/o .ts

//
// Provide node globals for packages that use them
//  - https://github.com/parshap/node-libs-react-native
//  - TODO Add fs,net,dgram (native code + nontrivial instructions, install as needed)
//    - https://github.com/parshap/node-libs-react-native#other-react-native-modules
//

// Make sure this is `import` instead of `require(...)`, else things almost work but don't [modules, how do they work?]
import 'node-libs-react-native/globals';

//
// Birdgram
//

// Warnings to ignore
import {YellowBox} from 'react-native';
YellowBox.ignoreWarnings([
  'Module AudioRecorderManager requires main queue setup', // https://github.com/jsierles/react-native-audio/issues/283
]);

//
// Boilerplate for react-native app
//

import {AppRegistry} from 'react-native';
import App from './app/App';
import {name as appName} from './app.json';

AppRegistry.registerComponent(appName, () => App);
