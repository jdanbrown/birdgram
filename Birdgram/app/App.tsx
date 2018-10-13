import React from 'React';
import { Component } from 'react';
import { Dimensions, Image, Platform, StyleSheet, Text, View, WebView } from 'react-native';
import KeepAwake from 'react-native-keep-awake';

import { Recorder } from './components/Recorder';
import { global } from './utils';

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
const timed = (desc: string, f: () => void) => { console.time(desc); f(); console.timeEnd(desc); };
global.sj = {};
global.d3 = {};
timed('React',              () => global.R               = require('React'));               // 0ms
timed('ReactNative',        () => global.RN              = require('ReactNative'));         // 13ms
timed('jimp',               () => global.Jimp            = require('jimp'));                // 170ms
timed('lodash',             () => global._               = require('lodash'));              // 0ms
// timed('d3',              () => global.d3              = require('d3'));                  // 50ms
timed('d3-color',           () => Object.assign(global.d3, require('d3-color')));           // 2ms
timed('d3-scale-chromatic', () => Object.assign(global.d3, require('d3-scale-chromatic'))); // 6ms
timed('ndarray',            () => global.ndarray         = require('ndarray'));             // 1ms
timed('nj',                 () => global.nj              = require('../third-party/numjs/dist/numjs.min')); // 130ms
timed('sj.ops',             () => global.sj.ops          = require('ndarray-ops'));         // 50ms
// timed('sj.getPixels',    () => global.sj.getPixels    = require('get-pixels'));          // 10ms
// timed('sj.savePixels',   () => global.sj.savePixels   = require('save-pixels'));         // 30ms
timed('sj.zeros',           () => global.sj.zeros        = require('zeros'));               // 0ms
timed('AudioUtils',         () => global.AudioUtils      = require('../third-party/magenta/music/transcription/audio_utils'));

const config = {
  // TODO -> state with a prefs view

  host: 'local',
  // host: 'remote',

  baseUris: {
    'local':  'http://192.168.0.195:8000',
    'remote': 'http://35.230.68.91',
  },

}

type Props = {};

export default class App extends Component<Props> {

  render() {

    console.log('KeepAwake', KeepAwake);

    return (
      <View style={styles.container}>

        {__DEV__ && <KeepAwake/>}

        <Text style={styles.banner}>
          Birdgram ({Platform.select({ios: 'ios', android: 'android'})})
        </Text>

        <Recorder sampleRate={22050} />

        {
          // TODO WebView: ugh, lots to do to make this work
          //  - useWebKit=true crashes in simulator (ios 9), works on phone
          //  - useWebKit=false doesn't play audio on click in simulator
          //  - No back/forward buttons (in simulator)
          //  - No keyboard (in simulator)
          //  - ...?
          // <WebView
          //   style={styles.webView}
          //   useWebKit={true}
          //   source={{uri: `${config.baseUris[config.host]}/recs/browse`}}
          // />
        }

      </View>
    );
  }

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  banner: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  // webView: {
  //   width: Dimensions.get('window').width, // Else WebView doesn't appear
  // },
});
