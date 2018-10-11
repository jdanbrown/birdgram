import React from 'react'
import {Component} from 'react';
import {Dimensions, Platform, StyleSheet, Text, View, WebView} from 'react-native';

import {Hello} from './components/Hello';
import {Recorder} from './components/Recorder';
import {Recorder2} from './components/Recorder2';

// TODO -> state with a prefs view
const config = {

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
    return (
      <View style={styles.container}>

        <Text style={styles.welcome}>Birdgram</Text>
        <Text style={styles.instructions}>{
          Platform.select({
            ios:     'Press Cmd+R to reload,\nCmd+D or shake for dev menu',
            android: 'Double tap R on your keyboard to reload,\nShake or press menu button for dev menu',
          })
        }</Text>

        {/* <Hello name={'person'} enthusiasmLevel={3} /> */}

        <Recorder2/>

        {
          // TODO Lots to do here
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
  welcome: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  instructions: {
    textAlign: 'center',
    color: '#333333',
    marginBottom: 5,
  },
  webView: {
    width: Dimensions.get('window').width, // Else WebView doesn't appear
  },
});
