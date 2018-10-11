import _ from 'lodash';
// import * as mm from '@magenta/music';
import React from 'react';
import {
    Button, EmitterSubscription, Platform, StyleSheet, Text, View,
} from 'react-native';
import Permissions from 'react-native-permissions'
import MicStream from 'react-native-microphone-stream';

// import * as mm from '@magenta/music'; // XXX Imports tfjs which fails on mobile ("No backend found in registry")
import * as AudioUtils from '../../third-party/magenta/music/transcription/audio_utils'

export interface Props {
}

interface State {
  status: string,
  audioSampleChunks: number[][],
}

// https://github.com/chadsmith/react-native-microphone-stream
export class Recorder extends React.Component<Props, State> {

  listener?: EmitterSubscription;

  constructor(props: Props) {
    super(props);
    this.state = {
      status: 'Not recording',
      audioSampleChunks: [],
    };
  }

  componentDidMount = () => {
    console.log('componentDidMount: this', this);

    console.log('AudioUtils', AudioUtils); // XXX

    // Request mic permissions
    Permissions.request('microphone').then(status => {
      // NOTE Buggy on ios simulator [https://github.com/yonahforst/react-native-permissions/issues/58]
      //  - Recording works, but permissions always return 'undetermined'
      console.log('Permissions.request: microphone', status);
    });

    // Register callbacks
    this.listener = MicStream.addListener(this.onRecordedChunk);

  }

  componentWillUnmount = () => {
    // Unregisterd callbacks
    if (this.listener) this.listener.remove();
  }

  startRecording = async () => {
    console.log('startRecording');

    // Init mic
    //  - init() here instead of componentDidMount because we have to call it after each stop()
    const eventRate = 4; // Set bufferSize to fire ~eventRate js events per sec
    const sampleRate = 22050;
    MicStream.init({
      // Options
      //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L23-L32
      //  - https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
      //  - TODO PR to change hardcoded kAudioFormatULaw -> any mFormatID (e.g. mp4)
      sampleRate,
      bitsPerChannel: 16,
      channelsPerFrame: 1,
      // Compute a bufferSize that will fire at eventRate
      //  - Hardcode `/ 2` here because MicStream does `* 2`
      //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L35
      bufferSize: Math.floor(sampleRate / eventRate / 2),
    });

    // Start recording
    MicStream.start();
    this.setState({status: 'Recording'});

  }

  stopRecording = async () => {
    console.log('stopRecording');
    // Stop recording
    MicStream.stop();
    this.setState({status: 'Stopped'});
  }

  onRecordedChunk = (samples: number[]) => {
    console.log('onRecordedChunk: samples', samples)
    this.setState((state, props) => ({
      audioSampleChunks: [...state.audioSampleChunks, samples],
    }));
  }

  render = () => {
    return (
      <View style={styles.root}>
        <View>
          <Text>Math.log10: {Math.log10.toString()}</Text>
        </View>
        <View>
          <Text>Status: {this.state.status}</Text>
        </View>
        <View>
          <Text>Chunks: {this.state.audioSampleChunks.length}</Text>
        </View>
        <View>
          <Text>Samples: {_.sum(this.state.audioSampleChunks.map((x: number[]) => x.length))}</Text>
        </View>
        <View style={styles.buttons}>
          <View style={styles.button}>
            <Button title="▶️" onPress={this.startRecording.bind(this)} />
          </View>
          <View style={styles.button}>
            <Button title="⏹️" onPress={this.stopRecording.bind(this)} />
          </View>
        </View>
      </View>
    );
  }

}

const styles = StyleSheet.create({
  root: {
    alignItems: 'center',
    alignSelf: 'center',
  },
  buttons: {
    flexDirection: 'row',
    minHeight: 70,
    alignItems: 'stretch',
    alignSelf: 'center',
    borderWidth: 1,
    borderColor: 'lightgray',
  },
  button: {
    flex: 1,
    paddingVertical: 0,
  },
});
