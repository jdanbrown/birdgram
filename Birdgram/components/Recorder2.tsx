import React from 'react';
import {
    Button, Platform, StyleSheet, Text, View,
} from 'react-native';
import Permissions from 'react-native-permissions'
import MicStream from 'react-native-microphone-stream';

export interface Props {
}

interface State {
  status: string,
}

// https://github.com/chadsmith/react-native-microphone-stream
export class Recorder2 extends React.Component<Props, State> {

  constructor(props: Props) {
    super(props);
    this.state = {
      status: 'Not recording',
    };
  }

  componentDidMount = () => {
    console.log('componentDidMount: this', this);

    // Request mic permissions
    Permissions.request('microphone').then(status => {
      // NOTE Buggy on ios simulator [https://github.com/yonahforst/react-native-permissions/issues/58]
      //  - Recording works, but permissions always return 'undetermined'
      console.log('Permissions.request: microphone', status);
    });

    // Register callbacks
    this.listener = MicStream.addListener(data => {
      console.log('MicStream listener: data', data)
    });

  }

  componentWillUnmount = () => {
    // Unregisterd callbacks
    listener.remove();
  }

  startRecording = async () => {
    console.log('startRecording');

    // Init mic
    //  - init() here instead of componentDidMount because we have to call it after each stop()
    const eventRate = 2; // Set bufferSize to fire ~eventRate js events per sec
    const sampleRate = 22050;
    MicStream.init({
      // Options
      //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L23
      //  - https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
      //  - TODO PR to change hardcoded kAudioFormatULaw -> any mFormatID (e.g. mp4)
      bufferSize: Math.floor(sampleRate / eventRate / 2), // TODO Fishy: why do I need the extra div 2?
      sampleRate,
      bitsPerChannel: 16,
      channelsPerFrame: 1,
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

  render = () => {
    return (
      <View style={styles.root}>
        <View><Text>Status: {this.state.status}</Text></View>
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
