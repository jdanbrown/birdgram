import React from 'react';
import {
    Button, Platform, StyleSheet, Text, View,
} from 'react-native';
import {AudioRecorder, AudioUtils} from 'react-native-audio';

export interface Props {
}

interface State {
  authorized: boolean,
  status: string,
  currentTime?: number,
  audioFileURL?: string,
  audioFileSize?: string,
}

export class Recorder extends React.Component<Props, State> {

  constructor(props: Props) {
    super(props);
    this.state = {
      authorized: false,
      status: 'Not recording',
    };
  }

  componentDidMount() {
    console.log('componentDidMount: this', this);
    console.log('componentDidMount', {AudioRecorder, AudioUtils});
    AudioRecorder.requestAuthorization().then(authorized => {
      console.debug(`AudioRecorder.requestAuthorization() -> ${authorized}`);
      console.log('componentDidMount: this', this);
      this.setState({authorized});
      if (!authorized) {
        console.warn('Failed to get microphone access');
      }
    });
  }

  async startRecording() {
    console.log('componentDidMount: this', this);
    console.log('startRecording');

    // Prepare
    const fresh = new Date().toISOString(); // TODO Add some random chars
    const preparePath = `${AudioUtils.DocumentDirectoryPath}/${fresh}.aac`;
    AudioRecorder.prepareRecordingAtPath(preparePath, {
      SampleRate: 22050,
      Channels: 1,
      AudioEncoding: "aac",
      // QUESTION What bitrate does these map to on ios? (on android?)
      AudioQuality: "Low",    // -> 15kbps?
      // AudioQuality: "Medium", // -> 20kbps?
      // AudioQuality: "High",   // -> 26kbps?
      // AudioEncodingBitRate: 64000 // Android only
    });
    AudioRecorder.onProgress = data => {
      console.log('onProgress', data);
      this.setState({currentTime: data.currentTime});
    };
    AudioRecorder.onFinished = data => {
      if (Platform.OS === 'ios') {
        // ios uses onFinished (here), android uses a promise (below)
        this.finished(data.status, data.audioFileURL, data.audioFileSize);
      }
    };

    // Start
    const startPath = await AudioRecorder.startRecording();
    this.setState({
      status: 'Recording...',
    });

  }

  async stopRecording() {
    console.log('stopRecording');
    const stopPath = await AudioRecorder.stopRecording();
    if (Platform.OS === 'android') {
      // ios uses onFinished (above), android uses a promise (here)
      this.finished('OK', stopPath, null);
    }
  }

  finished(status: string, audioFileURL: string, audioFileSize?: string) {
    console.log(
      `Finished recording: ${status}, ${audioFileURL}, ${audioFileSize}`,
    );
    this.setState({
      status,
      audioFileURL,
      audioFileSize,
    });
  }

  render() {
    return (
      <View style={styles.root}>
        <View><Text>Status: {this.state.status}</Text></View>
        <View><Text>Time: {Math.round(this.state.currentTime * 100) / 100} s</Text></View>
        <View><Text>Size: {this.state.audioFileSize} B</Text></View>
        <View><Text>Path: {this.state.audioFileURL}</Text></View>
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
