import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
import Jimp from 'jimp';
import _ from 'lodash';
import React from 'react';
import { Button, EmitterSubscription, Image, Platform, StyleSheet, Text, View } from 'react-native';
import Permissions from 'react-native-permissions'
import MicStream from 'react-native-microphone-stream';

import * as AudioUtils from '../../third-party/magenta/music/transcription/audio_utils'
import nj from '../../third-party/numjs/dist/numjs.min';
import { global } from '../utils';

export interface Props {
  sampleRate: number,
}

interface State {
  recordingState: string,
  audioSampleChunks: number[][],
  spectroImage?: Element, // QUESTION What is Element? It made tsc happy...
}

// https://github.com/chadsmith/react-native-microphone-stream
export class Recorder extends React.Component<Props, State> {

  listener?: EmitterSubscription;

  constructor(props: Props) {
    super(props);
    this.state = {
      recordingState: 'stopped',
      audioSampleChunks: [],
    };
    global.Recorder = this; // XXX dev
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
    this.listener = MicStream.addListener(this.onRecordedChunk);

  }

  componentWillUnmount = () => {
    // Unregisterd callbacks
    if (this.listener) this.listener.remove();
  }

  startRecording = async () => {
    console.log('startRecording');

    // Clear audio data
    this.setState({audioSampleChunks: []});

    // Init mic
    //  - init() here instead of componentDidMount because we have to call it after each stop()
    const eventRate = 4; // Set bufferSize to fire ~eventRate js events per sec
    const {sampleRate} = this.props;
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
    this.setState({recordingState: 'recording'});
    MicStream.start();

  }

  onRecordedChunk = (samples: number[]) => {
    // MicStream.stop is unobservably async, so ignore any audio capture after we think we've stopped
    if (this.state.recordingState === 'recording') {
      console.log('onRecordedChunk: samples', samples)
      this.setState((state, props) => ({
        audioSampleChunks: [...state.audioSampleChunks, samples],
      }));
    }
    // Display recorded audio
    this.drawSpectro();
  }

  stopRecording = async () => {
    console.log('stopRecording');
    // Stop recording
    MicStream.stop();
    this.setState({recordingState: 'stopped'});
    // Display recorded audio
    this.drawSpectro();
  }

  drawSpectro = (): Promise<void> => {

    // Util: wrap `new Jimp` in a promise
    const JimpPromise = (...args: any[]): Promise<Jimp> => new Promise((resolve, reject) => {
      new Jimp(...args, (err: Error | null, img: Jimp) => err ? reject(err) : resolve(img));
    })

    // https://github.com/d3/d3-scale-chromatic#sequential-multi-hue

    // Concat audio from chunks
    const chunks = this.state.audioSampleChunks;
    const audio = _.concat([], ...chunks);

    // Compute (mag) spectro from audio
    const {sampleRate} = this.props;
    const pow = 1; // NOTE pow=2 is invisible with magSpectrogram [obviated once we move to melSpectrogram]
    const [spectro, nfft] = AudioUtils.magSpectrogram(
      AudioUtils.stft(new Float32Array(audio), {sampleRate}),
      pow,
    );
    let S = nj.array(spectro);
    console.log('nfft', nfft);
    console.log('spectro.shape', S.shape);

    // Normalize values to [0,1]
    S = nj.divide(S, S.max());

    // Compute imageRGBA from S
    const [w_S, h_S] = S.shape;
    const imageRGBA = new Buffer(w_S * h_S * 4);
    for (let w = 0; w < w_S; w++) {
      for (let h = 0; h < h_S; h++) {
        // TODO Is this loop transposed?
        const x = S.get(w, -h) as unknown as number; // HACK Fix incorrect type: S.get(w,h): Float32Array
        const c = color(interpolateMagma(1 - x)) as RGBColor;
        imageRGBA[0 + 4*(w + w_S*h)] = c.r;
        imageRGBA[1 + 4*(w + w_S*h)] = c.g;
        imageRGBA[2 + 4*(w + w_S*h)] = c.b;
        imageRGBA[3 + 4*(w + w_S*h)] = c.opacity * 255;
      }
    }

    // Build image from imageRGBA
    const [w_img, h_img] = [
      w_S,
      400,
    ];
    return JimpPromise({
      data: imageRGBA,
      width: w_S,
      height: h_S,
    }).then(img => (img
      // .crop(0, 0, w_img, h_img)
      .resize(w_img, h_img)
      .getBase64Async('image/png')
    )).then(dataUrl => {

      console.log('dataUrl', dataUrl);

      // XXX Globals for dev
      Object.assign(global, {
        chunks, audio, spectro, S, w_S, h_S, w_img, h_img, imageRGBA,
      });

      // Render image
      this.setState({
        spectroImage: (
          <Image style={{width: w_img, height: h_img}} source={{uri: dataUrl}} />
        ),
      });

    });

  }

  render = () => {
    return (
      <View style={styles.root}>

        <View>
          <Text>recordingState: {this.state.recordingState}</Text>
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

        {this.state.spectroImage}

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
