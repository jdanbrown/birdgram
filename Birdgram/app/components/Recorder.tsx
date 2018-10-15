import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// FIXME jimp fails when not "Debug JS Remotely"
//  - "undefined is not an object (evaluating 'gl.Jimp = Jimp')"
//  - https://github.com/oliver-moran/jimp/blob/ced893d/packages/core/src/index.js#L1213-L1228
import Jimp from 'jimp';
import _ from 'lodash';
import React from 'react';
import { Button, EmitterSubscription, Image, Platform, StyleSheet, Text, View } from 'react-native';
import FastImage from 'react-native-fast-image';
import MicStream from 'react-native-microphone-stream';
import Permissions from 'react-native-permissions';
import RNFB from 'rn-fetch-blob';
const {fs, base64} = RNFB;

import { magSpectrogram, melSpectrogram, powerToDb, stft } from '../../third-party/magenta/music/transcription/audio_utils'
import nj from '../../third-party/numjs/dist/numjs.min';
import { chance, global } from '../utils';

export interface Props {
  sampleRate: number,
}

enum RecordingState {
  Stopped = 'Stopped',
  Recording = 'Recording',
}

interface State {
  recordingState: RecordingState,
  audioSampleChunks: number[][],
  spectroImgProps?: {source: object, style?: object},
}

// https://github.com/chadsmith/react-native-microphone-stream
export class Recorder extends React.Component<Props, State> {

  listener?: EmitterSubscription;

  constructor(props: Props) {
    super(props);
    this.state = {
      recordingState: RecordingState.Stopped,
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
    if (this.state.recordingState === RecordingState.Stopped) {
      console.log('startRecording');

      // Reset audio samples
      this.setState({
        audioSampleChunks: [],
        spectroImgProps: undefined,
      });

      // Init mic
      //  - init() here instead of componentDidMount because we have to call it after each stop()
      const eventRate = 4; // Set bufferSize to fire ~eventRate js events per sec
      const {sampleRate} = this.props;
      MicStream.init({
        // Options
        //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L23-L32
        //  - https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
        //  - TODO PR to change hardcoded kAudioFormatULaw -> any mFormatID (e.g. mp4)
        sampleRate, // (QA'd via tone generator -> MicStream -> magSpectrogram)
        bitsPerChannel: 16,
        channelsPerFrame: 1,
        // Compute a bufferSize that will fire at eventRate
        //  - Hardcode `/ 2` here because MicStream does `* 2`
        //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L35
        bufferSize: Math.floor(sampleRate / eventRate / 2),
      });

      // Start recording
      this.setState({recordingState: RecordingState.Recording});
      MicStream.start(); // (Async with no signal of success/failure)

      // Display recorded audio (~empty)
      await this.drawSpectro();

    }
  }

  onRecordedChunk = async (samples: number[]) => {
    // MicStream.stop is unobservably async, so ignore any audio capture after we think we've stopped
    if (this.state.recordingState === RecordingState.Recording) {
      console.log('onRecordedChunk: samples', samples)

      // Buffer samples
      this.setState((state, props) => ({
        audioSampleChunks: [...state.audioSampleChunks, samples],
      }));

      // Display recorded audio (incremental)
      await this.drawSpectro();

    }
  }

  stopRecording = async () => {
    if (this.state.recordingState === RecordingState.Recording) {
      console.log('stopRecording');

      // Stop recording
      this.setState({recordingState: RecordingState.Stopped});
      MicStream.stop(); // (Async with no signal of success/failure)

      // Display recorded audio (final)
      await this.drawSpectro();

    }
  }

  drawSpectro = async (): Promise<void> => {

    // Util: wrap `new Jimp` in a promise
    const JimpAsync = (...args: any[]): Promise<Jimp> => new Promise((resolve, reject) => {
      new Jimp(...args, (err: Error | null, img: Jimp) => err ? reject(err) : resolve(img));
    })

    // https://github.com/d3/d3-scale-chromatic#sequential-multi-hue

    // Concat audio from chunks
    const chunks = this.state.audioSampleChunks;
    const audio = _.concat([], ...chunks);

    // Compute (mag) spectro from audio
    const {sampleRate} = this.props;

    // pow=1 without powerToDb kinda works, but pow=2 with powerToDb works way better
    const [pow, db] = [
      // 1, false, // Barely
      // 2, false, // Junk
      // 2, true,  // Decent
      3, true,  // Good enough (until we melSpectrogram)
    ];
    let [spectro, nfft] = magSpectrogram(
      stft(new Float32Array(audio), {sampleRate}), // (QA'd via tone generator -> MicStream -> magSpectrogram)
      pow,
    );
    if (db) spectro = powerToDb(spectro);
    let S = nj.array(spectro);

    // Normalize values to [0,1]
    //  - QUESTION max or max-min?
    console.log('S.min, S.max', S.min(), S.max());
    S = S.subtract(S.min());
    S = S.divide(S.max());
    console.log('S.min, S.max', S.min(), S.max());

    console.log('nfft', nfft);
    console.log('spectro.shape', S.shape);

    // Compute imageRGBA from S
    const [w_S, h_S] = S.shape;
    const imageRGBA = new Buffer(w_S * h_S * 4);
    for (let w = 0; w < w_S; w++) {
      for (let h = 0; h < h_S; h++) {
        const x = S.get(w, -h) as unknown as number; // (Fix bad type: S.get(w,h): Float32Array)
        const c = color(interpolateMagma(1 - x)) as RGBColor;
        if (c) {
          imageRGBA[0 + 4*(w + w_S*h)] = c.r;
          imageRGBA[1 + 4*(w + w_S*h)] = c.g;
          imageRGBA[2 + 4*(w + w_S*h)] = c.b;
          imageRGBA[3 + 4*(w + w_S*h)] = c.opacity * 255;
        }
      }
    }

    // Build image from imageRGBA
    //  - XXX Very slow, kills app after ~5s
    const [w_img, h_img] = [
      w_S,
      400,
    ];
    let img = await JimpAsync({
      data: imageRGBA,
      width: w_S,
      height: h_S,
    })
    img = (img
      // .crop(0, 0, w_img, h_img)
      .resize(w_img, h_img) // TODO Is it faster to resize before data url, or let Image.props.resizeMode do the resizing for us?
    );

    // XXX Globals for dev
    Object.assign(global, {
      chunks, audio, spectro, S, w_S, h_S, w_img, h_img, imageRGBA,
    });

    // Render image [via file]
    //  - Scratch: fs.createFile(pngPath, imageRGBA.toString('base64'), 'base64');
    const dataUrl = await img.getBase64Async('image/png');
    const filename = `${new Date().toISOString()}-${chance.hash({length: 8})}.png`;
    const pngPath = `${fs.dirs.CacheDir}/${filename}`;
    const pngBase64 = dataUrl.replace('data:image/png;base64,', '');
    await fs.createFile(pngPath, pngBase64, 'base64');
    this.setState({
      spectroImgProps: {
        source: {uri: `file://${pngPath}`},
        style: {width: w_img, height: h_img}, // For file:// uris, else image doesn't show
      },
    });

    // // Render image [via dataUrl]
    // //  - XXX Faster via file(s)?
    // const dataUrl = await img.getBase64Async('image/png');
    // this.setState({
    //   spectroImgProps: {
    //     style: {width: w_img, height: h_img},
    //     source: {uri: dataUrl},
    //   },
    // });

  }

  // TODO Bigger button hitbox: https://stackoverflow.com/questions/50065928/change-button-font-size-on-react-native
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

        {
          // HACK Using FastImage instead of Image to avoid RCTLog "Reloading image <dataUrl>" killing my rndebugger...
          //  - https://github.com/facebook/react-native/blob/1151c09/Libraries/Image/RCTImageView.m#L422
          //  - https://github.com/DylanVann/react-native-fast-image
          this.state.spectroImgProps && <FastImage {...this.state.spectroImgProps} />
        }

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
