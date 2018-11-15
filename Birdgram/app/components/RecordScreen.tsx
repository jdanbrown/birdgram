import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// console.warn('XXX: RecordScreen: import jimp ...'); // XXX Debug
import Jimp from 'jimp';
// console.warn('XXX: RecordScreen: import jimp: done'); // XXX Debug
import _ from 'lodash';
import React, { PureComponent } from 'react';
import { Button, Dimensions, EmitterSubscription, Image, Platform, Text, TextProps, View, ViewProps } from 'react-native';
import FastImage from 'react-native-fast-image';
import { BaseButton, BorderlessButton, LongPressGestureHandler, RectButton, TapGestureHandler } from 'react-native-gesture-handler';
import MicStream from 'react-native-microphone-stream';
import Permissions from 'react-native-permissions';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import RNFB from 'rn-fetch-blob';
const {fs, base64} = RNFB;

import { magSpectrogram, melSpectrogram, powerToDb, stft } from '../../third-party/magenta/music/transcription/audio_utils'
import nj from '../../third-party/numjs/dist/numjs.min';
import { SourceId } from '../datatypes';
import { log } from '../log';
import { Go } from '../router';
import { SettingsWrites } from '../settings';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, Styles } from '../styles';
import { chance, global, match, shallowDiffPropsState } from '../utils';

// Util: wrap `new Jimp` in a promise
const JimpAsync = (...args: Array<any>): Promise<Jimp> => new Promise((resolve, reject) => {
  new Jimp(...args, (err: Error | null, img: Jimp) => err ? reject(err) : resolve(img));
});

enum RecordingState {
  Stopped = 'Stopped',
  Recording = 'Recording',
}

export interface Props {
  // App globals
  go: Go;
  // Settings
  settings: SettingsWrites;
  showDebug: boolean;
  // RecordScreen
  sampleRate: number;
  refreshRate: number;
  spectroHeight: number;
}

interface State {
  recordingState: RecordingState,
  audioSampleChunks: Array<Array<number>>,
  spectroChunksImageProps: Array<{source: {uri: string}, style?: object}>,
}

// https://github.com/chadsmith/react-native-microphone-stream
export class RecordScreen extends React.Component<Props, State> {

  static defaultProps = {
    sampleRate: 22050,
    refreshRate: 2, // TODO Make updates faster so we can refresh at lower latency
    spectroHeight: 400,
  };

  state: State = {
    recordingState: RecordingState.Stopped,
    audioSampleChunks: [],
    spectroChunksImageProps: [],
  };

  // Getters for props
  get spectroChunksPerScreenWidth(): number { return Dimensions.get('window').width / (44 / this.props.refreshRate) + 1; }

  // Private attrs
  _listener?: EmitterSubscription;

  componentDidMount = () => {
    log.info(`${this.constructor.name}.componentDidMount`);
    global.RecordScreen = this; // XXX Debug

    // Request mic permissions
    Permissions.request('microphone').then(status => {
      // NOTE Buggy on ios simulator [https://github.com/yonahforst/react-native-permissions/issues/58]
      //  - Recording works, but permissions always return 'undetermined'
      log.info('Permissions.request: microphone', status);
    });

    // Register callbacks
    this._listener = MicStream.addListener(this.onRecordedChunk);

  }

  componentWillUnmount = () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
    // Unregisterd callbacks
    if (this._listener) this._listener.remove();
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));
  }

  // TODO Bigger button hitbox: https://stackoverflow.com/questions/50065928/change-button-font-size-on-react-native
  render = () => {
    log.info(`${this.constructor.name}.render`);
    return (
      <View style={[
        Styles.fill,
        Styles.center,
      ]}>

        <View style={{
          ...Styles.center,
          marginVertical: 5,
          flexGrow: 1,
          flexDirection: 'row',
          justifyContent: 'flex-end',
          width: Dimensions.get('window').width,
          height: this.props.spectroHeight,
        }}>
          {
            // HACK Using FastImage instead of Image to avoid RCTLog "Reloading image <dataUrl>" killing my rndebugger...
            //  - https://github.com/facebook/react-native/blob/1151c09/Libraries/Image/RCTImageView.m#L422
            //  - https://github.com/DylanVann/react-native-fast-image
            (this.state.spectroChunksImageProps
              .slice(-this.spectroChunksPerScreenWidth)
              .map(props => (
                <FastImage key={props.source.uri} {...props} />
              ))
            )
          }
        </View>

        <View style={{
          flexShrink: 1,
          marginTop: 5,
        }}>
          <RectButton
            style={{
              ...Styles.center,
              width: Dimensions.get('window').width,
              paddingVertical: 15,
              backgroundColor: match(this.state.recordingState,
                [RecordingState.Stopped,   '#b2df8a'],
                [RecordingState.Recording, '#fb9a99'],
              ),
            }}
            onPress={match(this.state.recordingState,
              [RecordingState.Stopped,   this.startRecording],
              [RecordingState.Recording, this.stopRecording],
            )}
          >
            <FontAwesome5
              style={{
                ...material.display2Object,
                color: iOSColors.black,
              }}
              // size={50} color={'black'} backgroundColor={'white'} borderRadius={0} iconStyle={{marginRight: 0}}
              name={match(this.state.recordingState,
                [RecordingState.Stopped,   'play'],
                [RecordingState.Recording, 'stop'],
              )}
            />
          </RectButton>
        </View>

        <this.DebugView style={{
          marginVertical: 5,
          width: '100%',
        }}>
          <this.DebugText>
            state.recordingState: {this.state.recordingState}
          </this.DebugText>
          <this.DebugText>
            state.audioSampleChunks: {this.state.audioSampleChunks.length}
          </this.DebugText>
          <this.DebugText>
            state.audioSampleChunks.sum: {_.sum(this.state.audioSampleChunks.map((x: Array<number>) => x.length))}
          </this.DebugText>
          <this.DebugText>
            state.spectroChunksImageProps: {this.state.spectroChunksImageProps.length}
          </this.DebugText>
        </this.DebugView>

      </View>
    );
  }

  startRecording = async () => {
    if (this.state.recordingState === RecordingState.Stopped) {
      log.info('startRecording');

      // Update recordingState + reset audio chunks
      this.setState({
        recordingState: RecordingState.Recording,
        audioSampleChunks: [],
        spectroChunksImageProps: [],
      });

      // Init mic
      //  - init() here instead of componentDidMount because we have to call it after each stop()
      const {sampleRate, refreshRate} = this.props;
      MicStream.init({
        // Options
        //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L23-L32
        //  - https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
        //  - TODO PR to change hardcoded kAudioFormatULaw -> any mFormatID (e.g. mp4)
        sampleRate, // (QA'd via tone generator -> MicStream -> magSpectrogram)
        bitsPerChannel: 16,
        channelsPerFrame: 1,
        // Compute a bufferSize that will fire ~refreshRate buffers per sec
        //  - Hardcode `/2` here to counteract the `*2` in MicStream...
        //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L35
        bufferSize: Math.floor(sampleRate / refreshRate / 2),
      });

      // Start recording
      MicStream.start(); // (Async with no signal of success/failure)

    }
  }

  onRecordedChunk = async (samples: Array<number>) => {
    // MicStream.stop is unobservably async, so ignore any audio capture after we think we've stopped
    if (this.state.recordingState === RecordingState.Recording) {
      // log.debug('onRecordedChunk: samples', samples.length,
      //   // samples, // Noisy
      // );

      // Buffer samples
      this.setState((state, props) => ({
        audioSampleChunks: ([...state.audioSampleChunks, samples]
          // HACK Trim audio for O(1) mem usage -- think harder how to buffer long durations of audio (e.g. files i/o ram)
          .slice(-this.spectroChunksPerScreenWidth)
        ),
      }));

      // Display recorded audio (incremental)
      await this.renderChunk(samples);

    }
  }

  stopRecording = async () => {
    if (this.state.recordingState === RecordingState.Recording) {
      log.info('stopRecording');

      // Update recordingState
      this.setState({
        recordingState: RecordingState.Stopped,
      });

      // Stop recording
      MicStream.stop(); // (Async with no signal of success/failure)

    }
  }

  renderChunk = async (chunk: Array<number>): Promise<void> => {

    // TODO Include previous chunk(s) in stft
    //  - Defer until melSpectrogram, so we can couple to the right mel params
    const audio = chunk;

    // Compute (mag) spectro from audio
    const {sampleRate} = this.props;
    const [pow, db] = [
      // pow=1 without powerToDb kinda works, but pow=2 with powerToDb works way better
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
    // log.debug('S.min, S.max', S.min(), S.max());
    S = S.subtract(S.min());
    S = S.divide(S.max());
    // log.debug('S.min, S.max', S.min(), S.max());

    // log.debug('nfft', nfft);
    // log.debug('spectro.shape', S.shape);

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
      this.props.spectroHeight,
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

    // Render image (via file)
    //  - Scratch: fs.createFile(pngPath, imageRGBA.toString('base64'), 'base64');
    const dataUrl = await img.getBase64Async('image/png');
    const filename = `${new Date().toISOString()}-${chance.hash({length: 8})}.png`;
    const pngPath = `${fs.dirs.CacheDir}/${filename}`;
    const pngBase64 = dataUrl.replace('data:image/png;base64,', '');
    await fs.createFile(pngPath, pngBase64, 'base64');
    this.setState((state, props) => ({
      spectroChunksImageProps: (
        [...state.spectroChunksImageProps, {
          source: {uri: `file://${pngPath}`},
          style: {width: w_img, height: h_img}, // For file:// uris, else image doesn't show
        }]
        .slice(-this.spectroChunksPerScreenWidth) // Trim spectro chunks for O(1) mem usage
      ),
    }));

    // // XXX Globals for dev
    // Object.assign(global, {
    //   audio, spectro, S, w_S, h_S, w_img, h_img, imageRGBA,
    // });

  }

  // Debug components
  //  - [Tried and gave up once to make well-typed generic version of these (DebugFoo = addStyle(Foo, ...) = withProps(Foo, ...))]
  DebugView = (props: ViewProps & {children: any}) => (
    !this.props.showDebug ? null : (
      <View {...{
        ...props,
        style: [Styles.debugView, ...normalizeStyle(props.style)],
      }}/>
    )
  );
  DebugText = (props: TextProps & {children: any}) => (
    !this.props.showDebug ? null : (
      <Text {...{
        ...props,
        style: [Styles.debugText, ...normalizeStyle(props.style)],
      }}/>
    )
  );

}

const styles = StyleSheet.create({
})
