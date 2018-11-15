import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// console.warn('XXX: RecordScreen: import jimp ...'); // XXX Debug
import Jimp from 'jimp';
// console.warn('XXX: RecordScreen: import jimp: done'); // XXX Debug
import _ from 'lodash';
import React, { PureComponent } from 'react';
import { Button, Dimensions, EmitterSubscription, Image, Platform, Text, TextProps, View, ViewProps } from 'react-native';
import AudioRecord from 'react-native-audio-record';
import FastImage from 'react-native-fast-image';
import { BaseButton, BorderlessButton, LongPressGestureHandler, RectButton, TapGestureHandler } from 'react-native-gesture-handler';
import MicStream from 'react-native-microphone-stream';
import Permissions from 'react-native-permissions';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import RNFB from 'rn-fetch-blob';
import WaveFile from 'wavefile/dist/wavefile';
const {fs, base64} = RNFB;

import { magSpectrogram, melSpectrogram, powerToDb, stft } from '../../third-party/magenta/music/transcription/audio_utils'
import nj from '../../third-party/numjs/dist/numjs.min';
import { SourceId } from '../datatypes';
import { log } from '../log';
import { Go } from '../router';
import { SettingsWrites } from '../settings';
import Sound from '../sound';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, Styles } from '../styles';
import { chance, global, json, match, pretty, shallowDiffPropsState } from '../utils';

// Util: wrap `new Jimp` in a promise
const JimpAsync = (...args: Array<any>): Promise<Jimp> => new Promise((resolve, reject) => {
  new Jimp(...args, (err: Error | null, img: Jimp) => err ? reject(err) : resolve(img));
});

enum RecordingState {
  Stopped = 'Stopped',
  Recording = 'Recording',
  Saving = 'Saving',
}

export interface Props {
  // App globals
  go: Go;
  // Settings
  settings: SettingsWrites;
  showDebug: boolean;
  // RecordScreen
  library: 'MicStream' | 'AudioRecord';
  sampleRate: number;
  channels: number;
  refreshRate: number;
  spectroHeight: number;
  renderBufferSize: number;
}

interface State {
  recordingState: RecordingState;
  spectroChunksImageProps: Array<SpectroChunksImageProps>;
}

interface SpectroChunksImageProps {
  source: {uri: string};
  style?: object;
}

// https://github.com/chadsmith/react-native-microphone-stream
export class RecordScreen extends React.Component<Props, State> {

  static defaultProps = {
    library: 'AudioRecord', // TODO(write_audio)
    sampleRate: 22050,
    channels: 1,
    // refreshRate: 2, // TODO Make updates faster so we can refresh at lower latency
    refreshRate: 22050 / 2048, // HACK 2048 hardcoded in react-native-audio-record/ios/RNAudioRecord.m
    spectroHeight: 400,
    renderBufferSize: 1024 * 8,
  };

  state: State = {
    recordingState: RecordingState.Stopped,
    spectroChunksImageProps: [],
  };

  // Getters for props
  get spectroChunksPerScreenWidth(): number {
    return Math.ceil(Dimensions.get('window').width / (44 / this.props.refreshRate) + 1);
  }

  // Private attrs
  _listener?: EmitterSubscription;
  _audioSampleChunks: Array<Array<number>> = [];

  componentDidMount = () => {
    log.info(`${this.constructor.name}.componentDidMount`);
    global.RecordScreen = this; // XXX Debug

    // Request mic permissions
    Permissions.request('microphone').then(status => {
      // NOTE Buggy on ios simulator [https://github.com/yonahforst/react-native-permissions/issues/58]
      //  - Recording works, but permissions always return 'undetermined'
      log.info('[componentDidMount] Permissions.request: microphone', status);
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
                [RecordingState.Saving,    '#ffff99'],
              ),
            }}
            onPress={match(this.state.recordingState,
              [RecordingState.Stopped,   this.startRecording],
              [RecordingState.Recording, this.stopRecording],
              [RecordingState.Saving,    () => {}],
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
                [RecordingState.Saving,    'spinner'],
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
            _audioSampleChunks: {this._audioSampleChunks.length}
          </this.DebugText>
          <this.DebugText>
            _audioSampleChunks.sum: {_.sum(this._audioSampleChunks.map(x => x.length))}
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
      log.info('[startRecording]');

      // Update recordingState + reset audio chunks
      this._audioSampleChunks = [];
      this.setState({
        recordingState: RecordingState.Recording,
        spectroChunksImageProps: [],
      });

      if (this.props.library === 'MicStream') {

        // Init mic
        //  - init() here instead of componentDidMount because we have to call it after each stop()
        const {sampleRate, refreshRate} = this.props;
        MicStream.init({
          // Options
          //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L23-L32
          //  - https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
          //  - TODO PR to change hardcoded kAudioFormatULaw -> any mFormatID (e.g. mp4)
          sampleRate, // (QA'd via tone generator -> MicStream -> magSpectrogram)
          bitsPerChannel: 16, // TODO(write_audio)
          // bitsPerChannel: 8, // XXX(write_audio)
          channelsPerFrame: 1,
          // Compute a bufferSize that will fire ~refreshRate buffers per sec
          //  - Hardcode `/2` here to counteract the `*2` in MicStream...
          //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L35
          bufferSize: Math.floor(sampleRate / refreshRate / 2),
        });

        // Start recording
        MicStream.start(); // (Async with no signal of success/failure)

      } else if (this.props.library === 'AudioRecord') {

        // TODO(write_audio)
        AudioRecord.init({
          sampleRate: this.props.sampleRate,
          channels: this.props.channels,
          bitsPerSample: 16,  // TODO(write_audio)
          wavFile: 'AudioRecord-test-0.wav',
        });
        AudioRecord.start();

        // TODO(write_audio)
        AudioRecord.on('data', (data: string) => {
          const samples = Buffer.from(data, 'base64'); // Decode base64 string -> uint8 audio samples
          this.onRecordedChunk(Array.from(samples));
        });

      }

    }
  }

  stopRecording = async () => {
    if (this.state.recordingState === RecordingState.Recording) {
      log.info('[stopRecording]');

      this.setState({
        recordingState: RecordingState.Saving,
      });

      let wavPath;
      if (this.props.library === 'MicStream') {

        // Stop recording
        log.debug('[stopRecording] Stopping mic');
        MicStream.stop(); // (Async with no signal of success/failure)

        // TODO(write_audio)

        // Encode audioSamples as wav data
        //  - https://github.com/rochars/wavefile#create-wave-files-from-scratch
        //  - https://github.com/rochars/wavefile#the-wavefile-methods
        let audioSamples = _.flatten(this._audioSampleChunks);
        // audioSamples = audioSamples.map(x => x - 128) // XXX Error: "Overflow at input index 14: -1"
        const wav = new WaveFile()
        wav.fromScratch(
          this.props.channels,
          this.props.sampleRate,
          // MicStream records using kAudioFormatULaw
          //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L31
          // '8',
          // '8a',
          '8m', // 8-bit int, mu-Law
          audioSamples,
        );
        wav.fromMuLaw(); // TODO "Decode 8-bit mu-Law as 16-bit"

        // Write wav data to file
        const filename = `${new Date().toISOString()}-${chance.hash({length: 8})}.wav`;
        wavPath = `${fs.dirs.CacheDir}/${filename}`;
        // await fs.createFile(wavPath, audioSamples as unknown as string, 'ascii'); // HACK Ignore bad type
        await fs.createFile(wavPath, Array.from(wav.toBuffer()) as unknown as string, 'ascii'); // HACK Ignore bad type
        const {size} = await fs.stat(wavPath);
        log.debug('[stopRecording] Wrote file', json({wavPath, size}));

        // XXX Debug
        global.audioSamples = audioSamples;
        global.wav = wav;

      } else if (this.props.library === 'AudioRecord') {

        // TODO(write_audio)
        wavPath = await AudioRecord.stop();

      }

      const sound = await Sound.newAsync(wavPath);

      // XXX Debug
      global.wavPath = wavPath;
      global.sound = sound;

      // XXX Debug
      Sound.setActive(true);
      Sound.setCategory(
        'PlayAndRecord',
        true, // mixWithOthers
      );
      Sound.setMode(
        'Default',
        // 'Measurement', // XXX? like https://github.com/jsierles/react-native-audio/blob/master/index.js#L42
      );
      sound.play();

      this.setState({
        recordingState: RecordingState.Stopped,
      });

    }
  }

  onRecordedChunk = async (samples: Array<number>) => {
    log.info('[onRecordedChunk]', json({samplesLength: samples.length}));
    // MicStream.stop is unobservably async, so ignore any audio capture after we think we've stopped
    if (this.state.recordingState === RecordingState.Recording) {

      // Buffer chunks (don't setState until flush, to throttle render)
      this._audioSampleChunks.push(samples);

      // Flush if ≥renderBufferSize (throttle render)
      if (_.sum(this._audioSampleChunks.map(x => x.length)) >= this.props.renderBufferSize) {

        // Pop + close over buffered audio samples
        const {_audioSampleChunks} = this;
        this._audioSampleChunks = [];

        // Compute spectros on this thread, not setState thread
        //  - TODO Batch size?
        const spectroChunksImageProps = await Promise.all(_audioSampleChunks.map(this.spectroChunksImage));
        // const spectroChunksImageProps = [await this.spectroChunksImage(_.flatten(_audioSampleChunks))];

        // setState + render
        this.setState((state, props) => ({
          spectroChunksImageProps: (
            [...state.spectroChunksImageProps, ...spectroChunksImageProps]
            .slice(-this.spectroChunksPerScreenWidth) // Trim spectro chunks for O(1) mem usage
          ),
        }));

      }

    }
  }

  spectroChunksImage = async (chunk: Array<number>): Promise<SpectroChunksImageProps> => {
    log.info('[spectroChunksImage]', json({chunkLength: chunk.length}));

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

    return {
      source: {uri: `file://${pngPath}`},
      style: {width: w_img, height: h_img}, // For file:// uris, else image doesn't show
    };

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
