import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// console.warn('XXX: RecordScreen: import jimp ...'); // XXX Debug
import Jimp from 'jimp';
// console.warn('XXX: RecordScreen: import jimp: done'); // XXX Debug
import Humanize from 'humanize-plus';
import _ from 'lodash';
import React, { PureComponent, RefObject } from 'react';
import {
  ActivityIndicator, Button, Dimensions, EmitterSubscription, Image, ImageStyle, Platform, ScrollView, Text, TextProps,
  View, ViewProps,
} from 'react-native';
import AudioRecord from 'react-native-audio-record';
import FastImage from 'react-native-fast-image';
import { BaseButton, BorderlessButton, LongPressGestureHandler, RectButton, TapGestureHandler } from 'react-native-gesture-handler';
import MicStream from 'react-native-microphone-stream';
import Permissions from 'react-native-permissions';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
import WaveFile from 'wavefile/dist/wavefile';
const {fs, base64} = RNFB;

import { magSpectrogram, melSpectrogram, powerToDb, stft } from '../../third-party/magenta/music/transcription/audio_utils'
import nj from '../../third-party/numjs/dist/numjs.min';
import * as Colors from '../colors';
import { SourceId } from '../datatypes';
import { log, puts } from '../log';
import { Go } from '../router';
import { SettingsWrites } from '../settings';
import Sound from '../sound';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, Styles } from '../styles';
import {
  chance, ExpWeightedMean, global, json, match, matchNil, matchNull, matchUndefined, pretty, round,
  shallowDiffPropsState, Timer, yaml, yamlPretty,
} from '../utils';

// Util: wrap `new Jimp` in a promise
const JimpAsync = (...args: Array<any>): Promise<Jimp> => new Promise((resolve, reject) => {
  new Jimp(...args, (err: Error | null, img: Jimp) => err ? reject(err) : resolve(img));
});

// "Samples" means "audio samples" throughout
type Samples = Buffer; // Buffer extends Uint8Array

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
  bitsPerSample: number;
  spectroHeight: number;
}

interface State {
  recordingState: RecordingState;
  refreshRate: number;
  follow: boolean;
  spectroImages: Array<SpectroImage>;
  nSamples: number;
  nSpectroWidth: number;
}

enum RecordingState {
  Stopped = 'Stopped',
  Recording = 'Recording',
  Saving = 'Saving',
}

interface SpectroImage {
  source: {uri: string};
  width: number;
  height: number;
}

export class RecordScreen extends React.Component<Props, State> {

  static defaultProps = {
    // library: 'MicStream', // XXX(write_audio)
    library: 'AudioRecord', // TODO(write_audio)
    sampleRate: 22050,
    channels: 1,
    bitsPerSample: 16,
    spectroHeight: 100,
  };

  state: State = {
    recordingState: RecordingState.Stopped,
    refreshRate: 2,
    follow: true,
    spectroImages: [],
    nSamples: 0,
    nSpectroWidth: 0,
  };

  // Getters for state
  get nSamplesPerImage(): number { return this.props.sampleRate / this.state.refreshRate; }

  // Recorded samples -> reshaped samples -> spectro images
  //  - Private attrs instead of state to avoid excessive render calls
  _samplesChunks:  Array<Buffer> = [];
  _partialRechunk: Array<Buffer> = [];

  // Precompute spectro color table for fast lookup in the tight loop in spectroImageFromSamples
  _magmaTable: Array<{r: number, g: number, b: number}> = (
    _.range(256).map(x => color(interpolateMagma(x / 256)) as RGBColor)
  );

  // Listeners
  _listener?: EmitterSubscription;

  // Refs
  _scrollViewRef: RefObject<ScrollView> = React.createRef();

  componentDidMount = () => {
    log.info(`${this.constructor.name}.componentDidMount`);
    global.RecordScreen = this; // XXX Debug

    // Request mic permissions
    Permissions.request('microphone').then(status => {
      // NOTE Buggy on ios simulator [https://github.com/yonahforst/react-native-permissions/issues/58]
      //  - Recording works, but permissions always return 'undetermined'
      log.info(`${this.constructor.name}.componentDidMount Permissions.request: microphone`, status);
    });

    // Register callbacks
    this._listener = MicStream.addListener(samples => {
      this.onSamplesChunk(Buffer.from(samples));
    });

  }

  componentWillUnmount = () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
    // Unregisterd callbacks
    if (this._listener) this._listener.remove();
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));

    if (this.state.follow) {
      this._scrollViewRef.current!.scrollToEnd();
    }

  }

  // TODO Bigger button hitbox: https://stackoverflow.com/questions/50065928/change-button-font-size-on-react-native
  render = () => {
    log.info(`${this.constructor.name}.render`);
    return (
      <View style={[
        Styles.fill,
        Styles.center,
      ]}>

        {/* Spectro images */}
        <ScrollView
          ref={this._scrollViewRef}
          style={{
            flex: 1,
            width: '100%',
          }}
          contentContainerStyle={{
            flexDirection: 'row',
            flexWrap: 'wrap',
            alignContent: 'flex-start',
            alignItems: 'flex-start',
          }}
          onScrollBeginDrag={() => {
            log.debug('onScrollBeginDrag'); // XXX
            this.setState({follow: false});
          }}
        >
          {
            // HACK Using FastImage instead of Image to avoid RCTLog "Reloading image <dataUrl>" killing my rndebugger...
            //  - https://github.com/facebook/react-native/blob/1151c09/Libraries/Image/RCTImageView.m#L422
            //  - https://github.com/DylanVann/react-native-fast-image
            (this.state.spectroImages.map(({source, width, height}, index) => (
              <FastImage
                key={index} // (Use source.uri if index causes trouble)
                style={{
                  width,
                  height,
                  marginBottom: 1,
                  marginRight: this.props.showDebug ? 1 : 0,
                }}
                source={source}
                // resizeMode='stretch' // Handled upstream by a jimp.resize
              />
            )))
          }
        </ScrollView>

        {/* Controls bar */}
        <View style={{
          flexDirection: 'row',
        }}>

          {/* Refresh rate +/– */}
          <RectButton style={[styles.button, {flex: 2/3}]} onPress={() => {
            this.setState((state, props) => ({refreshRate: _.clamp(state.refreshRate - 1, 1, 10)}))
          }}>
            <Feather name='minus' style={styles.buttonIcon} />
          </RectButton>
          <RectButton style={[styles.button, {flex: 2/3}]} onPress={() => {}}>
            <Text style={[styles.buttonIcon, material.headline]}>
              {this.state.refreshRate}/s
            </Text>
          </RectButton>
          <RectButton style={[styles.button, {flex: 2/3}]} onPress={() => {
            this.setState((state, props) => ({refreshRate: _.clamp(state.refreshRate + 1, 1, 10)}))
          }}>
            <Feather name='plus' style={styles.buttonIcon} />
          </RectButton>

          {/* Toggle follow */}
          <RectButton style={styles.button} onPress={() => {
            this.setState((state, props) => ({follow: !state.follow}))
          }}>
            <Feather name='chevrons-down' style={[styles.buttonIcon, {
              color: this.state.follow ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>

          {/* Record/stop */}
          {match(this.state.recordingState,
            [RecordingState.Stopped, (
              <RectButton style={styles.button} onPress={this.startRecording}>
                <FontAwesome5 style={[styles.buttonIcon, {color: Colors.Paired.darkGreen}]}
                  name='circle' solid
                />
              </RectButton>
            )],
            [RecordingState.Recording, (
              <RectButton style={styles.button} onPress={this.stopRecording}>
                <FontAwesome5 style={[styles.buttonIcon, {color: Colors.Paired.darkRed}]}
                  name='stop' solid
                />
              </RectButton>
            )],
            [RecordingState.Saving, (
              <RectButton style={styles.button} onPress={() => {}}>
                <ActivityIndicator size='large' />
              </RectButton>
            )],
          )}

        </View>

        {/* Debug info */}
        <this.DebugView style={{
          width: '100%',
        }}>
          <this.DebugText>
            recordingState: {this.state.recordingState}
          </this.DebugText>
          <this.DebugText>
            refreshRate: {this.state.refreshRate} ({this.nSamplesPerImage} samples per image / {this.props.sampleRate} Hz)
          </this.DebugText>
          <this.DebugText>
            audio: {}
            {sprintf('%.1fs', this.state.nSamples / this.props.sampleRate)} {}
            ({Humanize.compactInteger(this.state.nSamples, 2)} samples)
          </this.DebugText>
          <this.DebugText>
            spectro: {}
            {this.state.nSpectroWidth} w × {this.props.spectroHeight} h (
            {Humanize.compactInteger(this.state.nSpectroWidth * this.props.spectroHeight, 2)} px, {}
            {this.state.spectroImages.length} images)
          </this.DebugText>
          <this.DebugText>
            handoff: {}
            {this.state.spectroImages.length} images / {this._samplesChunks.length} chunks
          </this.DebugText>
        </this.DebugView>

      </View>
    );
  }

  startRecording = async () => {
    if (this.state.recordingState === RecordingState.Stopped) {
      log.info('RecordScreen.startRecording', yaml({
        sampleRate: this.props.sampleRate,
        channels: this.props.channels,
        bitsPerSample: this.props.bitsPerSample,
      }));

      // Update recordingState + reset audio chunks
      this._samplesChunks = [];
      this.setState({
        recordingState: RecordingState.Recording,
        spectroImages: [],
        nSamples: 0,
        nSpectroWidth: 0,
      });

      if (this.props.library === 'MicStream') {
        // https://github.com/chadsmith/react-native-microphone-stream

        // Init mic
        //  - init() here instead of componentDidMount because we have to call it after each stop()
        const refreshRate = 2; // TODO Make updates faster so we can refresh at lower latency
        const {sampleRate} = this.props;
        MicStream.init({
          // Options
          //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L23-L32
          //  - https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
          //  - TODO PR to change hardcoded kAudioFormatULaw -> any mFormatID (e.g. mp4)
          sampleRate, // (QA'd via tone generator -> MicStream -> magSpectrogram)
          bitsPerChannel: this.props.bitsPerSample,
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
          bitsPerSample: this.props.bitsPerSample,
          wavFile: this.freshFilename('recording', 'wav'), // FIXME Hardcoded to fs.dirs.DocumentDir
        });
        AudioRecord.start();

        // TODO(write_audio)
        AudioRecord.on('data', (data: string) => {
          const samples = Buffer.from(data, 'base64'); // Decode base64 string -> audio samples (uint8)
          this.onSamplesChunk(samples);
        });

      }

    }
  }

  stopRecording = async () => {
    if (this.state.recordingState === RecordingState.Recording) {
      log.info('RecordScreen.stopRecording');

      this.setState({
        recordingState: RecordingState.Saving,
      });

      let wavPath;
      if (this.props.library === 'MicStream') {

        // Stop recording
        log.debug('RecordScreen.stopRecording: Stopping mic');
        MicStream.stop(); // (Async with no signal of success/failure)

        // TODO(write_audio)

        // Encode audio samples as wav data
        //  - https://github.com/rochars/wavefile#create-wave-files-from-scratch
        //  - https://github.com/rochars/wavefile#the-wavefile-methods
        let samples = Array.from(Buffer.concat(this._samplesChunks));
        // samples = samples.map(x => x - 128) // XXX Error: "Overflow at input index 14: -1"
        const wav = new WaveFile()
        wav.fromScratch(
          this.props.channels,
          this.props.sampleRate,
          // MicStream records using kAudioFormatULaw
          //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L31
          // '8',
          // '8a',
          '8m', // 8-bit int, mu-Law
          samples,
        );
        wav.fromMuLaw(); // TODO "Decode 8-bit mu-Law as 16-bit"

        // Write wav data to file
        wavPath = `${fs.dirs.CacheDir}/${this.freshFilename('recording', 'wav')}`;
        // await fs.createFile(wavPath, samples as unknown as string, 'ascii'); // HACK Ignore bad type
        await fs.createFile(wavPath, Array.from(wav.toBuffer()) as unknown as string, 'ascii'); // HACK Ignore bad type
        const {size} = await fs.stat(wavPath);
        log.debug('RecordScreen.stopRecording: Wrote file', yaml({wavPath, size}));

        // XXX Debug
        global.samples = samples;
        global.wav = wav;

      } else if (this.props.library === 'AudioRecord') {

        // TODO(write_audio)
        wavPath = await AudioRecord.stop();

      } else {
        throw `Invalid library[${this.props.library}]`; // (Eliminate wavPath: undefined)
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

  onSamplesChunk = async (chunk: Samples) => {
    log.debug('RecordScreen.onSamplesChunk', yaml({['chunk.length']: chunk.length}));
    // MicStream.stop doesn't tell us when it completes, so manually ignore audio capture after we think we've stopped
    if (this.state.recordingState === RecordingState.Recording) {

      // Re-chunk incoming samples to fixed size for rendering
      const rechunks = [];
      while (chunk.length > 0) {
        let need = this.nSamplesPerImage - _.sum(this._partialRechunk.map(x => x.length));
        if (chunk.length <= need) {
          // Consume the rest of chunk (and stop)
          this._partialRechunk.push(chunk);
          need -= chunk.length;
          chunk = Buffer.alloc(0); // Empty
        } else {
          // Consume a prefix of chunk and loop on the rest
          this._partialRechunk.push(chunk.slice(0, need));
          need = 0;
          chunk = chunk.slice(need); // Rest
        }
        if (need === 0) {
          // Produce the next rechunk
          rechunks.push(Buffer.concat(this._partialRechunk));
          this._partialRechunk = [];
        }
      }

      // TODO Helpful? Currently unused and consuming ram [though empirically not a significant amount]
      // // Persist rechunks [before render, which we might need to change to throttle or shed load]
      // for (let rechunk of rechunks) {
      //   this._samplesChunks.push(rechunk);
      // }

      // Render rechunks (heavy)
      for (let rechunk of rechunks) {
        // Samples -> spectro -> image (heavy js)
        //  - Avoid heavy compute inside setState
        const spectroImage = await this.spectroImageFromSamples(rechunk);
        // setState + render (heavy dom)
        this.setState((state, props) => ({
          spectroImages: [...state.spectroImages, spectroImage],
          nSamples:      state.nSamples      + rechunk.length,
          nSpectroWidth: state.nSpectroWidth + spectroImage.width,
        }));
      }

    }
  }

  spectroImageFromSamples = async (samples: Samples): Promise<SpectroImage> => {
    const timer = new Timer();

    // TODO Include samples from previous chunks in stft
    //  - Defer until melSpectrogram, so we can couple to the right mel params

    // Compute (mag) spectro from samples
    const {sampleRate} = this.props;
    const [pow, db] = [
      // pow=1 without powerToDb kinda works, but pow=2 with powerToDb works way better
      // 1, false, // Barely
      // 2, false, // Junk
      // 2, true,  // Decent
      3, true,  // Good enough (until we melSpectrogram)
    ];
    let [spectro, nfft] = magSpectrogram(
      // TODO TODO `new Float32Array` is dubious -- is it converting numbers or casting bytes?
      stft(new Float32Array(samples), {sampleRate}), // (QA'd via tone generator -> MicStream -> magSpectrogram)
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
    // log.debug('S.shape', S.shape);

    // Compute imageRGBA from S
    const [w_S, h_S] = S.shape;
    const imageRGBA = new Buffer(w_S * h_S * 4);
    for (let w = 0; w < w_S; w++) {
      for (let h = 0; h < h_S; h++) {
        const x = S.get(w, -h) as unknown as number; // (Fix bad type: S.get(w,h): Float32Array)
        const c = this._magmaTable[Math.round((1 - x) * 255)]; // Lookup table to avoid slow color() in inner loop
        imageRGBA[0 + 4*(w + w_S*h)] = c.r;
        imageRGBA[1 + 4*(w + w_S*h)] = c.g;
        imageRGBA[2 + 4*(w + w_S*h)] = c.b;
        imageRGBA[3 + 4*(w + w_S*h)] = 255;
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
      // Noticeably faster (at spectroHeight=100) to downsize before data url i/o skip jimp.resize and <Image resizeMode='stretch'>
      .resize(w_img, h_img)
    );

    // Render image (via file)
    //  - Scratch: fs.createFile(pngPath, imageRGBA.toString('base64'), 'base64');
    const dataUrl = await img.getBase64Async('image/png');
    const filename = `${new Date().toISOString()}-${chance.hash({length: 8})}.png`;
    const pngPath = `${fs.dirs.CacheDir}/${filename}`;
    const pngBase64 = dataUrl.replace('data:image/png;base64,', '');
    await fs.createFile(pngPath, pngBase64, 'base64');

    const spectroImage: SpectroImage = {
      source: {uri: `file://${pngPath}`}, // Image {style:{width,height}} required for file:// uris, else image doesn't show
      width: w_img,
      height: h_img,
    };

    // XXX Globals for dev
    Object.assign(global, {
      samples, spectro, S, w_S, h_S, w_img, h_img, imageRGBA, spectroImage,
    });

    const time = timer.time();
    log.info('RecordScreen.spectroImageFromSamples', yaml({
      ['samples.length']: samples.length,
      time,
      timePerAudioSec: round(time / (samples.length / (this.props.sampleRate * this.props.channels)), 3),
      nfft,
      ['S.shape']: S.shape,
      ['S.min_max']: [S.min(), S.max()],
      ['wh_img']: [w_img, h_img],
    }));

    return spectroImage;
  }

  freshFilename = (prefix: string, ext: string): string => {
    return puts(`${prefix}-${new Date().toISOString()}-${chance.hash({length: 8})}.${ext}`);
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
  button: {
    ...Styles.center,
    flex: 1,
    paddingVertical: 15,
    backgroundColor: iOSColors.midGray,
  },
  buttonIcon: {
    ...material.display2Object,
    color: iOSColors.black
  },
})
