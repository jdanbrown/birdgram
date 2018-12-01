import base64js from 'base64-js';
import concatTypedArray from 'concat-typed-array';
import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// console.warn('XXX: RecordScreen: import jimp ...'); // XXX Debug
import Jimp from 'jimp';
// console.warn('XXX: RecordScreen: import jimp: done'); // XXX Debug
import Humanize from 'humanize-plus';
import _ from 'lodash';
import React, { PureComponent, RefObject } from 'react';
import {
  ActivityIndicator, Animated, Button, Dimensions, EmitterSubscription, Image, ImageStyle, Platform, ScrollView, Text,
  TextProps, View, ViewProps,
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
import { Spectro } from '../native/Spectro';
import { Go } from '../router';
import { SettingsWrites } from '../settings';
import Sound from '../sound';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, Styles } from '../styles';
import {
  chance, deepEqual, Dim, ExpWeightedMean, finallyAsync, global, json, match, matchNil, matchNull, matchUndefined,
  pretty, round, shallowDiffPropsState, timed, Timer, tryElse, tryElseAsync, yaml, yamlPretty,
} from '../utils';

global.Spectro = Spectro; // XXX Debug

// Util: wrap `new Jimp` in a promise
const JimpAsync = (...args: Array<any>): Promise<Jimp> => new Promise((resolve, reject) => {
  new Jimp(...args, (err: Error | null, img: Jimp) => err ? reject(err) : resolve(img));
});

// "Samples" means "audio samples" throughout
export type Samples = Int16Array; // HACK Assumes bitsPerSample=16
export const Samples = Int16Array;
export function concatSamples(args: Samples[]): Samples { return concatTypedArray(Samples, ...args); }
global.concatTypedArray = concatTypedArray; // XXX Debug
global.concatSamples = concatSamples; // XXX Debug

export interface Props {
  // App globals
  go: Go;
  // Settings
  settings: SettingsWrites;
  showDebug: boolean;
  // RecordScreen
  library: Library;
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  spectroHeight: number;
  nMels: number;
}

interface State {
  recordingState: RecordingState;
  refreshRate: number;
  follow: boolean;
  // In-progress recording
  spectroImages: Array<SpectroImage>;
  nSamples: number;
  nSpectroWidth: number;
  nativeSpectro: null | object; // TODO(swift_spectro)
  // Done recording
  doneRecording: null | DoneRecording;
}

interface DoneRecording {
  audioPath: string;
  spectros: Array<{
    path: string,
    dims: Dim<number>,
  }>;
}

type Library = 'MicStream' | 'AudioRecord' | 'Spectro';

enum RecordingState {
  Stopped = 'Stopped',
  Recording = 'Recording',
  Saving = 'Saving',
}

interface SpectroImage {
  source: {uri?: string};
  width: number;
  height: number;
  debugTimes?: DebugTimes;
}

type DebugTimes = Array<{k: string, v: number}>; // Array<{k,v}> because swift Dictionary doesn't preserve order

export class RecordScreen extends PureComponent<Props, State> {

  static defaultProps: Partial<Props> = {
    // library: 'MicStream', // XXX(write_audio)
    // library: 'AudioRecord', // TODO(write_audio)
    library: 'Spectro', // TODO(swift_spectro)
    sampleRate: 22050, // TODO(swift_spectro) Stuck having to work with 44K i/o 22K (see comments in Spectro.swift)
    channels: 1,
    bitsPerSample: 16,
    spectroHeight: 128, // TODO(mel_spectro) Settings?
    nMels: 128, // Should match spectroHeight for best results (refresh rate + no image smoothing)
  };

  state: State = {
    recordingState: RecordingState.Stopped,
    refreshRate: 4, // XXX Good for debug (avoids overloading react refresh)
    // refreshRate: 16, // TODO Good for prod
    follow: true,
    spectroImages: [],
    nSamples: 0,
    nSpectroWidth: 0,
    nativeSpectro: null,
    doneRecording: null,
  };

  // Getters for state
  get nSamplesPerImage(): number { return this.props.sampleRate / this.state.refreshRate; }

  // Recorded samples -> reshaped samples -> spectro images
  //  - Private attrs instead of state to avoid excessive render calls
  _samplesChunks:  Array<Samples> = [];
  _partialRechunk: Array<Samples> = [];

  // Precompute spectro color table for fast lookup in the tight loop in spectroImageFromSamples
  _magmaTable: Array<{r: number, g: number, b: number}> = (
    _.range(256).map(x => color(interpolateMagma(x / 256)) as RGBColor)
  );

  // Listeners
  _listeners: Array<EmitterSubscription> = [];

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


    match(this.props.library,
      ['MicStream', () => {
        // TODO(mel_spectro): Needs update for Buffer -> Int16Array
        // // Register callbacks
        // this._listeners.push(MicStream.addListener(samples => {
        //   this.onSamplesChunk(Buffer.from(samples));
        // }));
      }],
      ['Spectro', () => {
        // TODO(swift_spectro)

        // this._listeners.push(Spectro.onAudioChunk((data: string) => {
        //   // Convert base64:str -> bytes:Uint8Array -> samples:Int16Array (int16 to match bitsPerSample=16)
        //   if (this.props.bitsPerSample !== 16) throw 'Samples=Int16Array assumes bitsPerSample=16'; // HACK
        //   this.onSamplesChunk(new Samples(base64js.toByteArray(data).buffer));
        // }));

        this._listeners.push(Spectro.onSpectroFilePath(this.onSpectroFilePath));

      }],
    );

  }

  componentWillUnmount = () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
    // Unregisterd callbacks
    this._listeners.forEach(listener => listener.remove());
  }

  // Component updates in tight loop (spectro refresh)
  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    // Avoid logging in tight loop (bottleneck at refreshRate=16)
    // log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));

    if (this.state.follow) {
      // (Not a bottleneck at refreshRate=16)
      this._scrollViewRef.current!.scrollToEnd();
    }

    // TODO(swift_spectro)
    //  - TODO Maybe a bottleneck at refreshRate=16 [profile again and decide]
    const nativeSpectro: object = {
      stats: await tryElseAsync<object | null>(null, () => Spectro.stats()),
    };
    if (!deepEqual(nativeSpectro, this.state.nativeSpectro)) {
      this.setState({nativeSpectro});
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
            (this.state.spectroImages.map(({source, width, height, debugTimes}, index) => (
              <SpectroImageComp
                key={index} // (Use source.uri if index causes trouble)
                // key={this.props.source.uri}
                source={source}
                width={width}
                height={height}
                debugTimes={debugTimes}
                showDebug={this.props.showDebug}
              />
            )))
          }
        </ScrollView>

        {/* Done recording */}
        {this.state.doneRecording && (
          // TODO ScrollView for horizontal scroll (and pinch to zoom, and tap to play, ...)
          <View>
            {this.state.doneRecording.spectros.map(spectro => (
              <Animated.Image
                key={spectro.path}
                style={{
                  width: Dimensions.get('window').width, // HACK Want parent width (and '100%' seems to not work)
                  // width:  spectro.dims.width,
                  height: spectro.dims.height,
                }}
                // resizeMode='cover'   // Scale both dims to ≥container, maintaining aspect
                resizeMode='contain' // Scale both dims to ≤container, maintaining aspect
                // resizeMode='stretch' // Scale both dims to =container, ignoring aspect
                // resizeMode='center'  // Maintain dims and aspect
                source={{uri: spectro.path}}
              />
            ))}
          </View>
        )}

        {/* Debug info */}
        <this.DebugView style={{
          width: '100%',
        }}>
          <this.DebugText>
            Spectro: {json(this.state.nativeSpectro)}
          </this.DebugText>
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

        {/* Controls bar */}
        <ControlsBar
          recordingState={this.state.recordingState}
          refreshRate={this.state.refreshRate}
          follow={this.state.follow}
          parentSetState={f => this.setState(f)}
          startRecording={this.startRecording}
          stopRecording={this.stopRecording}
        />

      </View>
    );
  }

  startRecording = async () => {
    if (this.state.recordingState === RecordingState.Stopped) {
      log.info('RecordScreen.startRecording', yaml({
        sampleRate: this.props.sampleRate,
        channels: this.props.channels,
        bitsPerSample: this.props.bitsPerSample,
        library: this.props.library,
      }));

      // Update recordingState + reset audio chunks
      this._samplesChunks = [];
      this.setState({
        recordingState: RecordingState.Recording,
        spectroImages: [],
        nSamples: 0,
        nSpectroWidth: 0,
        doneRecording: null,
      });

      await match(this.props.library,
        ['MicStream', async () => {
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
            channelsPerFrame: 1,
            // Compute a bufferSize that will fire ~refreshRate buffers per sec
            //  - Hardcode `/2` here to counteract the `*2` in MicStream...
            //  - https://github.com/chadsmith/react-native-microphone-stream/blob/4cca1e7/ios/MicrophoneStream.m#L35
            bufferSize: Math.floor(sampleRate / refreshRate / 2),
          });

          // Start recording
          MicStream.start(); // (Async with no signal of success/failure)

        }],
        ['AudioRecord', async () => {
          // TODO(write_audio)

          AudioRecord.init({
            sampleRate: this.props.sampleRate,
            channels: this.props.channels,
            bitsPerSample: this.props.bitsPerSample,
            wavFile: this.freshFilename('wav'), // FIXME dir is hardcoded to fs.dirs.DocumentDir (in RNAudioRecord.m)
          });
          AudioRecord.start();

          AudioRecord.on('data', (data: string) => {
            // Convert base64:str -> bytes:Uint8Array -> samples:Int16Array (int16 to match bitsPerSample=16)
            if (this.props.bitsPerSample !== 16) throw 'Samples=Int16Array assumes bitsPerSample=16'; // HACK
            this.onSamplesChunk(new Samples(base64js.toByteArray(data).buffer));
          });

        }],
        ['Spectro', async () => {
          // TODO(swift_spectro)

          await Spectro.setup({
            outputFile: this.freshFilename('wav'), // FIXME dir is hardcoded to BaseDirectory.temp (in Spectro.swift)
            sampleRate: this.props.sampleRate,
            bitsPerChannel: this.props.bitsPerSample,
            channelsPerFrame: this.props.channels,
            refreshRate: this.state.refreshRate, // HACK Only updates on stop/record
            // bufferSize: 2048, // HACK Manually tuned for (22050hz,1ch,16bit)
          });
          await Spectro.start();

        }],
      );

    }
  }

  stopRecording = async () => {
    if (this.state.recordingState === RecordingState.Recording) {
      log.info('RecordScreen.stopRecording', yaml({library: this.props.library}));

      this.setState({
        recordingState: RecordingState.Saving,
      });

      const audioPath: string | null = await match(this.props.library,
        ['MicStream', async () => {

          // Stop recording
          log.debug('RecordScreen.stopRecording: Stopping mic');
          MicStream.stop(); // (Async with no signal of success/failure)

          // TODO(write_audio)

          // Encode audio samples as wav data
          //  - https://github.com/rochars/wavefile#create-wave-files-from-scratch
          //  - https://github.com/rochars/wavefile#the-wavefile-methods
          let samples = Array.from(concatSamples(this._samplesChunks));
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
          const audioPath = `${fs.dirs.CacheDir}/${this.freshFilename('wav')}`;
          // await fs.createFile(audioPath, samples as unknown as string, 'ascii'); // HACK Ignore bad type
          await fs.createFile(audioPath, Array.from(wav.toBuffer()) as unknown as string, 'ascii'); // HACK Ignore bad type
          const {size} = await fs.stat(audioPath);
          log.debug('RecordScreen.stopRecording: Wrote file', yaml({audioPath, size}));

          // XXX Debug
          // global.samples = samples;
          global.wav = wav;

          return audioPath;

        }],
        ['AudioRecord', async () => {
          // TODO(write_audio)

          return await AudioRecord.stop();

        }],
        ['Spectro', async () => {
          // TODO(swift_spectro)

          return await Spectro.stop();

        }],
      );

      log.info(`RecordScreen.stopRecording: Got audioPath[${audioPath}]`);
      var doneRecording: null | DoneRecording;
      if (audioPath === null) {
        doneRecording = null;
      } else {

        // Render audioPath -> spectros
        const spectros = await Promise.all([true, false].map(async denoise => {
          const path = `${audioPath}-denoise=${denoise}.png`;
          const dims = await Spectro.renderAudioPathToSpectroPath(audioPath, path, {denoise});
          return {path, dims};
        }));
        doneRecording = {
          audioPath,
          spectros,
        };

        // // XXX Debug: Play rec
        // const sound = await Sound.newAsync(audioPath);
        // log.debug('XXX Playing rec', json({duration: sound.getDuration(), filename: sound.getFilename()}))
        // Sound.setActive(true);
        // Sound.setCategory(
        //   'PlayAndRecord',
        //   true, // mixWithOthers
        // );
        // Sound.setMode(
        //   'Default',
        //   // 'Measurement', // XXX? like https://github.com/jsierles/react-native-audio/blob/master/index.js#L42
        // );
        // finallyAsync(sound.playAsync(), async () => {
        //   sound.release();
        //   Sound.setActive(false);
        // });

      }

      this.setState({
        recordingState: RecordingState.Stopped,
        doneRecording,
      });

    }
  }

  // TODO(swift_spectro)
  onSpectroFilePath = async ({spectroFilePath, width, height, nSamples, debugTimes}: {
    spectroFilePath?: string,
    width: number,
    height: number,
    nSamples: number,
    debugTimes: DebugTimes,
  }) => {
    log.info('RecordScreen.onSpectroFilePath', json({
      spectroFilePath: _.defaultTo(spectroFilePath, null),
      size: !spectroFilePath ? undefined : (await fs.stat(spectroFilePath)).size,
      width,
      height,
      nSamples,
      debugTimes,
    }));

    // TODO(debug_slow): refreshRate=32 seems to work just fine in Release build

    const spectroImage: SpectroImage = {
      source: !spectroFilePath ? {} : {uri: `file://${spectroFilePath}`},
      width,
      height: this.props.spectroHeight, // (Resize height->spectroHeight via Image render)
      debugTimes,
    };

    this.setState((state, props) => ({
      spectroImages: [...state.spectroImages, spectroImage],
      nSamples:      state.nSamples      + nSamples,
      nSpectroWidth: state.nSpectroWidth + width,
    }));

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
          chunk = new Samples(0); // Empty
        } else {
          // Consume a prefix of chunk and loop on the rest
          this._partialRechunk.push(chunk.slice(0, need));
          need = 0;
          chunk = chunk.slice(need); // Rest
        }
        if (need === 0) {
          // Produce the next rechunk
          rechunks.push(concatSamples(this._partialRechunk));
          this._partialRechunk = [];
        }
      }

      // TODO Required for MicStream file write (in stopRecording)
      // // TODO Helpful? Currently unused and consuming ram [though empirically not a significant amount]
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
    const debugTimes: {[key: string]: number} = {};

    // TODO Include samples from previous chunks in stft
    //  - Defer until melSpectrogram, so we can couple to the right mel params

    const {sampleRate, nMels} = this.props;

    // Convert int16->float32 for fft lib (no byte reinterp, just element-wise conversion)
    const y = Float32Array.from(samples);

    const spectroType: string = (
      // 'mag'
      'mel'
      // 'mock'
    );
    const melParams = {
      // TODO(mel_spectro)
      sampleRate,
      // nFft: 2048 / 2,
      // nFft: 2048, // Default: 2048 [I think]
      // winLength, // Default: nFft
      // hopLength, // Default: winLength // 4
      nMels, // Default: 128
      power: 2, // Default: 2
      fMin: 0, // Default: 0
      fMax: sampleRate / 2, // Default: sampleRate / 2
    };

    let nFft: number, spectro: Array<Float32Array>;
    if (spectroType === 'mag') {
      // log.debug('magSpectrogram');

      // Compute (mag) spectro from samples
      const [pow, db] = [
        // pow=1 without powerToDb kinda works, but pow=2 with powerToDb works way better
        // 1, false, // Barely
        // 2, false, // Junk
        2, true,  // Decent [~ melSpectrogram]
        // 3, true,  // Good enough (until we melSpectrogram)
      ];
      // (QA'd via tone generator -> MicStream -> magSpectrogram)
      // const _stft = stft(y, {sampleRate});
      const _stft = stft(y, _.clone(melParams)); // (Clone to avoid caller mutation)
      [spectro, nFft] = magSpectrogram(_stft, pow);

      // Linear power -> log power
      if (db) spectro = powerToDb(spectro);

    } else if (spectroType === 'mel') {
      // log.debug('melSpectrogram');

      nFft = _.get(melParams, 'nFft', null);
      const db = true;
      // const db = false;
      // log.debug('melParams', '\n'+yamlPretty(melParams)); // Debug

      spectro = melSpectrogram(y, _.clone(melParams)); // (Clone to avoid caller mutation)
      debugTimes.melSplit = timer.lap();

      // Linear power -> log power
      if (db) spectro = powerToDb(spectro);

    } else if (spectroType === 'mock') {

      nFft = _.get(melParams, 'nFft', null);

      const [w, h] = [20, melParams.nMels];
      spectro = (nj.random([w, h]).tolist() as number[][]).map(x => Float32Array.from(x));

    } else {
      throw `Invalid spectroType[${spectroType}]`;
    }

    // Float32Array[] -> nj.array
    let S = nj.array(spectro);

    // Normalize values to [0,1] for imageRGBA [QUESTION max? max - min?]
    // log.debug('S.min, S.max', S.min(), S.max()); // Debug
    S = S.subtract(S.min());
    S = S.divide(S.max());
    // log.debug('S.min, S.max', S.min(), S.max()); // Debug

    // log.debug('nFft', nFft); // Debug
    // log.debug('S.shape', S.shape); // Debug

    debugTimes.spectro = timer.lap();

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

    debugTimes.rgba = timer.lap();

    // Build image from imageRGBA
    //  - XXX Very slow, kills app after ~5s
    const [w_img, h_img] = [
      w_S,
      this.props.spectroHeight,
    ];

    let source: {uri: string};
    const imageType: string = (
      'jimp'
      // 'mock'
    );
    if (imageType === 'jimp') {

      let img = await JimpAsync({
        data: imageRGBA,
        width: w_S,
        height: h_S,
      })
      img = (img
        // .crop(0, 0, w_img, h_img)
        // Noticeably faster (at spectroHeight=100) to downsize before data url i/o skip jimp.resize and <Image resizeMode='stretch'>
        // .resize(w_img, h_img) // XXX(mel_spectro)
      );

      debugTimes.jimp = timer.lap();

      // Render image (via file)
      //  - Scratch: fs.createFile(pngPath, imageRGBA.toString('base64'), 'base64');
      const dataUrl = await img.getBase64Async('image/png');
      debugTimes.dataUrl = timer.lap();
      const filename = `${new Date().toISOString()}-${chance.hash({length: 8})}.png`;
      const pngPath = `${fs.dirs.CacheDir}/${filename}`;
      const pngBase64 = dataUrl.replace('data:image/png;base64,', '');
      await fs.createFile(pngPath, pngBase64, 'base64');
      debugTimes.file = timer.lap();

      source = {uri: `file://${pngPath}`}; // Image {style:{width,height}} required for file:// uris, else image doesn't show

    } else if (imageType === 'mock') {

      source = {uri: ''};

    } else {
      throw `Invalid imageType[${imageType}]`;
    }

    const time = timer.time();
    const spectroImage: SpectroImage = {
      source,
      width: w_img,
      height: h_img,
      debugTimes: Array.from(_.entries(debugTimes).map(([k, v]) => ({k, v}))),
    };

    // XXX Globals for dev
    Object.assign(global, {
      samples, y, melParams, spectro, S, w_S, h_S, w_img, h_img, imageRGBA, spectroImage,
    });

    log.info('RecordScreen.spectroImageFromSamples', yaml({
      ['samples.length']: samples.length,
      time,
      timePerAudioSec: round(time / (samples.length / (this.props.sampleRate * this.props.channels)), 3),
      nFft,
      ['S.shape']: S.shape,
      ['S.min_max']: [S.min(), S.max()],
      ['wh_img']: [w_img, h_img],
    }));

    return spectroImage;
  }

  freshFilename = (ext: string): string => {
    const prefix = 'rec-v2';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-'); // (Avoid at least ':' chars, for osx)
    return puts(`${prefix}-${timestamp}-${chance.hash({length: 8})}.${ext}`);
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

// Split out spectro image as component else excessive updates cause render bottleneck
//  - TODO Rename SpectroImage -> ? so we can rename SpectroImageComp -> SpectroImage
export interface SpectroImageCompProps {
  source: {uri?: string};
  width: number;
  height: number;
  debugTimes?: DebugTimes;
  showDebug: boolean;
}
export interface SpectroImageCompState {}
export class SpectroImageComp extends PureComponent<SpectroImageCompProps, SpectroImageCompState> {

  static defaultProps = {};
  state = {};

  componentDidMount = async () => {
    // log.info(`${this.constructor.name}.componentDidMount`);
  }

  componentWillUnmount = async () => {
    // log.info(`${this.constructor.name}.componentWillUnmount`);
  }

  componentDidUpdate = async (prevProps: SpectroImageCompProps, prevState: SpectroImageCompState) => {
    // log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));
  }

  render = () => {
    // log.info(`${this.constructor.name}.render`);
    return (
      <View>
        <FastImage
          style={{
            width: this.props.width,
            height: this.props.height,
            marginBottom: 1,
            marginRight: this.props.showDebug ? 1 : 0,
            paddingVertical: 1, backgroundColor: 'red', // XXX(swift_spectro): Debug
          }}
          source={this.props.source}
          // resizeMode='cover'   // Scale both dims to ≥container, maintaining aspect
          // resizeMode='contain' // Scale both dims to ≤container, maintaining aspect
          // resizeMode='stretch' // Scale both dims to =container, ignoring aspect
          resizeMode='center'  // Maintain dims and aspect
        />
        {this.props.showDebug && (
          <this.DebugView style={{flexDirection: 'column', padding: 0, marginRight: 1}}>
            {(this.props.debugTimes || []).map(({k, v}, i) => (
              <this.DebugText key={i} style={{fontSize: 8}}>{k}:{Math.round(v * 1000)}</this.DebugText>
            ))}
          </this.DebugView>
        )}
      </View>
    );
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

// Split out control buttons as component else excessive updates cause render bottleneck
export interface ControlsBarProps {
  recordingState: State["recordingState"];
  refreshRate:    State["refreshRate"];
  follow:         State["follow"];
  parentSetState: typeof RecordScreen.prototype.setState;
  startRecording: typeof RecordScreen.prototype.startRecording;
  stopRecording:  typeof RecordScreen.prototype.stopRecording;
}
export interface ControlsBarState {}
export class ControlsBar extends PureComponent<ControlsBarProps, ControlsBarState> {

  static defaultProps = {};
  state = {};

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);
  }

  componentWillUnmount = async () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
  }

  componentDidUpdate = async (prevProps: ControlsBarProps, prevState: ControlsBarState) => {
    log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));
  }

  render = () => {
    log.info(`${this.constructor.name}.render`);
    return (
      <View style={{
        flexDirection: 'row',
      }}>

        {/* Refresh rate +/– */}
        <RectButton style={[styles.button, {flex: 2/3}]} onPress={() => {
          this.props.parentSetState((state, props) => ({refreshRate: _.clamp(state.refreshRate / 2, 1, 16)}))
        }}>
          <Feather name='minus' style={styles.buttonIcon} />
        </RectButton>
        <RectButton style={[styles.button, {flex: 2/3}]} onPress={() => {}}>
          <Text style={[styles.buttonIcon, material.headline]}>
            {this.props.refreshRate}/s
          </Text>
        </RectButton>
        <RectButton style={[styles.button, {flex: 2/3}]} onPress={() => {
          this.props.parentSetState((state, props) => ({refreshRate: _.clamp(state.refreshRate * 2, 1, 16)}))
        }}>
          <Feather name='plus' style={styles.buttonIcon} />
        </RectButton>

        {/* Toggle follow */}
        <RectButton style={styles.button} onPress={() => {
          this.props.parentSetState((state, props) => ({follow: !state.follow}))
        }}>
          <Feather name='chevrons-down' style={[styles.buttonIcon, {
            color: this.props.follow ? iOSColors.blue : iOSColors.black,
          }]}/>
        </RectButton>

        {/* Record/stop */}
        {match(this.props.recordingState,
          [RecordingState.Stopped, () => (
            <RectButton style={styles.button} onPress={this.props.startRecording}>
              <FontAwesome5 style={[styles.buttonIcon, {color: Colors.Paired.darkGreen}]}
                name='circle' solid
              />
            </RectButton>
          )],
          [RecordingState.Recording, () => (
            <RectButton style={styles.button} onPress={this.props.stopRecording}>
              <FontAwesome5 style={[styles.buttonIcon, {color: Colors.Paired.darkRed}]}
                name='stop' solid
              />
            </RectButton>
          )],
          [RecordingState.Saving, () => (
            <RectButton style={styles.button} onPress={() => {}}>
              <ActivityIndicator size='large' />
            </RectButton>
          )],
        )}

      </View>
    );
  }

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
