import base64js from 'base64-js';
import concatTypedArray from 'concat-typed-array';
import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// import Jimp from 'jimp'; // XXX Unused
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
  chance, deepEqual, Dim, ExpWeightedMean, finallyAsync, global, into, json, match, matchNil, matchNull, matchUndefined,
  pretty, round, shallowDiffPropsState, timed, Timer, tryElse, tryElseAsync, yaml, yamlPretty,
} from '../utils';

global.Spectro = Spectro; // XXX Debug

// XXX Unused
// // Util: wrap `new Jimp` in a promise
// const JimpAsync = (...args: Array<any>): Promise<Jimp> => new Promise((resolve, reject) => {
//   new Jimp(...args, (err: Error | null, img: Jimp) => err ? reject(err) : resolve(img));
// });

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
  spectroHeight: number;
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
}

interface State {
  recordingState: RecordingState;
  refreshRate: number;
  spectroScale: number;
  follow: boolean;
  denoised: boolean;
  // In-progress recording
  spectroImages: Array<SpectroImage>;
  nSamples: number;
  nSpectroWidth: number;
  nativeSpectroStats: null | object; // XXX Debug
  // Done recording
  doneRecording: null | DoneRecording;
}

interface DoneRecording {
  audioPath: string;
  spectros: DoneRecordingSpectros;
}

type DoneRecordingSpectros = Map<Denoise, DoneRecordingSpectro>;
type Denoise = boolean;
interface DoneRecordingSpectro {
  path: string;
  dims: Dim<number>;
}

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

  // Many of these are hardcoded to match Bubo/Models.swift
  static defaultProps: Partial<Props> = {
    spectroHeight: 40, // = swift:Features.f_bins
    sampleRate: 22050, // = swift:Features.sample_rate
    channels: 1,
    bitsPerSample: 16,
  };

  state: State = {
    recordingState: RecordingState.Stopped,
    refreshRate: 16,
    spectroScale: 2, // TODO Expose controls like SearchScreen [think through RecordScreen.state vs. Settings vs. SearchScreen.state]
    follow: true,
    denoised: true,
    spectroImages: [],
    nSamples: 0,
    nSpectroWidth: 0,
    nativeSpectroStats: null,
    doneRecording: null,
  };

  // Getters for state
  get nSamplesPerImage(): number { return this.props.sampleRate / this.state.refreshRate; }

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

    this._listeners.push(Spectro.onSpectroFilePath(this.onSpectroFilePath));

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

    // XXX Debug
    const nativeSpectroStats: object = {
      stats: await tryElseAsync<object | null>(null, () => Spectro.stats()),
    };
    if (!deepEqual(nativeSpectroStats, this.state.nativeSpectroStats)) {
      this.setState({nativeSpectroStats});
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
          {/* Condition on !doneRecording so that the transition from recording->stop->rendered is gapless */}
          {!this.state.doneRecording ? (

            // Recording in progress: streaming spectro chunks
            //  - HACK Using FastImage instead of Image to avoid RCTLog "Reloading image <dataUrl>" killing my rndebugger...
            //    - https://github.com/facebook/react-native/blob/1151c09/Libraries/Image/RCTImageView.m#L422
            //    - https://github.com/DylanVann/react-native-fast-image
            (this.state.spectroImages.map(({source, width, height, debugTimes}, index) => (
              <SpectroImageComp
                key={index} // (Use source.uri if index causes trouble)
                // key={this.props.source.uri}
                source={source}
                spectroScale={this.state.spectroScale}
                width={width}
                height={height}
                debugTimes={debugTimes}
                showDebug={this.props.showDebug}
              />
            )))

          ) : (

            // Done recording: recorded spectro
            <View>
              {into(this.state.doneRecording.spectros.get(this.state.denoised)!, spectro => (
                <SpectroImageComp
                  key={spectro.path}
                  source={{uri: spectro.path}}
                  spectroScale={this.state.spectroScale}
                  width={spectro.dims.width}
                  height={spectro.dims.height}
                  showDebug={this.props.showDebug}
                />
              ))}
            </View>

          )}
        </ScrollView>

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
            {this.state.nSpectroWidth} w × {this.props.spectroHeight} h ({this.state.spectroImages.length} images)
          </this.DebugText>
          <this.DebugText>
            Spectro: {json(this.state.nativeSpectroStats)}
          </this.DebugText>
        </this.DebugView>

        {/* Controls bar */}
        <ControlsBar
          recordingState={this.state.recordingState}
          refreshRate={this.state.refreshRate}
          follow={this.state.follow}
          denoised={this.state.denoised}
          doneRecording={this.state.doneRecording}
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
      }));

      // Reset recordingState
      this.setState({
        recordingState: RecordingState.Recording,
        spectroImages: [],
        nSamples: 0,
        nSpectroWidth: 0,
        doneRecording: null,
      });

      await Spectro.setup({
        outputFile: this.freshFilename('wav'), // FIXME dir is hardcoded to BaseDirectory.temp (in Spectro.swift)
        sampleRate: this.props.sampleRate,
        bitsPerChannel: this.props.bitsPerSample,
        channelsPerFrame: this.props.channels,
        refreshRate: this.state.refreshRate, // NOTE Only updates on stop/record
        // bufferSize: 2048, // HACK Manually tuned for (22050hz,1ch,16bit)
      });
      await Spectro.start();

    }
  }

  stopRecording = async () => {
    if (this.state.recordingState === RecordingState.Recording) {
      log.info('RecordScreen.stopRecording');

      this.setState({
        recordingState: RecordingState.Saving,
      });
      const audioPath = await Spectro.stop();

      log.info(`RecordScreen.stopRecording: Got audioPath[${audioPath}]`);
      var doneRecording: null | DoneRecording;
      if (audioPath === null) {
        doneRecording = null;
      } else {

        // Render audioPath -> spectros
        const spectros = new Map(await Promise.all([true, false].map(async denoise => {
          const path = `${audioPath}-denoise=${denoise}.png`;
          const dims = await Spectro.renderAudioPathToSpectroPath(audioPath, path, {denoise});
          return [denoise, {path, dims}] as [Denoise, DoneRecordingSpectro];
        })));
        doneRecording = {
          audioPath,
          spectros,
        };

        // TODO Add tap to play for recorded rec
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
    const spectroImage: SpectroImage = {
      source: !spectroFilePath ? {} : {uri: `file://${spectroFilePath}`},
      width,
      height,
      debugTimes,
    };
    this.setState((state, props) => ({
      spectroImages: [...state.spectroImages, spectroImage],
      nSamples:      state.nSamples      + nSamples,
      nSpectroWidth: state.nSpectroWidth + width,
    }));
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
  spectroScale: number;
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
            width:  this.props.spectroScale * this.props.width,
            height: this.props.spectroScale * this.props.height,
            marginBottom: 1,
            marginRight: this.props.showDebug ? 1 : 0, // Separate chunks for showDebug
          }}
          source={this.props.source}
          // resizeMode='cover'   // Scale both dims to ≥container, maintaining aspect
          // resizeMode='contain' // Scale both dims to ≤container, maintaining aspect
          resizeMode='stretch' // Scale both dims to =container, ignoring aspect
          // resizeMode='center'  // Maintain dims and aspect
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
  denoised:       State["denoised"];
  doneRecording:  State["doneRecording"];
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

        {!this.props.doneRecording ? (
          // Toggle follow
          <RectButton style={styles.button} onPress={() => {
            this.props.parentSetState((state, props) => ({follow: !state.follow}))
          }}>
            <Feather name='chevrons-down' style={[styles.buttonIcon, {
              color: this.props.follow ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>
        ) : (
          // Toggle denoised
          <RectButton style={styles.button} onPress={() => {
            this.props.parentSetState((state, props) => ({denoised: !state.denoised}))
          }}>
            {/* 'eye' / 'zap' / 'sun' */}
            <Feather name='eye' style={[styles.buttonIcon, {
              color: this.props.denoised ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>
        )}

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
