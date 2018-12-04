import base64js from 'base64-js';
import concatTypedArray from 'concat-typed-array';
import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// import Jimp from 'jimp'; // XXX Unused
import Humanize from 'humanize-plus';
import _ from 'lodash';
import React, { Component, PureComponent, RefObject } from 'react';
import shallowCompare from 'react-addons-shallow-compare';
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
import { Spectro, SpectroStats } from '../native/Spectro';
import { Go } from '../router';
import { SettingsWrites } from '../settings';
import Sound from '../sound';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, Styles } from '../styles';
import {
  chance, deepEqual, Dim, ExpWeightedMean, ExpWeightedRate, finallyAsync, global, into, json, match, matchNil, matchNull, matchUndefined,
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
  refreshRate: number;
  spectroHeight: number;
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
}

interface State {
  showMoreDebug: boolean;
  recordingState: RecordingState;
  spectroScale: number;
  follow: boolean;
  denoised: boolean;
  // In-progress recording
  spectroImages: Array<SpectroImage>;
  nSamples: number;
  nSpectroWidth: number;
  // Done recording
  doneRecording: null | DoneRecording;
}

interface DoneRecording {
  audioPath: string;
  spectros: DoneRecordingSpectros;
}

type DoneRecordingSpectros = Map<Denoise, null | DoneRecordingSpectro>;
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

export class RecordScreen extends Component<Props, State> {

  // Many of these are hardcoded to match Bubo/Models.swift:Features (which is in turn hardcoded to match py Features config)
  static defaultProps: Partial<Props> = {
    spectroHeight: Spectro.f_bins,
    sampleRate:    Spectro.sample_rate,
    channels:      1,
    bitsPerSample: 16,
  };

  state: State = {
    showMoreDebug: false,
    recordingState: RecordingState.Stopped,
    // TODO Expose controls like SearchScreen [think through RecordScreen.state vs. Settings vs. SearchScreen.state]
    spectroScale: Spectro.f_bins / 80,
    follow: true,
    denoised: true,
    spectroImages: [],
    nSamples: 0,
    nSpectroWidth: 0,
    doneRecording: null,
  };

  // Getters for state
  get nSamplesPerImage(): number { return this.props.sampleRate / this.props.refreshRate; }

  // Listeners
  _listeners: Array<EmitterSubscription> = [];

  // Refs
  _scrollViewRef: RefObject<ScrollView> = React.createRef();

  // Throttle update/render if Spectro has produced more spectros than we have consumed
  //  - Measure render vs. spectro lag using Spectro.stats (which is async)
  //  - Store as internal state i/o react state to avoid unnecessary updates (~2x)
  _nativeStats: null | SpectroStats = null
  get lag(): null | number {
    return !this._nativeStats ? null : this._nativeStats.nPathsSent - this.state.spectroImages.length;
  }
  updateNativeStats = () => {
    // We're updating internal state, so prevent caller from trying to block on our completion
    (async () => {
      // Condition on Recording to avoid spurious failures before Spectro.setup()
      if (this.state.recordingState === RecordingState.Recording) {
        this._nativeStats = await tryElseAsync<null | SpectroStats>(null, Spectro.stats);
      }
    })();
  }

  // Measure rate of render() vs. nominal refreshRate (for showDebug)
  _renderRate = new ExpWeightedRate(.1);

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

  shouldComponentUpdate = (nextProps: Props, nextState: State): boolean => {

    // Throttle update/render if Spectro has produced more spectros than we have consumed
    //  - Not a foolproof approach to keeping the UI responsive, but it seems to be a good first step
    //  - Condition on recordingState to avoid races where we get stuck without being able to update on stopRecording
    if (this.state.recordingState === RecordingState.Recording) {
      const lag = this.lag; // (Observe once to avoid races)
      if (lag !== null && lag > 0) {
        // log.debug('RecordScreen.shouldComponentUpdate=false', {lag}); // XXX Debug
        this.updateNativeStats();
        return false;
      }
    }

    // Else mimic PureComponent
    return shallowCompare(this, nextProps, nextState);

  }

  // Component updates in tight loop (spectro refresh)
  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    // XXX Debug: avoid logging in tight loop (bottleneck at refreshRate=16)
    log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));

    // If we're updating then shouldComponentUpdate didn't call updateNativeStats, so we should do it instead
    //  - Avoid calling updateNativeStats on every call to shouldComponentUpdate since that could be many more
    this.updateNativeStats();

    // If follow mode was just switched off->on, scroll ScollView to bottom
    //  - Other calls handled elsewhere to avoid bottlenecks from calling scrollToEnd() >>1 times per scroll
    if (!prevState.follow && this.state.follow) {
      this.scrollToEnd();
    }

  }

  // TODO Bigger button hitbox: https://stackoverflow.com/questions/50065928/change-button-font-size-on-react-native
  render = () => {
    log.debug(`${this.constructor.name}.render`, json({lag: this.lag}));
    this._renderRate.mark();
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
            log.debug('onScrollBeginDrag');
            this.setState({follow: false});
          }}
          onContentSizeChange={(width, height) => {
            log.debug('onContentSizeChange', json({width, height}));
            if (this.state.follow) {
              this.scrollToEnd();
            }
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
                showMoreDebug={this.state.showMoreDebug}
              />
            )))

          ) : (

            // Done recording: recorded spectro
            <View>
              {into(this.state.doneRecording.spectros.get(this.state.denoised), spectro => spectro && (
                <SpectroImageComp
                  key={spectro.path}
                  source={{uri: spectro.path}}
                  spectroScale={this.state.spectroScale}
                  width={spectro.dims.width}
                  height={spectro.dims.height}
                  showDebug={this.props.showDebug}
                  showMoreDebug={this.state.showMoreDebug}
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
            refreshRate: {this.props.refreshRate} ({round(this.nSamplesPerImage, 1)} samples/img / {this.props.sampleRate} Hz)
          </this.DebugText>
          <this.DebugText>
            _renderRate: {round(this._renderRate.value, 3)}
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
          {this.state.showMoreDebug && (
            <this.DebugText>
              native: {json(this._nativeStats)}
            </this.DebugText>
          )}
        </this.DebugView>

        {/* Controls bar */}
        <ControlsBar
          showDebug={this.props.showDebug}
          showMoreDebug={this.state.showMoreDebug}
          recordingState={this.state.recordingState}
          follow={this.state.follow}
          denoised={this.state.denoised}
          doneRecording={this.state.doneRecording}
          // NOTE Pass these as bound methods i/o lambdas else ControlsBar will unnecessarily update on every render (and be slow)
          setStateProxy={this} // Had some 'undefined' trouble with passing this.setState, so passing this instead
          startRecording={this.startRecording}
          stopRecording={this.stopRecording}
        />

      </View>
    );
  }

  scrollToEnd = () => {
    log.debug('scrollToEnd');
    this._scrollViewRef.current!.scrollToEnd();
  }

  // Catches and logs errors so callers don't have to (e.g. event handlers)
  startRecording = async () => {
    try {
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
        this._renderRate.reset();

        await Spectro.setup({
          outputFile: this.freshFilename('wav'), // FIXME dir is hardcoded to BaseDirectory.temp (in Spectro.swift)
          sampleRate: this.props.sampleRate,
          bitsPerChannel: this.props.bitsPerSample,
          channelsPerFrame: this.props.channels,
          refreshRate: this.props.refreshRate, // NOTE Only updates on stop/record
          // bufferSize: 2048, // HACK Manually tuned for (22050hz,1ch,16bit)
        });
        await Spectro.start();

      }
    } catch (e) {
      log.error('Error in startRecording', e);
    }
  }

  // Catches and logs errors so callers don't have to (e.g. event handlers)
  stopRecording = async () => {
    try {
      if (this.state.recordingState === RecordingState.Recording) {
        log.info('RecordScreen.stopRecording');

        // Reset state (1/2)
        this.setState({
          recordingState: RecordingState.Saving,
        });
        this._nativeStats = null; // Else shouldComponentUpdate gets stuck with lag>0
        const audioPath = await Spectro.stop();

        log.info(`RecordScreen.stopRecording: Got audioPath[${audioPath}]`);
        var doneRecording: null | DoneRecording;
        if (audioPath === null) {
          doneRecording = null;
        } else {

          // Render audioPath -> spectros
          const spectros: DoneRecordingSpectros = new Map(await Promise.all([true, false].map(async denoise => {
            const path = `${audioPath}-denoise=${denoise}.png`;
            const dims: null | Dim<number> = await Spectro.renderAudioPathToSpectroPath(audioPath, path, {denoise});
            return [denoise, dims && {path, dims}] as [Denoise, null | DoneRecordingSpectro];
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

        // Reset state (2/2)
        this.setState({
          recordingState: RecordingState.Stopped,
          doneRecording,
        });

      }
    } catch (e) {
      log.error('Error in stopRecording', e);
    }
  }

  onSpectroFilePath = async ({spectroFilePath, width, height, nSamples, debugTimes}: {
    spectroFilePath?: string,
    width: number,
    height: number,
    nSamples: number,
    debugTimes: DebugTimes,
  }) => {
    if (this.state.recordingState !== RecordingState.Recording) {
      log.info('RecordScreen.onSpectroFilePath: skipping', json({recordingState: this.state.recordingState}));
    } else {
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
  showMoreDebug: boolean;
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
            marginRight: this.props.showDebug && this.props.showMoreDebug ? 1 : 0, // Separate chunks for debug
          }}
          source={this.props.source}
          // resizeMode='cover'   // Scale both dims to ≥container, maintaining aspect
          // resizeMode='contain' // Scale both dims to ≤container, maintaining aspect
          resizeMode='stretch' // Scale both dims to =container, ignoring aspect
          // resizeMode='center'  // Maintain dims and aspect
        />
        {this.props.showDebug && this.props.showMoreDebug && (
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
  showDebug:      Props["showDebug"];
  showMoreDebug:  State["showMoreDebug"];
  recordingState: State["recordingState"];
  follow:         State["follow"];
  denoised:       State["denoised"];
  doneRecording:  State["doneRecording"];
  setStateProxy:  RecordScreen;
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

        {this.props.showDebug && (
          // Toggle showMoreDebug
          <RectButton style={styles.button} onPress={() => {
            this.props.setStateProxy.setState((state, props) => ({showMoreDebug: !state.showMoreDebug}))
          }}>
            <Feather name='terminal' style={[styles.buttonIcon, {
              color: this.props.showMoreDebug ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>
        )}

        {!this.props.doneRecording ? (
          // Toggle follow
          <RectButton style={styles.button} onPress={() => {
            this.props.setStateProxy.setState((state, props) => ({follow: !state.follow}))
          }}>
            <Feather name='chevrons-down' style={[styles.buttonIcon, {
              color: this.props.follow ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>
        ) : (
          // Toggle denoised
          <RectButton style={styles.button} onPress={() => {
            this.props.setStateProxy.setState((state, props) => ({denoised: !state.denoised}))
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
