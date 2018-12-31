import base64js from 'base64-js';
import concatTypedArray from 'concat-typed-array';
import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// import Jimp from 'jimp'; // XXX Unused
import { Location } from 'history';
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
import {
  F_Preds, matchRecordPathParams, ModelsSearch, Rec, recordPathParamsFromLocation, SourceId,
} from '../datatypes';
import { DB } from '../db';
import { debug_print, Log, logErrors, logErrorsAsync, puts, rich } from '../log';
import { NativeSearch } from '../native/Search';
import { ImageFile, NativeSpectro, NativeSpectroStats } from '../native/Spectro';
import { Go } from '../router';
import { SettingsWrites } from '../settings';
import Sound from '../sound';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, Styles } from '../styles';
import {
  basename, catchTry, catchTryAsync, chance, deepEqual, Dim, ensureParentDir, ExpWeightedMean, ExpWeightedRate,
  finallyAsync, global, into, json, match, matchNil, matchNull, matchUndefined, pretty, round, safePath,
  shallowDiffPropsState, timed, Timer, tryElse, tryElseAsync, yaml, yamlPretty,
} from '../utils';

const log = new Log('RecordScreen');

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
  modelsSearch: ModelsSearch;
  location: Location;
  go: Go;
  // Settings
  settings: SettingsWrites;
  db: DB;
  showDebug: boolean;
  // RecordScreen
  f_bins: number;
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  refreshRate: number;
  doneSpectroChunkWidth: number;
  spectroChunkLimit: number;
}

interface State {
  showMoreDebug: boolean;
  recordingState: RecordingState;
  spectroScale: number;
  follow: boolean;
  denoised: boolean;
  // Recording (in progress)
  spectroChunks: Array<SpectroChunk>;
  nSpectroChunks: number; // Separate from spectroChunks b/c we truncate the latter
  nSamples: number;
  nSpectroWidth: number;
  // Editing (after done recording)
  editRecording: null | EditRecording;
}

interface EditRecording {
  sourceId: string;
  audioPath: string;
  spectros: EditRecordingSpectros;
}

type EditRecordingSpectros = Map<Denoise, EditRecordingSpectro>;
type Denoise = boolean;
interface EditRecordingSpectro {
  single: ImageFile;
  chunked: Array<ImageFile>;
}

type RecordingState = 'loading' | 'stopped' | 'recording' | 'saving';

interface SpectroChunk {
  source: {uri?: string};
  width: number;
  height: number;
  debugTimes?: DebugTimes;
}

type DebugTimes = Array<{k: string, v: number}>; // Array<{k,v}> because swift Dictionary doesn't preserve order

export class RecordScreen extends Component<Props, State> {

  static defaultProps: Partial<Props> = {
  };

  state: State = {
    showMoreDebug: false,
    recordingState: 'stopped',
    // TODO Expose controls like SearchScreen [think through RecordScreen.state vs. Settings vs. SearchScreen.state]
    spectroScale: this.props.f_bins / 80,
    follow: true,
    denoised: true,
    spectroChunks: [],
    nSpectroChunks: 0,
    nSamples: 0,
    nSpectroWidth: 0,
    editRecording: null,
  };

  // Getters for state
  get nSamplesPerImage(): number { return this.props.sampleRate / this.props.refreshRate; }

  // Listeners
  _listeners: Array<EmitterSubscription> = [];

  // Refs
  _scrollViewRef: RefObject<ScrollView> = React.createRef();

  // Throttle update/render if NativeSpectro has produced more spectros than we have consumed
  //  - Measure render vs. spectro lag using NativeSpectro.stats (which is async)
  //  - Store as internal state i/o react state to avoid unnecessary updates (~2x)
  _nativeStats: null | NativeSpectroStats = null
  get lag(): null | number {
    return !this._nativeStats ? null : this._nativeStats.nPathsSent - this.state.nSpectroChunks;
  }
  updateNativeStats = () => {
    // We're updating internal state, so prevent caller from trying to block on our completion
    (async () => {
      // Condition on Recording to avoid spurious failures before NativeSpectro.create()
      if (this.state.recordingState === 'recording') {
        this._nativeStats = await tryElseAsync<null | NativeSpectroStats>(null, NativeSpectro.stats);
      }
    })();
  }

  // Measure rate of render() vs. nominal refreshRate (for showDebug)
  _renderRate = new ExpWeightedRate(.1);

  componentDidMount = async () => {
    log.info('componentDidMount');
    global.RecordScreen = this; // XXX Debug

    // Request mic permissions
    Permissions.request('microphone').then(status => {
      // NOTE Buggy on ios simulator [https://github.com/yonahforst/react-native-permissions/issues/58]
      //  - Recording works, but permissions always return 'undetermined'
      log.info('componentDidMount: Permissions.request: microphone', {status});
    });

    // Register listeners
    this._listeners.push(NativeSpectro.onSpectroFilePath(this.onSpectroFilePath));

    // Show this.props.location
    await this.updateForLocation(null);

  }

  componentWillUnmount = () => {
    log.info('componentWillUnmount');

    // Unregister listeners
    this._listeners.forEach(listener => listener.remove());

  }

  shouldComponentUpdate = (nextProps: Props, nextState: State): boolean => {

    // Throttle update/render if NativeSpectro has produced more spectros than we have consumed
    //  - Not a foolproof approach to keeping the UI responsive, but it seems to be a good first step
    //  - Condition on recordingState to avoid races where we get stuck without being able to update on stopRecording
    if (this.state.recordingState === 'recording') {
      const lag = this.lag; // (Observe once to avoid races)
      if (lag !== null && lag > 0) {
        // log.debug('shouldComponentUpdate', '->false', {lag}); // XXX Debug
        this.updateNativeStats();
        return false;
      }
    }

    // Else mimic PureComponent
    return shallowCompare(this, nextProps, nextState);

  }

  // Component updates in tight loop (spectro refresh)
  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));

    // If we're updating then shouldComponentUpdate didn't call updateNativeStats, so we should do it instead
    //  - Avoid calling updateNativeStats on every call to shouldComponentUpdate since that could be many more
    this.updateNativeStats();

    // If follow mode was just switched off->on, scroll ScollView to bottom
    //  - Other calls handled elsewhere to avoid bottlenecks from calling scrollToEnd() >>1 times per scroll
    if (!prevState.follow && this.state.follow) {
      this.scrollToEnd();
    }

    // Show this.props.location
    await this.updateForLocation(prevProps.location);

  }

  updateForLocation = async (prevLocation: null | Location) => {
    if (this.props.location !== prevLocation) {
      if (this.state.recordingState !== 'stopped') {

        // TODO Is it less surprising to the user if we stop the recording or drop the nav change?
        //  - Simpler to drop the nav change, so that's what we do for now
        log.info('updateForLocation: Skipping, recording in progress', {
          location: this.props.location,
          prevLocation,
          recordingState: this.state.recordingState,
        });

      } else {

        log.info('updateForLocation', {location: this.props.location, prevLocation});

        // Reset state
        //  - TODO Dedupe with startRecording
        this.setState({
          recordingState: 'loading',
          spectroChunks: [],
          nSpectroChunks: 0,
          nSamples: 0,
          nSpectroWidth: 0,
          editRecording: null,
        });
        this._renderRate.reset();

        await matchRecordPathParams(recordPathParamsFromLocation(this.props.location), {
          root: async () => {
            // Show blank record screen
            this.setState({
              recordingState: 'stopped',
            });
          },
          edit: async ({sourceId}) => {
            // Show editRecording for sourceId
            const editRecording = await EditRecording({
              doneSpectroChunkWidth: this.props.doneSpectroChunkWidth,
              sourceId,
              audioPath: Rec.audioPath(await this.props.db.loadRec(sourceId)),
            });
            this.setState({
              recordingState: 'stopped',
              editRecording,
            });
          },
        });

      }
    }
  }

  // TODO Bigger button hitbox: https://stackoverflow.com/questions/50065928/change-button-font-size-on-react-native
  render = () => {
    log.info('render', {lag: this.lag});
    this._renderRate.mark();
    return (
      <View style={[
        Styles.fill,
        Styles.center,
      ]}>

        {/* Loading spinner */}
        {this.state.recordingState === 'loading' && (
          <View style={{
            flex: 1,
            justifyContent: 'center',
          }}>
            <ActivityIndicator size='large' />
          </View>
        )}

        {/* Spectro images */}
        {this.state.recordingState !== 'loading' && (
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
              log.debug('render: onScrollBeginDrag');
              this.setState({follow: false});
            }}
            onContentSizeChange={(width, height) => {
              log.debug('render: onContentSizeChange', {width, height});
              if (this.state.follow) {
                this.scrollToEnd();
              }
            }}
          >
            {/* Condition on !editRecording so that the transition from recording->stop->rendered is gapless */}
            {!this.state.editRecording ? (

              // Recording in progress: streaming spectro chunks
              <WrappedSpectroImages
                spectros={this.state.spectroChunks}
                spectroScale={this.state.spectroScale}
                showDebug={this.props.showDebug}
                showMoreDebug={this.state.showMoreDebug}
              />

            ) : (

              // Done recording: recorded spectro (chunks computed in stopRecording)
              into(this.state.editRecording.spectros.get(this.state.denoised), spectros => spectros && (
                <WrappedSpectroImages
                  spectros={spectros.chunked.map(({path, ...props}) => ({
                    source: {uri: path},
                    ...props,
                  }))}
                  spectroScale={this.state.spectroScale}
                  showDebug={this.props.showDebug}
                  showMoreDebug={this.state.showMoreDebug}
                />
              ))

            )}
          </ScrollView>
        )}

        {/* Debug info */}
        <this.DebugView style={{
          width: '100%',
        }}>
          <this.DebugText>
            recordingState: {this.state.recordingState}
          </this.DebugText>
          <this.DebugText>
            refreshRate: {round(this._renderRate.value, 2)} / {this.props.refreshRate}
          </this.DebugText>
          <this.DebugText>
            sampleRate: {this.props.sampleRate} Hz (/ {this.props.refreshRate} = {round(this.nSamplesPerImage, 1)} samples/img)
          </this.DebugText>
          <this.DebugText>
            doneSpectroChunkWidth: {this.props.doneSpectroChunkWidth} (screen: {Dimensions.get('window').width})
          </this.DebugText>
          <this.DebugText>
            spectroChunkLimit: {this.props.spectroChunkLimit}
          </this.DebugText>
          <this.DebugText>
            audio: {}
            {sprintf('%.1fs', this.state.nSamples / this.props.sampleRate)} {}
            ({Humanize.compactInteger(this.state.nSamples, 2)} samples)
          </this.DebugText>
          <this.DebugText>
            spectro: {}
            {this.state.nSpectroWidth} w × {this.props.f_bins} h ({this.state.spectroChunks.length} images)
          </this.DebugText>
          {this.state.showMoreDebug && (
            <this.DebugText>
              native: {json(this._nativeStats)}
            </this.DebugText>
          )}
        </this.DebugView>

        {/* Controls bar */}
        <ControlsBar
          go={this.props.go}
          showDebug={this.props.showDebug}
          showMoreDebug={this.state.showMoreDebug}
          recordingState={this.state.recordingState}
          follow={this.state.follow}
          denoised={this.state.denoised}
          editRecording={this.state.editRecording}
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

  startRecording = async () => {
    logErrorsAsync('startRecording', async () => { // So we're safe to use as an event handler
      if (this.state.recordingState === 'stopped') {
        log.info('startRecording', {
          sampleRate: this.props.sampleRate,
          channels: this.props.channels,
          bitsPerSample: this.props.bitsPerSample,
        });

        // Reset state
        //  - TODO Dedupe with updateForLocation
        this.setState({
          recordingState: 'recording',
          spectroChunks: [],
          nSpectroChunks: 0,
          nSamples: 0,
          nSpectroWidth: 0,
          editRecording: null,
        });
        this._renderRate.reset();

        // Start recording
        await NativeSpectro.start({
          outputPath:  await this.freshPath('wav'),
          refreshRate: this.props.refreshRate,
        });

      }
    });
  }

  stopRecording = async () => {
    logErrorsAsync('stopRecording', async () => { // So we're safe to use as an event handler
      if (this.state.recordingState === 'recording') {
        log.info('stopRecording');

        // State: recording -> saving
        this.setState({
          recordingState: 'saving',
        });
        this._nativeStats = null; // Else shouldComponentUpdate gets stuck with lag>0

        // Stop recording
        const audioPath = await NativeSpectro.stop();

        // Compute editRecording
        //  - null if audioPath is null [When does this happen? Not on no audio samples]
        log.info('stopRecording: Got', {audioPath});
        var editRecording: null | EditRecording;
        if (audioPath === null) {
          log.info('stopRecording: Noop: No audioPath');
          editRecording = null;
        } else {
          editRecording = await EditRecording({
            doneSpectroChunkWidth: this.props.doneSpectroChunkWidth,
            sourceId: SourceId('user', basename(audioPath)),
            audioPath,
          });
        }

        // State: saving -> stopped
        this.setState({
          recordingState: 'stopped',
          editRecording,
        });

      }
    });
  }

  onSpectroFilePath = async ({spectroFilePath, width, height, nSamples, debugTimes}: {
    spectroFilePath?: string,
    width: number,
    height: number,
    nSamples: number,
    debugTimes: DebugTimes,
  }) => {
    if (this.state.recordingState !== 'recording') {
      log.info('onSpectroFilePath: Skipping', {recordingState: this.state.recordingState});
    } else {
      log.info('onSpectroFilePath', {
        spectroFilePath: _.defaultTo(spectroFilePath, null),
        size: !spectroFilePath ? undefined : (await fs.stat(spectroFilePath)).size,
        width,
        height,
        nSamples,
        debugTimes,
      });
      const spectroChunk: SpectroChunk = {
        source: !spectroFilePath ? {} : {uri: `file://${spectroFilePath}`},
        width,
        height,
        debugTimes,
      };
      this.setState((state, props) => ({
        // Truncate spectroChunks else the record UX gets very slow after a few minutes
        //  - Use spectroChunkLimit=0 to not truncate
        spectroChunks:  [...state.spectroChunks, spectroChunk].slice(-this.props.spectroChunkLimit),
        nSpectroChunks: state.nSpectroChunks + 1, // Separate from spectroChunks b/c we truncate the latter
        nSamples:       state.nSamples       + nSamples,
        nSpectroWidth:  state.nSpectroWidth  + width,
      }));
    }
  }

  freshPath = async (ext: string, dir: string = fs.dirs.DocumentDir): Promise<string> => {
    const subdir = 'user-recs-v0';
    const timestamp = safePath( // Avoid ':' for ios paths
      new Date().toISOString()
      .slice(0, 19)                            // Cosmetic: drop millis/tz
      .replace(/[-:]/g, '').replace(/T/g, '-') // Cosmetic: 'YYYY-MM-DDThh:mm:ss' -> 'YYYYMMDD-hhmmss'
    );
    const hash = chance.hash({length: 8}); // Long enough to be unique across users
    return ensureParentDir(`${dir}/${subdir}/${timestamp}-${hash}.${ext}`);
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

export interface WrappedSpectroImagesProps {
  spectros: Array<{
    source: {uri?: string},
    width: number,
    height: number,
    debugTimes?: DebugTimes,
  }>;
  spectroScale: number,
  showDebug: boolean;
  showMoreDebug: boolean;
}
export interface WrappedSpectroImagesState {}
export class WrappedSpectroImages extends PureComponent<WrappedSpectroImagesProps, WrappedSpectroImagesState> {

  log = new Log('WrappedSpectroImages');

  static defaultProps = {};
  state = {};

  componentDidMount = async () => {
    // this.log.info('componentDidMount');
  }

  componentWillUnmount = async () => {
    // this.log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: WrappedSpectroImagesProps, prevState: WrappedSpectroImagesState) => {
    // this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  render = () => {
    // this.log.info('render');
    return (
      this.props.spectros.map(({source, width, height, debugTimes}) => (
        <SpectroImage
          key={source.uri}
          source={source}
          spectroScale={this.props.spectroScale}
          width={width}
          height={height}
          debugTimes={debugTimes}
          showDebug={this.props.showDebug}
          showMoreDebug={this.props.showMoreDebug}
        />
      ))
    );
  }

}

// Split out spectro image as component else excessive updates cause render bottleneck
export interface SpectroImageProps {
  source: {uri?: string};
  width: number;
  height: number;
  debugTimes?: DebugTimes;
  spectroScale: number;
  showDebug: boolean;
  showMoreDebug: boolean;
}
export interface SpectroImageState {}
export class SpectroImage extends PureComponent<SpectroImageProps, SpectroImageState> {

  log = new Log('SpectroImage');

  static defaultProps = {};
  state = {};

  componentDidMount = async () => {
    // this.log.info('componentDidMount');
  }

  componentWillUnmount = async () => {
    // this.log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: SpectroImageProps, prevState: SpectroImageState) => {
    // this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  render = () => {
    // this.log.info('render');
    return (
      <View>
        <FastImage
          //  - HACK Using FastImage instead of Image to avoid RCTLog "Reloading image <dataUrl>" killing rndebugger
          //    - https://github.com/facebook/react-native/blob/1151c09/Libraries/Image/RCTImageView.m#L422
          //    - https://github.com/DylanVann/react-native-fast-image
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
  go:             Props["go"];
  showDebug:      Props["showDebug"];
  showMoreDebug:  State["showMoreDebug"];
  recordingState: State["recordingState"];
  follow:         State["follow"];
  denoised:       State["denoised"];
  editRecording:  State["editRecording"];
  setStateProxy:  RecordScreen;
  startRecording: typeof RecordScreen.prototype.startRecording;
  stopRecording:  typeof RecordScreen.prototype.stopRecording;
}
export interface ControlsBarState {}
export class ControlsBar extends PureComponent<ControlsBarProps, ControlsBarState> {

  log = new Log('ControlsBarState');

  static defaultProps = {};
  state = {};

  componentDidMount = async () => {
    this.log.info('componentDidMount');
  }

  componentWillUnmount = async () => {
    this.log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: ControlsBarProps, prevState: ControlsBarState) => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  render = () => {
    this.log.info('render');
    return (
      <View style={styles.bottomControls}>

        {/* Record/stop */}
        {/* - On left side for left-handed phone use [TODO Add setting to toggle left/right] */}
        {match(this.props.recordingState,
          ['loading', () => (
            <RectButton style={styles.bottomControlsButton} onPress={() => {}}>
              <ActivityIndicator size='small' />
            </RectButton>
          )],
          ['stopped', () => (
            <RectButton style={styles.bottomControlsButton} onPress={this.props.startRecording}>
              <FontAwesome5 style={[styles.bottomControlsButtonIcon, {color: Colors.Paired.darkGreen}]}
                name='circle' solid
              />
            </RectButton>
          )],
          ['recording', () => (
            <RectButton style={styles.bottomControlsButton} onPress={this.props.stopRecording}>
              <FontAwesome5 style={[styles.bottomControlsButtonIcon, {color: Colors.Paired.darkRed}]}
                name='stop' solid
              />
            </RectButton>
          )],
          ['saving', () => (
            <RectButton style={styles.bottomControlsButton} onPress={() => {}}>
              <ActivityIndicator size='small' />
            </RectButton>
          )],
        )}

        {/* Search */}
        <RectButton style={styles.bottomControlsButton} onPress={() => {
          if (this.props.editRecording) {
            this.props.go('search', {path: `/rec/${this.props.editRecording.sourceId}`});
          }
        }}>
          <Feather style={[styles.bottomControlsButtonIcon, {
            ...(this.props.editRecording ? {} : {color: iOSColors.gray}),
          }]}
            name='search'
          />
        </RectButton>

        {this.props.showDebug && (
          // Toggle showMoreDebug
          <RectButton style={styles.bottomControlsButton} onPress={() => {
            this.props.setStateProxy.setState((state, props) => ({showMoreDebug: !state.showMoreDebug}))
          }}>
            <Feather name='terminal' style={[styles.bottomControlsButtonIcon, {
              color: this.props.showMoreDebug ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>
        )}
        {!this.props.editRecording ? (
          // Toggle follow
          <RectButton style={styles.bottomControlsButton} onPress={() => {
            this.props.setStateProxy.setState((state, props) => ({follow: !state.follow}))
          }}>
            <Feather name='chevrons-down' style={[styles.bottomControlsButtonIcon, {
              color: this.props.follow ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>
        ) : (
          // Toggle denoised
          <RectButton style={styles.bottomControlsButton} onPress={() => {
            this.props.setStateProxy.setState((state, props) => ({denoised: !state.denoised}))
          }}>
            {/* 'eye' / 'zap' / 'sun' */}
            <Feather name='eye' style={[styles.bottomControlsButtonIcon, {
              color: this.props.denoised ? iOSColors.blue : iOSColors.black,
            }]}/>
          </RectButton>
        )}

      </View>
    );
  }

}

// TODO -> datatypes (along with types for EditRecording)
export async function EditRecording(props: {
  doneSpectroChunkWidth: number,
  sourceId: string,
  audioPath: string,
}): Promise<null | EditRecording> {
  // Render audioPath -> spectros
  //  - null if any value is null (audio samples < nperseg)
  //  - Compute spectros for denoise=true/false (true for preds, false so user can toggle to see pre-denoise)
  //  - Compute spectros for single/chunked (single for preds, chunked for wrapped display)
  var _spectros: EditRecordingSpectros = await log.timedAsync('EditRecording: spectros', async () => new Map(
    await Promise.all([true, false].map(async denoise => {
      // HACK Duped in UserRec.spectroPath
      //  - TODO DocumentDir -> CacheDir to avoid bloat
      //    - [Requires updating all consumers to ensure computed (easy), async (hard)]
      const spectroPath = `${fs.dirs.DocumentDir}/spectros-v0/${safePath(props.sourceId)}.spectros/denoise=${denoise}.png`;
      const single = await NativeSpectro.renderAudioPathToSpectroPath(
        props.audioPath,
        await ensureParentDir(spectroPath),
        {denoise},
      );
      const spectros = !single ? null : {
        single,
        chunked: await NativeSpectro.chunkImageFile(single.path, props.doneSpectroChunkWidth),
      };
      return [denoise, spectros] as [Denoise, EditRecordingSpectro];
    })),
  ));
  const spectros: null | EditRecordingSpectros = _.values(_spectros).some(x => x === null) ? null : _spectros;
  if (spectros === null) {
    log.info('EditRecording: Noop: No spectro (samples < nperseg)');
    return null;
  } else {
    return {
      sourceId: props.sourceId,
      audioPath: props.audioPath,
      spectros,
    };
  }
}

const styles = StyleSheet.create({
  bottomControls: {
    flexDirection: 'row',
    height: 48, // Approx tab bar height (see TabRoutes.TabBarStyle)
  },
  bottomControlsButton: {
    ...Styles.center,
    flex: 1,
    paddingVertical: 5,
    backgroundColor: iOSColors.midGray,
  },
  bottomControlsButtonIcon: {
    ...material.headlineObject,
    color: iOSColors.black
  },
})
