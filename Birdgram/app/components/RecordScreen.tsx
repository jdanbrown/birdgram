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
  ActivityIndicator, Alert, Animated, Button, Dimensions, EmitterSubscription, Image, Platform, ScrollView, StyleProp,
  Text, TextProps, View, ViewProps,
} from 'react-native';
import AudioRecord from 'react-native-audio-record';
import FastImage, { ImageStyle } from 'react-native-fast-image';
import * as Gesture from 'react-native-gesture-handler';
import {
  BaseButton, BorderlessButton, LongPressGestureHandler, RectButton, TapGestureHandler,
} from 'react-native-gesture-handler';
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
import { Geo, GeoCoords } from './Geo';
import * as Colors from '../colors';
import {
  DraftEdit, Edit, EditRec, matchRec, matchRecordPathParams, ModelsSearch, Rec, recordPathParamsFromLocation, Source,
  SourceId, UserRec,
} from '../datatypes';
import { DB } from '../db';
import { debug_print, Log, logErrors, logErrorsAsync, puts, rich } from '../log';
import { NativeSearch } from '../native/Search';
import { ImageFile, NativeSpectro, NativeSpectroStats } from '../native/Spectro';
import { Go, Location } from '../router';
import { SettingsWrites } from '../settings';
import Sound from '../sound';
import { StyleSheet } from '../stylesheet';
import { normalizeStyle, Styles } from '../styles';
import {
  assertFalse, basename, catchTry, catchTryAsync, chance, deepEqual, Dim, ensureParentDir, ExpWeightedMean,
  ExpWeightedRate, finallyAsync, global, ifEmpty, Interval, into, json, local, mapNil, mapNull, mapUndefined, match,
  matchNil, matchNull, matchUndefined, pretty, round, setStateAsync, shallowDiffPropsState, timed, Timer, tryElse,
  tryElseAsync, vibrateNormal, yaml, yamlPretty, zipSame,
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
  geo: Geo;
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
  geoWarnIfNoCoords: boolean;
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
  editClipMode: EditClipMode;
  draftEdit: DraftEdit;
}

type RecordingState = 'loading-for-edit' | 'stopped' | 'recording' | 'saving';

interface SpectroChunk {
  source: {uri?: string};
  width: number;
  height: number;
  debugTimes?: DebugTimes;
}

type DebugTimes = Array<{k: string, v: number}>; // Array<{k,v}> because swift Dictionary doesn't preserve order

interface EditRecording {
  rec: Rec;
  spectros: EditRecordingSpectros;
}

type EditRecordingSpectros = Map<Denoise, EditRecordingSpectro>;
type Denoise = boolean;
interface EditRecordingSpectro {
  single:  ImageFile;
  chunked: Array<{
    imageFile:     ImageFile,
    widthInterval: Interval,
    timeInterval:  Interval,
  }>;
}

type EditClipMode = 'off' | 'lo' | 'hi';

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
    editClipMode: 'off',
    draftEdit: {},
  };

  // Getters for props/state
  get nSamplesPerImage():  number  { return this.props.sampleRate / this.props.refreshRate; }
  get draftEditHasEdits(): boolean { return DraftEdit.hasEdits(this.state.draftEdit); }
  get draftEditHasClips(): boolean { return !_.isEmpty(this.state.draftEdit.clips); }

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
      if (![
        'stopped',          // Safe: no recording in progress
        'loading-for-edit', // Safe: no recording in progress [else we drop transitions on go->go races]
      ].includes(this.state.recordingState)) {

        // TODO Is it less surprising to the user if we stop the recording or drop the nav change?
        //  - Simpler to drop the nav change, so that's what we do for now
        log.info('updateForLocation: Skipping, recording in progress', {
          location: this.props.location,
          prevLocation,
          recordingState: this.state.recordingState,
        });

      } else {

        log.info('updateForLocation', rich({location: this.props.location, prevLocation}));

        // Reset state
        //  - TODO Dedupe with startRecording
        this.setState({
          recordingState: 'loading-for-edit',
          spectroChunks: [],
          nSpectroChunks: 0,
          nSamples: 0,
          nSpectroWidth: 0,
          editRecording: null,
          editClipMode: 'off',
          draftEdit: {},
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
            const source = Source.parse(sourceId);
            if (!source) {
              log.warn('updateForLocation: Failed to parse sourceId', rich({sourceId, location: this.props.location}));
            } else {
              const editRecording = await EditRecording({
                rec: await this.props.db.loadRec(source),
                f_bins: this.props.f_bins,
                doneSpectroChunkWidth: this.props.doneSpectroChunkWidth,
              });
              this.setState({
                recordingState: 'stopped',
                editRecording,
              });
            }
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
        {this.state.recordingState === 'loading-for-edit' && (
          <View style={{
            flex: 1,
            justifyContent: 'center',
          }}>
            <ActivityIndicator size='large' />
          </View>
        )}

        {/* Spectro images */}
        {this.state.recordingState !== 'loading-for-edit' && (
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
              into(this.state.editRecording, editRecording => {
                const spectros = editRecording.spectros.get(this.state.denoised);
                return spectros && (
                  <WrappedSpectroImages
                    spectros={spectros.chunked.map(({imageFile: {path, ...props}}) => ({
                      source: {uri: path},
                      ...props,
                    }))}
                    spectroScale={this.state.spectroScale}
                    spectroStyle={this.spectroStyle(spectros)}
                    onSpectroPress={this.onSpectroPress(spectros)}
                    showDebug={this.props.showDebug}
                    showMoreDebug={this.state.showMoreDebug}
                  />
                );
              })

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
          editClipMode={this.state.editClipMode}
          draftEdit={this.state.draftEdit}
          draftEditHasEdits={this.draftEditHasEdits}
          draftEditHasClips={this.draftEditHasClips}
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

        var cancel: boolean = false;

        // Get geo coords for user rec
        const coords = this.props.geo.coords;
        if (coords === null) {
          log.info('startRecording: No coords', {coords, geoWarnIfNoCoords: this.props.geoWarnIfNoCoords});
          if (this.props.geoWarnIfNoCoords) {
            cancel = await new Promise<boolean>((resolve, reject) => {
              Alert.alert('No GPS', [
                "Failed to get GPS coordinates to save with your recording.",
                "Continue recording without GPS metadata?",
                "\n\nThis warning can be disabled in settings.",
              ].join(' '), [
                {style: 'cancel',  text: "Cancel recording",   onPress: () => resolve(true)},
                {style: 'default', text: "Record without GPS", onPress: () => resolve(false)},
              ]);
            });
          }
        }

        if (cancel) {
          log.info('startRecording: Cancelled', {coords, geoWarnIfNoCoords: this.props.geoWarnIfNoCoords});
        } else {

          // Reset state
          //  - TODO Dedupe with updateForLocation
          this.setState({
            recordingState: 'recording',
            spectroChunks: [],
            nSpectroChunks: 0,
            nSamples: 0,
            nSpectroWidth: 0,
            editRecording: null,
            editClipMode: 'off',
            draftEdit: {},
          });
          this._renderRate.reset();

          // Start recording
          await NativeSpectro.start({
            outputPath: await UserRec.newAudioPath('wav', {
              coords,
            }),
            refreshRate: this.props.refreshRate,
          });

        }

      }
    });
  }

  // TODO(stop_after_30s)
  //  - Start 30s timer in startRecording()
  //  - Invalidate timer in stopRecording()
  //  - On timer, stopRecording() and vibrateNormal() for user feedback
  stopRecording = async () => {
    logErrorsAsync('stopRecording', async () => { // So we're safe to use as an event handler
      if (this.state.recordingState === 'recording') {
        log.info('stopRecording');

        // State: recording -> saving
        //  - HACK setStateAsync to avoid races (setState->setState->go) [and can't await go]
        await setStateAsync(this, {
          recordingState: 'saving',
        });
        this._nativeStats = null; // Else shouldComponentUpdate gets stuck with lag>0

        // Stop recording
        const audioPath = await NativeSpectro.stop();

        // Compute sourceId
        //  - null if audioPath is null [When does this happen? Not on no audio samples]
        log.info('stopRecording: Got', {audioPath});
        var sourceId: SourceId | null;
        if (audioPath === null) {
          log.info('stopRecording: Noop: No audioPath');
          sourceId = null;
        } else {
          const source = await UserRec.new(audioPath);
          sourceId = Source.stringify(source);
        }

        // Edit rec via go() else it won't show up in history
        //  - HACK setStateAsync to avoid races (setState->go) [and can't await go]
        await setStateAsync(this, {
          recordingState: 'loading-for-edit', // (Else updateForLocation will noop on 'saving')
        });
        this.props.go('record', {
          path: !sourceId ? '/edit' : `/edit/${sourceId}`,
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

  onSpectroPress = (spectros: EditRecordingSpectro) => (i: number) => async (pointerInside: boolean) => {
    const spectro        = spectros.chunked[i];
    const {editClipMode} = this.state; // Consistent read (outside of setState -- ok)
    if (['lo', 'hi'].includes(editClipMode)) {
      log.debug('onSpectroPress', {
        i,
        pointerInside,
        editClipMode,
        draftEdit: this.state.draftEdit, // (Might observe an outdated value, but maybe bad to log inside setState?)
        timeInterval: spectro.timeInterval,
      });
      this.setState((state, props) => {

        // Update draftEdit.clips for pressed spectro
        //  - TODO(multi_clip): Add UI for multi-clip editing, and handle multiple clips here
        //  - Only a subset of draftEdit.clips are possible (since we don't yet support full multi-clip editing):
        //    - No edit:         {clips: undefined} / {clips: []}
        //    - Include {lo,hi}: {clips: [{time: {lo, hi}}]}
        //    - Exclude {hi,lo}: {clips: [{time: {-Infinity, lo}}, {time: {hi, Infinity}}]}

        // Unpack clips back to a simple {lo,hi} where lo > hi is possible
        var clips = ifEmpty(state.draftEdit.clips || [], () => [{time: Interval.top}]); // Treat "No edit" like "Include {-inf,inf}"
        var {lo, hi} = match<number, {lo: number, hi: number}>(clips.length,
          [1, () => {
            // Unpack lo < hi
            const [{time: {lo, hi}}] = clips;
            return {lo, hi};
          }],
          [2, () => {
            // Unpack lo > hi
            const [{time: {lo: _neginf, hi}}, {time: {lo, hi: _inf}}] = clips;
            if (!(_neginf === -Infinity && _inf === Infinity)) throw `Invalid 2-length clips[${json(clips)}]`;
            return {lo, hi};
          }],
          [match.default, n => {
            // Nothing should be producing this shape yet
            throw `Invalid clips.length[${n}]`;
          }],
        );

        // Update lo or hi, as selected by editClipMode
        if (state.editClipMode === 'lo') lo = spectro.timeInterval.lo;
        if (state.editClipMode === 'hi') hi = spectro.timeInterval.hi;

        // Reconstruct clips from {lo,hi}, depending on lo < hi vs. lo > hi
        //  -
        clips = (
          lo < hi ? [{time: new Interval(lo, hi)}] :                                            // lo < hi -> "Include {lo,hi}"
          lo > hi ? [{time: new Interval(-Infinity, hi)}, {time: new Interval(lo, Infinity)}] : // lo < hi -> "Exclude {hi,lo}"
          []                                                                                    // lo = hi -> reset (weird edge case)
        );

        // Advance editClipMode (lo -> hi -> off)
        //  - Let user manually press the lo/hi buttons when they want to redo
        const editClipMode = match<EditClipMode, EditClipMode>(state.editClipMode,
          ['lo',  () => 'hi'],
          ['hi',  () => 'off'],
        );

        // Return new state
        return {
          editClipMode,
          draftEdit: {
            ...state.draftEdit,
            clips,
          },
        };

      });
    }
  }

  // Method for shallowCompare in SpectroImage.shouldComponentUpdate (else excessive updates)
  spectroStyle = (spectros: EditRecordingSpectro) => (i: number): StyleProp<ImageStyle> => {
    // Fade spectro chunks outside of draftEdit.clips intervals
    const spectro = spectros.chunked[i];
    return !(
      !this.draftEditHasClips ||
      _.some(this.state.draftEdit.clips || [], clip => clip.time.overlaps(spectro.timeInterval))
    ) && {
      opacity: .333, // TODO tintColor [https://github.com/DylanVann/react-native-fast-image/issues/124]
    }
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
  spectroScale: number;
  spectroStyle?: (i: number) => StyleProp<ImageStyle>;
  onSpectroPress?: (i: number) => (pointerInside: boolean) => void;
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
      this.props.spectros.map(({source, width, height, debugTimes}, i) => (
        <SpectroImage
          key={source.uri}
          source={source}
          spectroScale={this.props.spectroScale}
          width={width}
          height={height}
          i={i}
          style={this.props.spectroStyle}
          onPress={this.props.onSpectroPress}
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
  // Take i as prop so that {style,onPress} can shallowCompare in shouldComponentUpdate (else excessive updates)
  i: number;
  style?: (i: number) => StyleProp<ImageStyle>;
  onPress?: (i: number) => (pointerInside: boolean) => void;
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
      <BaseButton onPress={this.props.onPress && this.props.onPress(this.props.i)}>
        <FastImage
          // HACK Using FastImage instead of Image to avoid RCTLog "Reloading image <dataUrl>" killing rndebugger
          //  - https://github.com/facebook/react-native/blob/1151c09/Libraries/Image/RCTImageView.m#L422
          //  - https://github.com/DylanVann/react-native-fast-image
          style={[this.props.style && this.props.style(this.props.i), {
            width:  this.props.spectroScale * this.props.width,
            height: this.props.spectroScale * this.props.height,
            marginBottom: 1,
            marginRight: this.props.showDebug && this.props.showMoreDebug ? 1 : 0, // Separate chunks for debug
          }]}
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
      </BaseButton>
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
  go:                Props["go"];
  showDebug:         Props["showDebug"];
  showMoreDebug:     State["showMoreDebug"];
  recordingState:    State["recordingState"];
  follow:            State["follow"];
  denoised:          State["denoised"];
  editRecording:     State["editRecording"];
  editClipMode:      State["editClipMode"];
  draftEdit:         State["draftEdit"];
  draftEditHasEdits: boolean;
  draftEditHasClips: boolean;
  setStateProxy:     RecordScreen;
  startRecording:    typeof RecordScreen.prototype.startRecording;
  stopRecording:     typeof RecordScreen.prototype.stopRecording;
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
          ['loading-for-edit', () => (
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

        {/* Go to parent rec */}
        {local(() => {
          const parent = mapNil(this.props.editRecording, ({rec}) => matchRec(rec, {
            xc:   rec => null,
            user: rec => null,
            edit: rec => rec.edit.parent,
          }));
          return (
            <RectButton style={styles.bottomControlsButton} onPress={() => {
              if (parent) {
                this.props.go('record', {path: `/edit/${encodeURIComponent(parent)}`});
              }
            }}>
              <Feather style={[styles.bottomControlsButtonIcon, {
                color: !parent ? iOSColors.gray : iOSColors.black,
              }]}
                name='rewind'
              />
            </RectButton>
          );
        })}

        {/* Clip lo/hi */}
        <RectButton style={styles.bottomControlsButton} onPress={() => {
          if (this.props.editRecording) {
            this.props.setStateProxy.setState((state, props) => ({
              editClipMode: match<EditClipMode, EditClipMode>(state.editClipMode,
                ['off', () => 'lo'],
                ['lo',  () => 'off'],
                ['hi',  () => 'lo'],
              ),
            }))
          }
        }}>
          <Feather style={[styles.bottomControlsButtonIcon, {
            color: (
              !this.props.editRecording ? iOSColors.gray :
              this.props.editClipMode === 'lo' ? iOSColors.blue :
              iOSColors.black
            ),
          }]}
            name='scissors'
          />
        </RectButton>
        <RectButton style={styles.bottomControlsButton} onPress={() => {
          if (this.props.editRecording) {
            this.props.setStateProxy.setState((state, props) => ({
              draftEdit: {
                ...state.draftEdit,
                clips: [],
              },
            }));
          }
        }}>
          <Feather style={[styles.bottomControlsButtonIcon, Styles.rotate45, {
            color: !this.props.editRecording || !this.props.draftEditHasClips ? iOSColors.gray : iOSColors.black,
          }]}
            name='maximize-2'
          />
        </RectButton>
        <RectButton style={styles.bottomControlsButton} onPress={() => {
          if (this.props.editRecording) {
            this.props.setStateProxy.setState((state, props) => ({
              editClipMode: match<EditClipMode, EditClipMode>(state.editClipMode,
                ['off', () => 'hi'],
                ['lo',  () => 'hi'],
                ['hi',  () => 'off'],
              ),
            }))
          }
        }}>
          <Feather style={[styles.bottomControlsButtonIcon, Styles.rotate180, {
            color: (
              !this.props.editRecording ? iOSColors.gray :
              this.props.editClipMode === 'hi' ? iOSColors.blue :
              iOSColors.black
            ),
          }]}
            name='scissors'
          />
        </RectButton>

        {this.props.editRecording && this.props.draftEditHasEdits ? (

          // Done editing: save and show edit rec
          <RectButton style={styles.bottomControlsButton} onPress={async () => {
            if (this.props.editRecording) {
              const editSource = await EditRec.new({
                parent:    this.props.editRecording.rec,
                draftEdit: this.props.draftEdit,
              });
              this.props.go('record', {path: `/edit/${encodeURIComponent(Source.stringify(editSource))}`});
            }
          }}>
            <Feather style={styles.bottomControlsButtonIcon}
              name='check'
            />
          </RectButton>

        ) : (

          // Search
          <RectButton style={styles.bottomControlsButton} onPress={() => {
            if (this.props.editRecording) {
              this.props.go('search', {path: `/rec/${encodeURIComponent(this.props.editRecording.rec.source_id)}`});
            }
          }}>
            <Feather style={[styles.bottomControlsButtonIcon, {
              ...(this.props.editRecording ? {} : {color: iOSColors.gray}),
            }]}
              name='search'
            />
          </RectButton>

        )}

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
  rec: Rec,
  f_bins: number,
  doneSpectroChunkWidth: number,
}): Promise<null | EditRecording> {

  // Render audioPath -> spectros
  //  - null if any value is null (audio samples < nperseg)
  //  - Compute spectros for denoise=true/false (true for preds, false so user can toggle to see pre-denoise)
  //  - Compute spectros for single/chunked (single for preds, chunked for wrapped display)
  var _spectros: EditRecordingSpectros = await log.timedAsync('EditRecording: spectros', async () => new Map(
    await Promise.all([true, false].map(async denoise => {

      // Get a writable spectroCachePath for nonstandard f_bins
      //  - TODO Refactor to simplify
      const spectroCachePath = Rec.spectroCachePath(Source.parseOrFail(props.rec.source_id), {
        f_bins: props.f_bins,
        denoise,
      });
      const single = await NativeSpectro.renderAudioPathToSpectroPath(
        Rec.audioPath(props.rec),
        await ensureParentDir(spectroCachePath),
        {
          f_bins: props.f_bins,
          denoise,
        },
      );

      // Compute spectro chunks
      const spectros = !single ? null : {
        single,
        chunked: await local(async () => {
          const imageFiles = await NativeSpectro.chunkImageFile(single.path, props.doneSpectroChunkWidth);

          // Compute width/time intervals
          var cumWidth = 0;
          const widthIntervals = imageFiles.map(({width}) => {
            cumWidth += width;
            return new Interval(cumWidth, cumWidth - width);
          });
          const timeIntervals = widthIntervals.map(({lo, hi}) => {
            return new Interval(
              lo / cumWidth * props.rec.duration_s,
              hi / cumWidth * props.rec.duration_s,
            );
          });

          return _.zip(imageFiles, widthIntervals, timeIntervals).map(([imageFile, widthInterval, timeInterval]) => ({
            imageFile,
            widthInterval,
            timeInterval,
          }));
        }),
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
      rec: props.rec,
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
