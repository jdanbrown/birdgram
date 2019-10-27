import base64js from 'base64-js';
import concatTypedArray from 'concat-typed-array';
import { color, RGBColor } from 'd3-color';
import { interpolateMagma } from 'd3-scale-chromatic';
// import Jimp from 'jimp'; // XXX Unused
import Humanize from 'humanize-plus';
import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import shallowCompare from 'react-addons-shallow-compare';
import {
  ActivityIndicator, Alert, Animated, Button, Dimensions, EmitterSubscription, Image, LayoutChangeEvent,
  LayoutRectangle, Platform, ScrollView, Share, StyleProp, Text, TextProps, View, ViewProps,
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

import { Geo, GeoCoords } from 'app/components/Geo';
import * as Colors from 'app/colors';
import {
  DraftEdit, matchRec, matchRecordPathParams, matchSource, ModelsSearch, Rec, recordPathParamsFromLocation, Source,
  SourceId, UserMetadata, UserRec, UserSource,
} from 'app/datatypes';
import { DB } from 'app/db';
import { debug_print, Log, logErrors, logErrorsAsync, puts, rich, tap } from 'app/log';
import { NativeSearch } from 'app/native/Search';
import { ImageFile, NativeSpectro, NativeSpectroStats } from 'app/native/Spectro';
import { Go, Location, locationKeyIsEqual, locationPathIsEqual } from 'app/router';
import { SettingsWrites } from 'app/settings';
import Sound from 'app/sound';
import { StyleSheet } from 'app/stylesheet';
import { normalizeStyle, Styles } from 'app/styles';
import {
  assertFalse, basename, catchTry, catchTryAsync, chance, Dim, ensureParentDir, ExpWeightedMean, ExpWeightedRate,
  fastIsEqual, finallyAsync, global, ifEmpty, ifNil, ifNull, ifUndefined, Interval, into, json, local, mapMapValues,
  mapNil, mapNull, mapUndefined, match, matchNil, matchNull, matchUndefined, pretty, round, setStateAsync,
  shallowDiffPropsState, timed, Timer, tryElse, tryElseAsync, vibrateNormal, yaml, yamlPretty, zipSame,
} from 'app/utils';
import { magSpectrogram, melSpectrogram, powerToDb, stft } from 'third-party/magenta/music/transcription/audio_utils'
import nj from 'third-party/numjs/dist/numjs.min';

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

// Like ImageFile, but .source.uri i/o .path
export interface ImageSource {
  width:  number;
  height: number;
  source: {uri?: string};
}

//
// RecordScreen
//

export interface Props {
  // App globals
  visible: boolean; // Manual visible/dirty to avoid background updates
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
  dirtyUpdateForLocation: boolean;
  showMoreDebug: boolean;
  recordingState: RecordingState;
  recordingUserMetadata: UserMetadata | null;
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
  editRecordingDerived: null | Map<Denoise, {
    single: ImageSource;
    chunked: Array<ImageSource>;
  }>,
  editClipMode: EditClipMode;
  draftEdit: DraftEdit;
}

type RecordingState = 'loading-for-edit' | 'stopped' | 'recording' | 'saving';

interface SpectroChunk extends ImageSource {
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
    dirtyUpdateForLocation: false,
    showMoreDebug: false,
    recordingState: 'stopped',
    recordingUserMetadata: null,
    // TODO Expose controls like SearchScreen [think through RecordScreen.state vs. Settings vs. SearchScreen.state]
    spectroScale: this.props.f_bins / 80,
    follow: true,
    denoised: true,
    spectroChunks: [],
    nSpectroChunks: 0,
    nSamples: 0,
    nSpectroWidth: 0,
    editRecording: null,
    editRecordingDerived: null,
    editClipMode: 'off',
    draftEdit: {},
  };

  // Getters for props/state
  get nSamplesPerImage():  number  { return this.props.sampleRate / this.props.refreshRate; }
  get draftEditHasEdits(): boolean { return DraftEdit.hasEdits(this.state.draftEdit); }
  get draftEditHasClips(): boolean { return !_.isEmpty(this.state.draftEdit.clips); }

  // State
  _wrappedSpectrosLayout?: LayoutRectangle;

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
    // log.info('shouldComponentUpdate', () => rich(shallowDiffPropsState(nextProps, nextState, this.props, this.state))); // Debug

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
    await this.updateForLocation(prevProps);

  }

  updateForLocation = async (prevProps: null | Props) => {
    const {visible} = this.props;
    const dirty     = this.state.dirtyUpdateForLocation;
    if (
      !(
        // Don't noop if we're visible and dirty (i.e. we just became visible, and some props/state changed in the meantime)
        visible && dirty
      ) && (
        // Noop if location didn't change
        locationPathIsEqual(this.props.location, _.get(prevProps, 'location'))
      )
    ) {
      log.info('updateForLocation: Skipping');
    } else {

      // Manual visible/dirty to avoid background updates (type 2: updateFor*)
      //  - If some props/state changed, mark dirty
      if (!visible && !dirty) this.setState({dirtyUpdateForLocation: true});
      //  - If we just became visible, mark undirty
      if (visible && dirty) this.setState({dirtyUpdateForLocation: false});
      //  - If we aren't visible, noop
      if (!visible) return;

      const prevLocation = _.get(prevProps, 'location');
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
        await log.timedAsync('updateForLocation', async () => {

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
              var editRecording: null | EditRecording; // null if audio file not found
              if (SourceId.isOldStyleEdit(sourceId)) {
                // Treat old-style edit recs (from history) like audio file not found
                editRecording = null;
              } else {
                const source = await log.timedAsync('updateForLocation: Source.load', async () => {
                  return await Source.load(sourceId);
                });
                if (!source) {
                  // Noisy for not found (e.g. user deleted a user rec, or xc dataset changed)
                  // log.warn('updateForLocation: Failed to load sourceId', rich({sourceId, location: this.props.location}));
                  editRecording = null; // Treat like audio file not found (else stuck on loading spinner)
                } else {
                  const rec = await log.timedAsync('updateForLocation: db.loadRec', async () => {
                    return await this.props.db.loadRec(source); // null if audio file not found
                  });
                  editRecording = await log.timedAsync('updateForLocation: EditRecording', async () => {
                    return await mapNull(rec, async rec => await EditRecording({
                      rec,
                      f_bins: this.props.f_bins,
                      doneSpectroChunkWidth: this.props.doneSpectroChunkWidth,
                    }));
                  });
                }
              }
              await log.timedAsync('updateForLocation: setState', async () => {
                // TODO Perf: This takes about as long as renderAudioPathToSpectroPath (in EditRecording)
                //  - Profiling indicates it's because of the num-chunks SpectroImage components
                //  - Now that we crop using css instead of img files, we can hugely cut this down to num-wraps img components
                //  - Blockers
                //    - [x] Global cropping controls (many onPress per chunk SpectroImage -> one (x,y) on WrappedSpectroImages)
                //    - [ ] Complexify opacity overlay (simple opacity per SpectroImage -> complex overlay rects within SpectroImage)
                //    - [ ] Change doneSpectroChunkWidth-many chunks (W/5) -> screen-row-many chunk (W/375) (~75x improvement!)
                this.setState({
                  recordingState: 'stopped',
                  editRecording, // null if audio file not found
                  // To avoid unnecessary updates, include all the stuff derived from editRecording into state
                  editRecordingDerived: mapNull(editRecording, editRecording => (
                    mapMapValues(editRecording!.spectros, ({single, chunked}) => ({
                      single: into(single, ({path, ...props}) => ({
                        source: {uri: path},
                        ...props,
                      })),
                      chunked: chunked.map(({imageFile: {path, ...props}}) => ({
                        source: {uri: path},
                        ...props,
                      })),
                    }))
                  )),
                });
              });
            },
          });

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
              flex:  1,
              width: '100%',
            }}
            contentContainerStyle={{
              // (Moved to child View style)
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
            <TapGestureHandler onHandlerStateChange={this.onWrappedSpectrosHandlerStateChange}>
              <Animated.View
                style={{
                  flexDirection: 'row',
                  flexWrap:      'wrap',
                  alignContent:  'flex-start',
                  alignItems:    'flex-start',
                }}
                onLayout={this.onWrappedSpectrosLayout}
              >
                {/* Condition on !editRecording so that the transition from recording->stop->rendered is gapless */}
                {!this.state.editRecording ? (
                  this.state.recordingState === 'stopped' ? (

                    // editRecording not found (e.g. user deleted a user rec, or xc dataset changed)
                    //  - This also shows on first app launch, so let's do "Start a recording" instead of "Recording not found"
                    //  - TODO Show both by switching on this.props.location.pathname === '/'
                    //    - Requires testing
                    <View style={[Styles.center, Styles.fill, {padding: 30}]}>
                      <Text style={material.subheading}>
                        {/* Recording not found */}
                        Start a recording
                      </Text>
                    </View>

                  ) : (

                    // Recording in progress: streaming spectro chunks
                    <WrappedSpectroImages
                      chunked={this.state.spectroChunks}
                      spectroScale={this.state.spectroScale}
                      // For spectroStyle (passed as top-level props to avoid excessive updates)
                      //  - Unused here, only used for edit rec below ("Done recording")
                      editRecording={this.state.editRecording}
                      denoised={this.state.denoised}
                      draftEditHasClips={this.draftEditHasClips}
                      draftEdit={this.state.draftEdit}
                      // Debug
                      showDebug={this.props.showDebug}
                      showMoreDebug={this.state.showMoreDebug}
                    />

                  )
                ) : (

                  // Done recording: recorded spectro (chunks computed in stopRecording)
                  into(this.state.editRecording, editRecording => {
                    const editRecordingDerived = this.state.editRecordingDerived!; // Defined if editRecording is
                    const spectro = editRecordingDerived.get(this.state.denoised)!;
                    return (
                      <WrappedSpectroImages
                        single={spectro.single}
                        chunked={spectro.chunked}
                        spectroScale={this.state.spectroScale}
                        // For spectroStyle (passed as top-level props to avoid excessive updates)
                        editRecording={this.state.editRecording}
                        denoised={this.state.denoised}
                        draftEditHasClips={this.draftEditHasClips}
                        draftEdit={this.state.draftEdit}
                        // Debug
                        showDebug={this.props.showDebug}
                        showMoreDebug={this.state.showMoreDebug}
                      />
                    );
                  })

                )}
              </Animated.View>
            </TapGestureHandler>
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
          <this.DebugText>
            {yamlPretty({
              editRecording: mapNull(this.state.editRecording, ({rec}) => matchSource<{}>(Rec.source(rec), {
                // HACK Get coords on the debug screen (add more fields later)
                xc:   source => ({coords: null}),
                user: source => ({coords: source.metadata.coords}),
              })),
            })}
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
          geo={this.props.geo}
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
          shareRec={this.shareRec}
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
        //  - Do at start i/o stop so user knows up front whether gps is working, since they have no way to add it later
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

          // Initial UserMetadata, as of record time
          const recordingUserMetadata = await UserMetadata.new({
            created: new Date(),
            uniq:    chance.hash({length: 8}), // Long enough to be unique across users
            edit:    null, // Not an edit of another rec
            coords,
          });

          // Make audioPath
          const source = UserSource.new({
            ext:      'wav',
            metadata: recordingUserMetadata,
          });
          const audioPath = UserRec.audioPath(source);

          // Reset state
          //  - TODO Dedupe with updateForLocation
          this.setState({
            recordingState: 'recording',
            recordingUserMetadata,
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
            outputPath:  await ensureParentDir(audioPath),
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

        // Unpack recording state
        const recordingUserMetadata = ifNull(this.state.recordingUserMetadata, () => {
          throw `Expected non-null recordingUserMetadata while recording`;
        });

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
          const source = await UserRec.new(audioPath, recordingUserMetadata);
          sourceId = Source.stringify(source);
        }

        // Edit rec via go() else it won't show up in history
        //  - HACK setStateAsync to avoid races (setState->go) [and can't await go]
        await setStateAsync(this, {
          recordingState: 'loading-for-edit', // (Else updateForLocation will noop on 'saving')
          recordingUserMetadata: null,
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

  onWrappedSpectrosLayout = async (event: LayoutChangeEvent) => {
    const {nativeEvent: {layout}} = event; // Unpack SyntheticEvent (before async)
    this._wrappedSpectrosLayout = layout;
  }

  onWrappedSpectrosHandlerStateChange = async (event: Gesture.TapGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state, oldState, x, y, absoluteX, absoluteY}} = event; // Unpack SyntheticEvent (before async)
    if (
      oldState === Gesture.State.ACTIVE &&
      state !== Gesture.State.CANCELLED
    ) {
      // log.debug('onWrappedSpectrosHandlerStateChange', {x, y, absoluteX, absoluteY}); // Debug
      await this.onWrappedSpectrosPress(x, y);
    }
  }

  onWrappedSpectrosPress = async (x: number, y: number) => {
    // TODO Propagate event to children if we don't respond to it [no children that care, currently]
    if (this.state.editRecording) {
      const spectros = this.state.editRecording.spectros.get(this.state.denoised);
      if (spectros && spectros.chunked.length > 0) {
        const {width, height} = spectros.chunked[0].imageFile; // HACK Assume all chunks have same width/height
        const layout = this._wrappedSpectrosLayout;
        if (layout) {
          const row  = Math.trunc(y / height);
          const col  = Math.trunc(x / width);
          const cols = Math.trunc(layout.width / width);
          const i    = row * cols + col;
          log.debug('onWrappedSpectrosPress', {layout, width, height, x, y, row, col, cols, i});
          await this.onSpectroPress(spectros)(i)();
        }
      }
    }
  }

  onSpectroPress = (spectros: EditRecordingSpectro) => (i: number) => async () => {
    if (i < spectros.chunked.length) {
      const spectro        = spectros.chunked[i];
      const {editClipMode} = this.state; // Consistent read (outside of setState -- ok)
      log.debug('onSpectroPress', {
        i,
        editClipMode,
        draftEdit: this.state.draftEdit, // (Might observe an outdated value, but maybe bad to log inside setState?)
        timeInterval: spectro.timeInterval,
      });
      if (['lo', 'hi'].includes(editClipMode)) {
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

          // XXX Don't auto-advance the controls, for the common case where you tap multiple times to get the clip right
          // // Advance editClipMode (lo -> hi -> off)
          // //  - Let user manually press the lo/hi buttons when they want to redo
          // const editClipMode = match<EditClipMode, EditClipMode>(state.editClipMode,
          //   ['lo',  () => 'hi'],
          //   ['hi',  () => 'off'],
          // );

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
  }

  shareRec = async (rec: Rec) => {
    // TODO(android): How to share file on android? Docs say url is ios only
    //  - https://facebook.github.io/react-native/docs/share.html
    const content = {
      url:     `file://${Rec.audioPath(rec)}`,
      message: Source.show(Rec.source(rec), {
        species: null, // TODO Add this.props.xc so we can show species for xc recs
        long:    true,
      }),
    };
    log.info('shareRec', {rec, content});
    await Share.share(content);
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

//
// WrappedSpectroImages
//

export interface WrappedSpectroImagesProps {
  single?: ImageSource;
  chunked: Array<ImageSource & {
    debugTimes?: DebugTimes,
  }>;
  spectroScale:      number;
  // For spectroStyle (passed as top-level props to avoid excessive updates)
  editRecording:     State['editRecording'];
  denoised:          State['denoised'];
  draftEditHasClips: RecordScreen['draftEditHasClips'];
  draftEdit:         State['draftEdit'];
  // Debug
  showDebug:         boolean;
  showMoreDebug:     boolean;
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
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  render = () => {
    // this.log.info('render');
    return (
      this.props.chunked.map((chunk, i) => (
        <SpectroImage
          key={`${chunk.source.uri}[i=${i}]`}
          i={i}
          chunk={chunk}
          single={this.props.single}
          style={this.spectroStyle(i)}
          spectroScale={this.props.spectroScale}
          showDebug={this.props.showDebug}
          showMoreDebug={this.props.showMoreDebug}
        />
      ))
    );
  }

  spectroStyle = (i: number): undefined | StyleProp<ImageStyle> => {
    // If edit rec, fade spectro chunks outside of draftEdit.clips intervals
    if (!this.props.editRecording) {
      return undefined;
    } else {
      const spectros = this.props.editRecording.spectros.get(this.props.denoised)!;
      const spectro  = spectros.chunked[i];
      return !(
        !this.props.draftEditHasClips ||
        _.some(this.props.draftEdit.clips || [], clip => clip.time.overlaps(spectro.timeInterval))
      ) && {
        opacity: .333, // TODO tintColor [https://github.com/DylanVann/react-native-fast-image/issues/124]
      }
    }
  }

}

//
// SpectroImage
//

// Split out spectro image as component else excessive updates cause render bottleneck
export interface SpectroImageProps {
  i: number; // Take i as prop so that style(i) can shallowCompare in shouldComponentUpdate (else excessive updates)
  chunk: ImageSource & {
    debugTimes?: DebugTimes;
  };
  single?: ImageSource;
  style?: StyleProp<ImageStyle>;
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
    var x: ReactNode = (

      // TODO Simplify, remove cruft, reduce coupling (EditRecording <-> SpectroImage)
      this.props.single ? (
        // Case: static edit rec
        //  - We have the single static spectro, so reuse it to drop all the cropped chunks (crop via css)
        //  - This is a perf optimization: we used to compute all the separate chunk imgs on save and it was a major bottleneck

        <View style={{
          width:  this.props.spectroScale * this.props.chunk.width,
          height: this.props.spectroScale * this.props.chunk.height,
          marginBottom: 1,
          overflow: 'hidden',
          backgroundColor: 'transparent',
          // marginRight: 1, // XXX Debug: If you want a gap, use margin i/o border/padding (child Image shows through)
        }}>
          {(
            // Crop full spectro image down to the (tiny) slice for this chunk
            //  - Based on: https://github.com/walmartlabs/react-native-cropping/blob/c5b8401/index.js#L56-L66
            //  - Use FastImage i/o Image because Image ooms on long recs (e.g. >40s rec on iphone 8)
            //    - Not clear why, since FastImage slurps each img into ram and Image uses a cache... but it works in practice!
            //    - https://github.com/DylanVann/react-native-fast-image/blob/89c0e2e/ios/FastImage/FFFastImageView.m#L80
            //    - https://github.com/facebook/react-native/blob/62d3409/Libraries/Image/Image.ios.js#L132-L139
            //    - https://github.com/facebook/react-native/blob/62d3409/Libraries/Image/RCTImageView.m#L317-L324
            //    - https://github.com/facebook/react-native/blob/62d3409/Libraries/Image/RCTImageLoader.m#L467-L470
            //    - https://github.com/facebook/react-native/blob/62d3409/Libraries/Image/RCTImageCache.m#L101
            <FastImage
              key={`${this.props.i}/${this.props.chunk.source.uri}`}
              style={[this.props.style, {
                width:    this.props.spectroScale * this.props.single.width,
                height:   this.props.spectroScale * this.props.single.height,
                position: 'absolute',
                left:     -this.props.i * this.props.chunk.width,
              }]}
              source={this.props.chunk.source}
              resizeMode='cover'   // Scale both dims to ≥container, maintaining aspect
              // resizeMode='contain' // Scale both dims to ≤container, maintaining aspect
              // resizeMode='stretch' // Scale both dims to =container, ignoring aspect
              // resizeMode='center'  // Maintain dims and aspect
            />
          )}
        </View>

      ) : (
        // Case: streaming spectro
        //  - We don't have a single static spectro, so we have to draw all the separate chunk imgs

        <FastImage
          // HACK Using FastImage instead of Image to avoid RCTLog "Reloading image <dataUrl>" killing rndebugger
          //  - https://github.com/facebook/react-native/blob/1151c09/Libraries/Image/RCTImageView.m#L422
          //  - https://github.com/DylanVann/react-native-fast-image
          style={[this.props.style, {
            width:  this.props.spectroScale * this.props.chunk.width,
            height: this.props.spectroScale * this.props.chunk.height,
            marginBottom: 1,
            marginRight: this.props.showDebug && this.props.showMoreDebug ? 1 : 0, // Separate chunks for debug
          }]}
          source={this.props.chunk.source}
          // resizeMode='cover'   // Scale both dims to ≥container, maintaining aspect
          // resizeMode='contain' // Scale both dims to ≤container, maintaining aspect
          resizeMode='stretch' // Scale both dims to =container, ignoring aspect
          // resizeMode='center'  // Maintain dims and aspect
        />

      )
    );

    if (this.props.showDebug && this.props.showMoreDebug) {
      x = (
        <View>
          {x}
          <this.DebugView style={{flexDirection: 'column', padding: 0, marginRight: 1}}>
            {(this.props.chunk.debugTimes || []).map(({k, v}, i) => (
              <this.DebugText key={i} style={{fontSize: 8}}>{k}:{Math.round(v * 1000)}</this.DebugText>
            ))}
          </this.DebugView>
        </View>
      );
    }

    return x;
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

//
// ControlsBar
//

// Split out control buttons as component else excessive updates cause render bottleneck
export interface ControlsBarProps {
  go:                Props["go"];
  geo:               Props["geo"];
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
  shareRec:          typeof RecordScreen.prototype.shareRec;
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

        {/* Go to parent rec */}
        {local(() => {
          const parent: null | Source = mapNull(this.props.editRecording, ({rec}) => matchRec(rec, {
            xc:   rec => null,
            user: rec => mapNull(rec.source.metadata.edit, edit => edit.parent),
          }));
          return (
            <RectButton style={styles.bottomControlsButton} onPress={() => {
              if (parent) {
                this.props.go('record', {path: `/edit/${encodeURIComponent(Source.stringify(parent))}`});
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
              const source = await UserRec.newFromEdit({
                parent:    this.props.editRecording.rec,
                draftEdit: this.props.draftEdit,
              });
              this.props.go('record', {path: `/edit/${encodeURIComponent(Source.stringify(source))}`});
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

        {/* Share */}
        <RectButton style={styles.bottomControlsButton} onPress={() => {
          if (this.props.editRecording) {
            this.props.shareRec(this.props.editRecording.rec);
          }
        }}>
          <Feather style={[styles.bottomControlsButtonIcon, {
            color: !this.props.editRecording || this.props.draftEditHasEdits ? iOSColors.gray : iOSColors.black,
          }]}
            name='share'
          />
        </RectButton>

        {/* XXX Debug gps */}
        {/* {this.props.showDebug && (
          // Manually refresh geo.coords
          <RectButton style={styles.bottomControlsButton} onPress={async () => {
            log.debug('[geo] getCurrentCoords...');
            const coords = await this.props.geo.getCurrentCoords();
            log.debug('[geo] getCurrentCoords', pretty({coords}));
          }}>
            <Feather name='at-sign' style={[styles.bottomControlsButtonIcon, {color: iOSColors.black}]}/>
          </RectButton>
        )} */}

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

      </View>
    );
  }

}

//
// EditRecording
//  - TODO -> datatypes (along with types for EditRecording)
//

export async function EditRecording(props: {
  rec: Rec,
  f_bins: number,
  doneSpectroChunkWidth: number,
}): Promise<null | EditRecording> {

  // Render audioPath -> spectros
  //  - null if any value is null (audio samples < nperseg)
  //  - Compute spectros for denoise=true/false (true for preds, false so user can toggle to see pre-denoise)
  //    - In parallel, which does empirically achieve ~2x speedup
  //  - Compute spectros for single/chunked (single for preds, chunked for wrapped display)
  var _spectros: EditRecordingSpectros = await log.timedAsync('EditRecording: spectros', async () => new Map(
    await Promise.all([true, false].map(async denoise => {

      // Get a writable spectroCachePath for nonstandard f_bins
      //  - TODO Refactor to simplify
      const spectroCachePath = Rec.spectroCachePath(Rec.source(props.rec), {
        f_bins: props.f_bins,
        denoise,
      });

      // Compute spectro (cache to file)
      const single = await local(async () => {
        if (await log.timedAsync(`EditRecording[denoise=${denoise}]: fs.exists`, () => (
          fs.exists(spectroCachePath)
        ))) {
          // Cache hit: read width/height from cache spectro file
          return await log.timedAsync(`EditRecording[denoise=${denoise}]: Cache hit: Read spectro width/height`, async () => {
            const path = spectroCachePath;
            const {width, height} = await new Promise<Dim<number>>((resolve, reject) => {
              Image.getSize(`file://${path}`, (width, height) => resolve({width, height}), reject);
            });
            return {path, width, height};
          });
        } else {
          // Cache miss: compute spectro file (from audio file)
          return await log.timedAsync(`EditRecording[denoise=${denoise}]: Cache miss: renderAudioPathToSpectroPath`, async () => {
            return await NativeSpectro.renderAudioPathToSpectroPath(
              Rec.audioPath(props.rec),
              await ensureParentDir(spectroCachePath),
              {
                f_bins: props.f_bins,
                denoise,
              },
            );
          });
        }
      });

      // Compute [old-style] spectro chunks
      //  - This used to be a heavy computation where we cropped separate files per chunk, but it's now simplified to
      //    just a bunch of lightweight chunk metadata over the one full spectro (single.path)
      //  - TODO Simplify, remove cruft, reduce coupling (EditRecording <-> SpectroImage)
      const spectros = !single ? null : {
        single,
        chunked: local(() => {
          const totalWidth = single.width;
          const chunkWidth = props.doneSpectroChunkWidth;
          return _.range(0, totalWidth / chunkWidth).map(i => {
            const widthInterval = new Interval(chunkWidth * i, chunkWidth * (i + 1));
            return {
              imageFile: {
                path:   single.path, // Full image i/o cropped image (else very slow to compute all the tiny crops)
                width:  chunkWidth,
                height: single.height,
              },
              widthInterval,
              timeInterval: new Interval(
                widthInterval.lo / totalWidth * props.rec.duration_s,
                widthInterval.hi / totalWidth * props.rec.duration_s,
              ),
            };
          });
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

//
// styles
//

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
