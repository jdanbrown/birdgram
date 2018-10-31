import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import {
  Animated, Dimensions, FlatList, GestureResponderEvent, Image, ImageStyle, LayoutChangeEvent, Modal, Platform,
  ScrollView, SectionList, SectionListData, SectionListStatic, Text, TextInput, TextStyle, TouchableHighlight, View,
  ViewStyle, WebView,
} from 'react-native';
import ActionSheet from 'react-native-actionsheet'; // [Must `import ActionSheet` i/o `import { ActionSheet }`, else barf]
import FastImage from 'react-native-fast-image';
import * as Gesture from 'react-native-gesture-handler';
import {
  BaseButton, BorderlessButton, LongPressGestureHandler, PanGestureHandler, PinchGestureHandler, RectButton,
  TapGestureHandler,
  // FlatList, ScrollView, Slider, Switch, TextInput, // TODO Needed?
} from 'react-native-gesture-handler';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import SQLite from 'react-native-sqlite-storage';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { IconProps } from 'react-native-vector-icons/Icon';
import Feather from 'react-native-vector-icons/Feather';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
const fs = RNFB.fs;

import { ActionSheetBasic } from './ActionSheets';
import { Settings, ShowMetadata } from './Settings';
import { config } from '../config';
import { Quality, Rec, RecId, ScreenProps, SearchRecs, ServerConfig } from '../datatypes';
import { log, puts, tap } from '../log';
import Sound from '../sound';
import { querySql } from '../sql';
import { StyleSheet } from '../stylesheet';
import {
  chance, Clamp, Dim, finallyAsync, getOrSet, global, json, match, Point, pretty, setStateAsync, Styles,
  TabBarBottomConstants,
} from '../utils';

const sidewaysTextWidth = 14;

type ScrollViewState = {
  contentOffset: Point,
  // (More fields available in NativeScrollEvent)
}

type State = {
  scrollViewKey: string;
  scrollViewState: ScrollViewState;
  showFilters: boolean;
  showHelp: boolean;
  totalRecs?: number;
  queryText: string;
  query?: string;
  queryConfig: {
    quality: Array<Quality>,
    limit: number,
  };
  status: string;
  recs: Array<Rec>;
  playing?: {
    rec: Rec,
    sound: Sound,
    startTime?: number,
  };
  // Sync from/to Settings (1/3)
  spectroScale: number;
};

type Props = {
  spectroBase:       Dim<number>;
  spectroScaleClamp: Clamp<number>;
  screenProps:       ScreenProps;
};

export class SearchScreen extends Component<Props, State> {

  static defaultProps = {
    spectroBase:       {height: 20, width: Dimensions.get('window').width},
    spectroScaleClamp: {min: 1, max: 8},
  };

  db?: SQLiteDatabase;
  soundsCache: Map<RecId, Promise<Sound> | Sound> = new Map();

  saveActionSheet: RefObject<ActionSheet> = React.createRef();
  addActionSheet:  RefObject<ActionSheet> = React.createRef();
  sortActionSheet: RefObject<ActionSheet> = React.createRef();

  // Else we have to do too many setState's, which makes animations jump (e.g. ScrollView momentum)
  _scrollViewState: ScrollViewState = {
    contentOffset: {x: 0, y: 0},
  };

  scrollViewRef: RefObject<SectionListStatic<Rec>> = React.createRef();

  get serverConfig(): ServerConfig {
    return this.props.screenProps.serverConfig;
  }

  get settings(): Settings {
    return this.props.screenProps.settings;
  }

  constructor(props: Props) {
    super(props);
    this.state = {
      scrollViewKey: '',
      scrollViewState: this._scrollViewState,
      showFilters: false,
      showHelp: false,
      queryText: '',
      queryConfig: { // TODO Move to (global) SettingsScreen.state
        quality: ['A', 'B'],
        limit: 100,
      },
      status: '',
      recs: [],
      // Sync from/to Settings (2/3)
      spectroScale: this.settings.spectroScale,
    };
    global.SearchScreen = this; // XXX Debug
  }

  componentDidMount = async () => {
    log.debug('SearchScreen.componentDidMount');

    // Configure react-native-sound
    //  - TODO Experiment to figure out which "playback mode" and "audio session mode" we want
    //  - https://github.com/zmxv/react-native-sound/wiki/API#soundsetcategoryvalue-mixwithothers-ios-only
    //  - https://apple.co/2q2osEd
    //  - https://developer.apple.com/documentation/avfoundation/avaudiosession/audio_session_modes
    //  - https://apple.co/2R22tcg
    Sound.setCategory(
      'Playback', // Enable playback in silence mode [cargo-culted from README]
      true,       // mixWithOthers
    );
    Sound.setMode(
      'Default', // "The default audio session mode"
    );

    // Tell other apps we're using the audio device
    Sound.setActive(true);

    // Open db conn
    const dbFilename = SearchRecs.dbPath;
    const dbExists = await fs.exists(`${fs.dirs.MainBundleDir}/${dbFilename}`);
    if (!dbExists) {
      log.error(`DB file not found: ${dbFilename}`);
    } else {
      const dbLocation = `~/${dbFilename}`; // Relative to app bundle (copied into the bundle root by react-native-asset)
      this.db = await SQLite.openDatabase({
        name: dbFilename,               // Just for SQLite bookkeeping, I think
        readOnly: true,                 // Else it will copy the (huge!) db file from the app bundle to the documents dir
        createFromLocation: dbLocation, // Else readOnly will silently not work
      });
    }

    // Query db size (once, up front)
    await querySql<{totalRecs: number}>(this.db!, `
      select count(*) as totalRecs
      from search_recs
    `)(async results => {
      const [{totalRecs}] = results.rows.raw();
      await setStateAsync(this, {
        totalRecs,
      });
    });

    // XXX Faster dev
    await this.editQueryText('GREG,LASP,HOFI,NOFL,GTGR,SWTH,GHOW');
    this.submitQuery();

  }

  componentWillUnmount = async () => {
    log.debug('SearchScreen.componentWillUnmount');

    // Tell other apps we're no longer using the audio device
    Sound.setActive(false);

    // Release cached sound resources
    await this.releaseSounds();

  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    // log.debug('SearchScreen.componentDidUpdate');

    // Else _scrollViewState falls behind on non-scroll/non-zoom events (e.g. +/- buttons)
    this._scrollViewState = this.state.scrollViewState;

    // Sync from/to Settings (3/3)
    //  - These aren't typical: we only use this for (global) settings keys that we also keep locally in state so we can
    //    batch-update them with other local state keys (e.g. global spectroScale + local scrollViewKey)
    //  - TODO Is this a good pattern for "setState(x,y,z) locally + settings.set(x) globally"?
    await Promise.all([
      this.settings.set('spectroScale', this.state.spectroScale),
    ]);

  }

  editQueryText = async (queryText: string) => {
    await setStateAsync(this, {
      queryText,
    });
  }

  submitQuery = async () => {
    let {queryText, query} = this.state;
    if (queryText && queryText !== query) {
      query = queryText;

      // Record query + clear previous results
      await this.releaseSounds();
      await setStateAsync(this, {
        query,
        recs: [],
        status: '[Loading...]',
      });

      // Can't use window functions until sqlite ≥3.25.x
      //  - TODO Waiting on: https://github.com/litehelpers/Cordova-sqlite-storage/issues/828

      log.debug('query', query);
      await querySql<Rec>(this.db!, `
        select *
        from (
          select
            *,
            cast(taxon_order as real) as taxon_order_num
          from search_recs
          where
            species in (?) and
            quality in (?)
          order by
            xc_id desc
          limit ?
        )
        order by
          taxon_order_num asc,
          xc_id desc
      `, [
        query.split(',').map(x => _.trim(x).toUpperCase()),
        this.state.queryConfig.quality,
        this.state.queryConfig.limit,
      ])(async results => {
        const recs = results.rows.raw();
        await setStateAsync(this, {
          recs,
          status: `${recs.length} recs`,
        });
      });

    }
  }

  releaseSounds = async () => {
    log.info('Releasing cached sounds');
    await Promise.all(
      Array.from(this.soundsCache).map(async ([recId, soundAsync]) => {
        log.info('Releasing sound',
          // recId, // Noisy
        );
        (await soundAsync).release();
      }),
    );
    this.soundsCache = new Map();
  }

  getOrAllocateSoundAsync = async (rec: Rec): Promise<Sound> => {
    // Is sound already allocated (in the cache)?
    let soundAsync = this.soundsCache.get(rec.id);
    if (!soundAsync) {
      log.debug('Allocating sound',
        // rec.id, // Noisy
      );
      // Allocate + cache sound resource
      //  - Cache the promise so that get+set is atomic, else we race and allocate multiple sounds per rec.id
      //  - (Observable via log counts in the console: if num alloc > num release, then we're racing)
      this.soundsCache.set(rec.id, Sound.newAsync(
        Rec.audioPath(rec),
        Sound.MAIN_BUNDLE,
      ));
      soundAsync = this.soundsCache.get(rec.id);
    }
    return await soundAsync!;
  }

  toggleRecPlaying = (rec: Rec) => {

    // Eagerly allocate Sound resource for rec
    //  - TODO How eagerly should we cache this? What are the cpu/mem costs and tradeoffs?
    const soundAsync = this.getOrAllocateSoundAsync(rec);

    return async (event: Gesture.TapGestureHandlerStateChangeEvent) => {
      const {nativeEvent: {state, oldState, x, absoluteX}} = event; // Unpack SyntheticEvent (before async)
      if (
        // [Mimic Gesture.BaseButton]
        oldState === Gesture.State.ACTIVE &&
        state !== Gesture.State.CANCELLED
      ) {
        log.debug('toggleRecPlaying', pretty({x, recId: rec.id,
          playing: this.state.playing && {recId: this.state.playing.rec.id},
        }));

        // FIXME Races? Tap a lot of spectros really quickly and watch the "Playing rec" logs pile up

        const {playing} = this.state;

        // Stop any recs that are currently playing
        if (playing) {
          const {rec, sound, startTime} = playing;
          global.sound = sound; // XXX Debug

          // Stop sound playback
          log.debug('Stopping: rec', rec.id);
          await setStateAsync(this, {
            playing: undefined,
          });
          await sound.stopAsync();

        }

        // If touched rec was the currently playing rec, then we're done (it's stopped)
        // Else, play the (new) touched rec
        //  - HACK Override if seekOnPlay, so we can tap with abandon
        if (!this.recIsPlaying(rec.id, playing) || this.settings.seekOnPlay) {
          const sound = await soundAsync;
          global.sound = sound; // XXX Debug

          // Compute startTime to seek rec (if enabled)
          let startTime;
          if (this.settings.seekOnPlay) {
            startTime = this.spectroTimeFromX(sound, x, absoluteX);
          }

          // Play rec (if startTime is valid)
          if (!startTime || startTime < sound.getDuration()) {
            log.debug('Playing: rec', rec.id);
            if (startTime) {
              sound.setCurrentTime(startTime);
            }
            await setStateAsync(this, {
              playing: {rec, sound, startTime},
            });
            finallyAsync(sound.playAsync(), async () => {
              // Promise fulfills after playback completes / is stopped / fails
              log.debug('Done: rec', rec.id);
              await setStateAsync(this, {
                playing: undefined,
              });
            });
          }

        }

      }
    };
  }

  spectroTimeFromX = (sound: Sound, x: number, absoluteX: number): number => {
    const {contentOffset} = this._scrollViewState;
    const {spectroScale} = this.state;
    const {width} = this.spectroDimensions();
    const {audio_s} = this.serverConfig.api.recs.search_recs.params;
    const duration = sound.getDuration();
    const time = x / width * audio_s;
    // log.debug('spectroTimeFromX', pretty({time, x, absoluteX, contentOffset, width, spectroScale, audio_s, duration}));
    return time;
  }

  spectroXFromTime = (sound: Sound, time: number): number => {
    const {contentOffset} = this._scrollViewState;
    const {spectroScale} = this.state;
    const {width} = this.spectroDimensions();
    const {audio_s} = this.serverConfig.api.recs.search_recs.params;
    const duration = sound.getDuration();
    const x = time / audio_s * width;
    // log.debug('spectroXFromTime', pretty({x, time, contentOffset, width, spectroScale, audio_s, duration}));
    return x;
  }

  recIsPlaying = (recId: RecId, playing: undefined | {rec: Rec}): boolean => {
    return !playing ? false : playing.rec.id === recId;
  }

  spectroDimensions = (): Dim<number> => {
    return {
      width: _.sum([
        this.props.spectroBase.width * this.state.spectroScale,
        this.settings.showMetadata === 'none' ? -sidewaysTextWidth : 0,
      ]),
      height: this.props.spectroBase.height * this.state.spectroScale,
    };
  }

  onLongPress = (rec: Rec) => async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    if (state === Gesture.State.ACTIVE) {
      log.debug('onLongPress');
    }
  }

  onBottomControlsLongPress = async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    await match(state,
      [Gesture.State.ACTIVE, async () => await setStateAsync(this, {showHelp: true})],
      [Gesture.State.END,    async () => await setStateAsync(this, {showHelp: false})],
    )();
  }

  onMockPress = (rec: Rec) => async () => {
    console.log('renderLeftAction.onMockPress');
  }

  Filters = () => (
    <View style={[
      styles.filtersModal,
      {marginBottom: TabBarBottomConstants.DEFAULT_HEIGHT},
    ]}>
      <TextInput
        style={styles.queryInput}
        value={this.state.queryText}
        onChangeText={this.editQueryText}
        onSubmitEditing={this.submitQuery}
        autoCorrect={false}
        autoCapitalize='characters'
        enablesReturnKeyAutomatically={true}
        placeholder='Species'
        returnKeyType='search'
      />
      <Text>Filters</Text>
      <Text>- quality</Text>
      <Text>- month</Text>
      <Text>- species likelihood [bucketed ebird priors]</Text>
      <Text>- rec text search [conflate fields]</Text>
      <RectButton onPress={async () => await setStateAsync(this, {showFilters: false})}>
        <View style={{padding: 10, backgroundColor: iOSColors.blue}}>
          <Text>Done</Text>
        </View>
      </RectButton>
    </View>
  );

  cycleMetadata = async () => {
    const next = (showMetadata: ShowMetadata) => match<ShowMetadata, ShowMetadata, ShowMetadata>(showMetadata,
      ['none',    'oneline'],
      ['oneline', 'full'],
      ['full',    'none'],
    );
    await this.settings.set('showMetadata', next(this.settings.showMetadata));

    // Scroll SectionList so that same ~top recs are showing after drawing with new item/section heights
    //  - TODO More experimentation needed
    // requestAnimationFrame(() => {
    //   if (this.scrollViewRef.current) {
    //     this.scrollViewRef.current.scrollToLocation({
    //       animated: false,
    //       sectionIndex: 3, itemIndex: 3, // TODO Calculate real values to restore
    //       viewPosition: 0, // 0: top, .5: middle, 1: bottom
    //     });
    //   }
    // });

  }

  scaleSpectros = async (delta: number) => {
    await setStateAsync(this, (state, props) => {
      // Round current spectroScale so that +1/-1 deltas snap back to non-fractional scales (e.g. after pinch zooms)
      const spectroScale = this.clampSpectroScaleY(Math.round(state.spectroScale) + delta);
      return {
        spectroScale,
        scrollViewState: {
          // FIXME Zoom in -> scroll far down+right -> use '-' button to zoom out -> scroll view clipped b/c contentOffset nonzero
          contentOffset: {
            x: this._scrollViewState.contentOffset.x * spectroScale / state.spectroScale,
            y: this._scrollViewState.contentOffset.y * spectroScale / state.spectroScale,
          },
        },
      };
    });
  }

  clampSpectroScaleY = (spectroScale: number): number => _.clamp(
    spectroScale,
    this.props.spectroScaleClamp.min,
    this.props.spectroScaleClamp.max,
  );

  BottomControls = (props: {}) => (
    <View style={styles.bottomControls}>
      {/* Filters */}
      <this.BottomControlsButton
        help='Filters'
        onPress={async () => await setStateAsync(this, {showFilters: true})}
        iconProps={{name: 'filter'}}
      />
      {/* Save as new list / add all to saved list / share list */}
      <this.BottomControlsButton
        help='Save'
        onPress={() => this.saveActionSheet.current!.show()}
        iconProps={{name: 'star'}}
        // iconProps={{name: 'share'}}
      />
      {/* Add species (select species manually) */}
      <this.BottomControlsButton
        help='Add'
        onPress={() => this.addActionSheet.current!.show()}
        // iconProps={{name: 'user-plus'}}
        // iconProps={{name: 'file-plus'}}
        // iconProps={{name: 'folder-plus'}}
        iconProps={{name: 'plus-circle'}}
        // iconProps={{name: 'plus'}}
      />
      {/* Toggle sort: species probs / rec dist / manual list */}
      <this.BottomControlsButton
        help='Sort'
        onPress={() => this.sortActionSheet.current!.show()}
        // iconProps={{name: 'chevrons-down'}}
        // iconProps={{name: 'chevron-down'}}
        iconProps={{name: 'arrow-down'}}
        // iconProps={{name: 'arrow-down-circle'}}
      />
      {/* Cycle metadata: none / oneline / full */}
      <this.BottomControlsButton
        help='Info'
        onPress={() => this.cycleMetadata()}
        iconProps={{name: 'file-text'}}
        // iconProps={{name: 'credit-card', style: Styles.flipVertical}}
        // iconProps={{name: 'sidebar', style: Styles.rotate270}}
      />
      {/* Toggle editing controls for rec/species */}
      <this.BottomControlsButton
        help='Edit'
        onPress={() => this.settings.toggle('editing')}
        active={this.settings.editing}
        iconProps={{name: 'sliders'}}
        // iconProps={{name: 'edit'}}
        // iconProps={{name: 'edit-2'}}
        // iconProps={{name: 'edit-3'}}
        // iconProps={{name: 'layout', style: Styles.flipBoth}}
      />
      {/* [Toggle play/pause crosshairs] */}
      <this.BottomControlsButton
        help='Seek'
        onPress={() => this.settings.toggle('seekOnPlay')}
        active={this.settings.seekOnPlay}
        iconProps={{name: 'crosshair'}}
      />
      {/* Zoom more/fewer recs (spectro height) */}
      {/* - TODO Disable when spectroScale is min/max */}
      <this.BottomControlsButton
        help='Dense'
        disabled={this.state.spectroScale === this.props.spectroScaleClamp.min}
        onPress={async () => await this.scaleSpectros(-1)}
        iconProps={{name: 'align-justify'}} // 4 horizontal lines
      />
      <this.BottomControlsButton
        help='Tall'
        disabled={this.state.spectroScale === this.props.spectroScaleClamp.max}
        onPress={async () => await this.scaleSpectros(+1)}
        iconProps={{name: 'menu'}}          // 3 horizontal lines
      />
    </View>
  );

  BottomControlsButton = (props: {
    help: string,
    iconProps: IconProps,
    onPress?: (pointerInside: boolean) => void,
    active?: boolean,
    disabled?: boolean,
  }) => {
    const {style: iconStyle, ...iconProps} = props.iconProps;
    return (
      <LongPressGestureHandler onHandlerStateChange={this.onBottomControlsLongPress}>
        <BorderlessButton
          style={styles.bottomControlsButton}
          onPress={props.disabled ? undefined : props.onPress}
        >
          {this.state.showHelp && (
            <Text style={styles.bottomControlsButtonHelp}>{props.help}</Text>
          )}
          <Feather
            style={[
              styles.bottomControlsButtonIcon,
              iconStyle,
              (
                props.disabled ? {color: iOSColors.gray} :
                props.active   ? {color: iOSColors.blue} :
                {}
              ),
            ]}
            {...iconProps}
          />
        </BorderlessButton>
      </LongPressGestureHandler>
    );
  }

  sectionsForRecs = (recs: Array<Rec>): Array<SectionListData<Rec>> => {
    const sections = [];
    let section;
    for (let rec of recs) {
      const title = rec.species;
      if (!section || title !== section.title) {
        if (section) sections.push(section);
        section = {
          title,
          data: [] as Rec[],
          species: rec.species,
          species_taxon_order: rec.species_taxon_order,
          species_com_name: rec.species_com_name,
          species_sci_name: rec.species_sci_name,
          recs_for_sp: rec.recs_for_sp,
        };
      }
      section.data.push(rec);
    }
    if (section) sections.push(section);
    return sections;
  }

  SpeciesEditingButtons = () => (
    <View style={styles.sectionSpeciesEditingView}>
      <BorderlessButton style={styles.sectionSpeciesEditingButton} onPress={() => {}}>
        <Feather style={styles.sectionSpeciesEditingIcon} name='move' />
      </BorderlessButton>
      <BorderlessButton style={styles.sectionSpeciesEditingButton} onPress={() => {}}>
        <Feather style={styles.sectionSpeciesEditingIcon} name='search' />
      </BorderlessButton>
      <BorderlessButton style={styles.sectionSpeciesEditingButton} onPress={() => {}}>
        <Feather style={styles.sectionSpeciesEditingIcon} name='user-x' />
      </BorderlessButton>
      <BorderlessButton style={styles.sectionSpeciesEditingButton} onPress={() => {}}>
        <Feather style={styles.sectionSpeciesEditingIcon} name='plus-circle' />
      </BorderlessButton>
    </View>
  );

  RecEditingButtons = () => (
    <View style={styles.recEditingView}>
      <BorderlessButton style={styles.recEditingButton} onPress={() => {}}>
        <Feather style={styles.recEditingIcon} name='move' />
      </BorderlessButton>
      <BorderlessButton style={styles.recEditingButton} onPress={() => {}}>
        <Feather style={styles.recEditingIcon} name='search' />
      </BorderlessButton>
      <BorderlessButton style={styles.recEditingButton} onPress={() => {}}>
        <Feather style={styles.recEditingIcon} name='x' />
      </BorderlessButton>
      <BorderlessButton style={styles.recEditingButton} onPress={() => {}}>
        <Feather style={styles.recEditingIcon} name='star' />
      </BorderlessButton>
    </View>
  );

  RecText = <X extends {children: any, flex?: number}>(props: X) => {
    const flex = props.flex || 1;
    return (<Text
      style={[styles.recText, {flex}]}
      numberOfLines={1}
      ellipsizeMode='tail'
      {...props}
    />);
  }

  ModalsAndActionSheets = () => (
    <View>
      <Modal
        visible={this.state.showFilters}
        animationType='none' // 'none' | 'slide' | 'fade'
        transparent={true}
        children={this.Filters()}
      />
      <ActionSheetBasic
        innerRef={this.saveActionSheet}
        options={[
          ['Save as new list',      () => {}],
          ['Add all to saved list', () => {}],
          ['Share list',            () => {}],
        ]}
      />
      <ActionSheetBasic
        innerRef={this.addActionSheet}
        options={[
          ['Add a species manually', () => {}],
          ['+ num species',          () => {}],
          ['– num species',          () => {}],
          ['+ num recs per species', () => {}],
          ['– num recs per species', () => {}],
        ]}
      />
      <ActionSheetBasic
        innerRef={this.sortActionSheet}
        options={
          // this.state.queryRec ? [ // TODO queryRec
          true ? [
            ['Sort by species, then by recs', () => {}],
            ['Sort by recs only',             () => {}],
            ['Order manually',                () => {}],
          ] : [
            ['Sort recs by similarity',       () => {}],
            ['Order manually',                () => {}],
          ]
        }
      />
    </View>
  );

  render = () => (
    <View style={styles.container}>

      {/* Recs list (with pan/pinch) */}
      {/* - We use ScrollView instead of SectionList to avoid _lots_ of opaque pinch-to-zoom bugs */}
      {/* - We use ScrollView instead of manual gestures (react-native-gesture-handler) to avoid _lots_ of opaque animation bugs */}
      <ScrollView
        // @ts-ignore [Why doesn't this typecheck?]
        ref={this.scrollViewRef as RefObject<Component<SectionListStatic<Rec>, any, any>>}
        style={styles.recList}

        // Scroll/zoom
        //  - Force re-layout on zoom change, else bad things (that I don't understand)
        key={this.state.scrollViewKey}
        //  - Expand container width else we can't scroll horizontally
        contentContainerStyle={{
          width: this.props.spectroBase.width * this.state.spectroScale,
        }}
        // This is (currently) the only place we use state.scrollViewState i/o this._scrollViewState
        contentOffset={tap(this.state.scrollViewState.contentOffset, x => {
          // log.debug('render.contentOffset', json(x)); // XXX Debug
        })}
        bounces={false}
        bouncesZoom={false}
        minimumZoomScale={this.props.spectroScaleClamp.min / this.state.spectroScale}
        maximumZoomScale={this.props.spectroScaleClamp.max / this.state.spectroScale}
        onScrollEndDrag={async ({nativeEvent}) => {
          // log.debug('onScrollEndDrag', json(nativeEvent)); // XXX Debug
          const {contentOffset, zoomScale, velocity} = nativeEvent;
          this._scrollViewState = {contentOffset};
          if (
            zoomScale !== 1              // Don't trigger zoom if no zooming happened (e.g. only scrolling)
            // && velocity !== undefined // [XXX Unreliable] Don't trigger zoom on 1/2 fingers released, wait for 2/2
          ) {
            const scale = zoomScale * this.state.spectroScale;
            // log.debug('ZOOM', json(nativeEvent)); // XXX Debug
            // Trigger re-layout so non-image components (e.g. text) redraw at non-zoomed size
            await setStateAsync(this, {
              scrollViewState: this._scrollViewState,
              spectroScale: this.clampSpectroScaleY(scale),
              scrollViewKey: chance.hash(), // Else bad things (that I don't understand)
            });
          }
        }}

        // Mimic a SectionList
        children={
          _.flatten(this.sectionsForRecs(this.state.recs).map(({
            title,
            data: recs,
            species,
            species_taxon_order,
            species_com_name,
            species_sci_name,
            recs_for_sp,
          }) => [

            // Species header
            this.settings.showMetadata === 'none' ? null : (
              <View
                key={`section-${title}`}
                style={styles.sectionSpecies}
              >
                {/* Species editing buttons */}
                {!this.settings.editing ? null : (
                  <this.SpeciesEditingButtons />
                )}
                {/* Species name */}
                <Text numberOfLines={1} style={styles.sectionSpeciesText}>
                  {species_com_name}
                </Text>
                {/* Debug info */}
                {this.settings.showDebug && (
                  <Text numberOfLines={1} style={[{marginLeft: 'auto', alignSelf: 'center'}, this.settings.debugText]}>
                    ({recs_for_sp} recs)
                  </Text>
                )}
              </View>
            ),

            // Rec rows
            ...recs.map((rec, index) => (

              // Rec row
              <Animated.View
                key={`row-${rec.id.toString()}`}
                style={styles.recRow}
              >

                {/* Rec editing buttons */}
                {/* - TODO Flex image width so we can show these on the right (as is, they'd be pushed off screen) */}
                {!this.settings.editing ? null : (
                  <this.RecEditingButtons />
                )}

                {/* Rec region without the editing buttons  */}
                <Animated.View style={[styles.recRowInner,
                  // HACK Visual feedback for playing rec (kill after adding scrolling bar)
                  (!this.recIsPlaying(rec.id, this.state.playing)
                    ? {borderColor: iOSColors.gray}
                    : {borderColor: iOSColors.red, borderTopWidth: 1}
                  ),
                ]}>

                  {/* Rec row */}
                  <View style={{flexDirection: 'row'}} collapsable={false}>

                    {/* Sideways species label (sometimes) */}
                    {/* - NOTE Keep outside of TapGestureHandler else spectroTimeFromX/spectroXFromTime have to adjust */}
                    {this.settings.showMetadata !== 'none' ? null : (
                      <View style={styles.recSpeciesSidewaysView}>
                        <View style={styles.recSpeciesSidewaysViewInner}>
                          <Text numberOfLines={1} style={[styles.recSpeciesSidewaysText, {
                            fontSize: this.state.spectroScale < 2 ? 6 : 11, // Compact species label to fit within tiny rows
                          }]}>
                            {rec.species}
                          </Text>
                        </View>
                      </View>
                    )}

                    {/* Spectro (tap) */}
                    <LongPressGestureHandler onHandlerStateChange={this.onLongPress(rec)}>
                      <TapGestureHandler onHandlerStateChange={this.toggleRecPlaying(rec)}>
                        <Animated.View style={{flex: 1}}>

                          {/* Image */}
                          <Animated.Image
                            style={this.spectroDimensions()}
                            resizeMode='stretch'
                            source={{uri: Rec.spectroPath(rec)}}
                          />

                          {/* Current time cursor (if playing + startTime) */}
                          {this.recIsPlaying(rec.id, this.state.playing) && (
                            this.state.playing!.startTime && (
                              <View style={{
                                position: 'absolute',
                                left: this.spectroXFromTime(
                                  this.state.playing!.sound,
                                  this.state.playing!.startTime!,
                                ),
                                width: 1,
                                height: '100%',
                                backgroundColor: 'black',
                              }}/>
                            )
                          )}

                        </Animated.View>
                      </TapGestureHandler>
                    </LongPressGestureHandler>

                  </View>

                  {/* Rec metadata */}
                  {match(this.settings.showMetadata,
                    ['none', null],
                    ['oneline', (
                      <View style={styles.recMetadataOneline}>
                        <this.RecText flex={3} children={rec.xc_id} />
                        <this.RecText flex={1} children={rec.quality} />
                        <this.RecText flex={2} children={rec.month_day} />
                        <this.RecText flex={4} children={Rec.placeNorm(rec)} />
                        {ccIcon({style: styles.recTextFont})}
                        <this.RecText flex={4} children={` ${rec.recordist}`} />
                      </View>
                    )],
                    ['full', (
                      <View style={styles.recMetadataFull}>
                        <this.RecText flex={3} children={rec.xc_id} />
                        <this.RecText flex={1} children={rec.quality} />
                        <this.RecText flex={2} children={rec.month_day} />
                        <this.RecText flex={4} children={Rec.placeNorm(rec)} />
                        {ccIcon({style: styles.recTextFont})}
                        <this.RecText flex={4} children={` ${rec.recordist}`} />
                      </View>
                    )],
                  )}

                </Animated.View>

              </Animated.View>

            )),

          ]))
        }

      />

      {/* Debug info */}
      <View style={this.settings.debugView}>
        <Text style={this.settings.debugText}>Status: {this.state.status} ({this.state.totalRecs || '?'} total)</Text>
        <Text style={this.settings.debugText}>Filters: {JSON.stringify(this.state.queryConfig)}</Text>
      </View>

      {/* Bottom controls */}
      <this.BottomControls/>

      {/* Modals + action sheets */}
      <this.ModalsAndActionSheets/>

    </View>
  );

}

function ccIcon(props?: object): Element {
  const [icon] = licenseTypeIcons('cc', props);
  return icon;
}

function licenseTypeIcons(license_type: string, props?: object): Array<Element> {
  license_type = `cc-${license_type}`;
  return license_type.split('-').map(k => (<FontAwesome5
    key={k}
    name={k === 'cc' ? 'creative-commons' : `creative-commons-${k}`}
    {...props}
  />));
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  filtersModal: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 300,
    backgroundColor: iOSColors.green,
  },
  bottomControls: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 5,
    backgroundColor: iOSColors.midGray,
  },
  bottomControlsButton: {
    flex: 1,
    alignItems: 'center',
  },
  bottomControlsButtonHelp: {
    ...material.captionObject,
  },
  bottomControlsButtonIcon: {
    ...material.headlineObject,
  },
  queryInput: {
    borderWidth: 1, borderColor: 'gray',
    ...material.display1Object,
  },
  summaryText: {
    ...material.captionObject,
  },
  recList: {
    // borderWidth: 1, borderColor: 'gray',
  },
  sectionSpecies: {
    flexDirection: 'row',
    // ...material.body1Object, backgroundColor: iOSColors.customGray, // Black on white
    ...material.body1WhiteObject, backgroundColor: iOSColors.gray, // White on black
  },
  sectionSpeciesText: {
    alignSelf: 'center', // Align text vertically
  },
  sectionSpeciesEditingView: {
    flexDirection: 'row',
    zIndex: 1, // Over spectro image
  },
  sectionSpeciesEditingButton: {
    width: 35, // Need explicit width (i/o flex:1) else view shows with width:0
    justifyContent: 'center', // Align icon vertically
    backgroundColor: iOSColors.gray,
  },
  sectionSpeciesEditingIcon: {
    ...material.headlineObject,
    alignSelf: 'center', // Align icon horizontally
  },
  recRow: {
    flex: 1, flexDirection: 'row',
  },
  recRowInner: {
    flex: 1, flexDirection: 'column',
    borderBottomWidth: 1,
    // borderColor: 'gray', // Set dynamically
  },
  recSpeciesSidewaysView: {
    backgroundColor: iOSColors.gray, // TODO Map rec.species -> color
    justifyContent: 'center',        // Else sideways text is to the above
    alignItems: 'center',            // Else sideways text is to the right
    width: sidewaysTextWidth,        // HACK Manually shrink outer view width to match height of sideways text
    zIndex: 1,                       // Over spectro image
  },
  recSpeciesSidewaysViewInner: {     // View>View>Text else the text aligment doesn't work
    transform: [{rotate: '270deg'}], // Rotate text sideways
    width: 100,                      // Else text is clipped to outer view's (smaller) width
    // borderWidth: 1, borderColor: 'black', // XXX Debug
  },
  recSpeciesSidewaysText: {
    alignSelf: 'center',             // Else sideways text is to the bottom
    // fontSize: ...,                // Set dynamically
    // ...material.captionObject,    // (Sticking with default color:'black')
  },
  recMetadataOneline: {
    flex: 2, flexDirection: 'row', // TODO Eh...
  },
  recMetadataFull: {
    flex: 1,
    flexDirection: 'column',
  },
  recText: {
    ...material.captionObject,
  },
  recTextFont: {
    ...material.captionObject,
  },
  recSpectro: {
  },
  recEditingView: {
    flexDirection: 'row',
    zIndex: 1, // Over spectro image
  },
  recEditingButton: {
    width: 35, // Need explicit width (i/o flex:1) else view shows with width:0
    justifyContent: 'center', // Align icon vertically
    backgroundColor: iOSColors.midGray,
  },
  recEditingIcon: {
    // ...material.titleObject,
    ...material.headlineObject,
    alignSelf: 'center', // Align icon horizontally
  },
  swipeButtons: {
    flexDirection: 'row',
  },
  swipeButton: {
    flex: 1,
    alignItems: 'center',
  },
  swipeButtonText: {
  },
});
