// @ts-ignore
import animated from "animated.macro";
import _ from 'lodash';
import React, { Component, ReactNode, RefObject } from 'react';
import {
  Animated, Dimensions, FlatList, GestureResponderEvent, Image, ImageStyle, LayoutChangeEvent, Modal, Platform,
  SectionList, SectionListData, Text, TextInput, TextStyle, TouchableHighlight, View, ViewStyle, WebView,
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
import { Quality, Rec, RecId, SearchRecs } from '../datatypes';
import { log, puts, tap } from '../log';
import Sound from '../sound';
import { querySql } from '../sql';
import { StyleSheet } from '../stylesheet';
import { finallyAsync, getOrSet, global, match, Styles, TabBarBottomConstants } from '../utils';

type State = {
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
  };
};

type Props = {
  spectroBase:        WidthHeight<number>;
  spectroScaleYClamp: Clamp<number>;
};

export class SearchScreen extends Component<Props, State> {

  static defaultProps = {
    spectroBase:        {h: 20, w: Dimensions.get('window').width},
    spectroScaleYClamp: {min: 1, max: 8},
  };

  db?: SQLiteDatabase;
  soundsCache: Map<RecId, Promise<Sound> | Sound>;
  saveActionSheet: RefObject<ActionSheet>;
  addActionSheet: RefObject<ActionSheet>;
  sortActionSheet: RefObject<ActionSheet>;
  pinchRef: RefObject<PinchGestureHandler>;
  pinchScaleX: PinchScaleX;
  panTranslateX: Map<RecId, PanTranslateX>;
  // panRefs: Map<RecId, RefObject<PanGestureHandler>>; // XXX Didn't work

  constructor(props: Props) {
    super(props);

    this.state = {
      showFilters: false,
      showHelp: false,
      queryText: '',
      queryConfig: { // TODO Move to (global) SettingsScreen.state
        quality: ['A', 'B'],
        limit: 100,
      },
      status: '',
      recs: [],
    };

    this.soundsCache = new Map();
    this.saveActionSheet = React.createRef();
    this.addActionSheet = React.createRef();
    this.sortActionSheet = React.createRef();
    this.pinchRef = React.createRef();
    this.pinchScaleX = new PinchScaleX(this.props.spectroBase,
      // 2, {min: 1, max: 10}, // TODO TODO Restore
      // .5, {min: .5, max: 10},
      1, {min: .5, max: 10},
    );
    this.panTranslateX = new Map();
    // this.panRefs = new Map(); // XXX Didn't work

    global.SearchScreen = this; // XXX Debug
  }

  componentDidUpdate = (prevProps: Props, prevState: State) => {

    // Drop PanTranslateX resources for recs we no longer have, and preserve PanTranslateX state for recs we still have
    this.panTranslateX = new Map(this.state.recs.map<[RecId, PanTranslateX]>(rec => [
      rec.id,
      this.panTranslateX.get(rec.id) || new PanTranslateX(
        this.pinchScaleX.outScale, // scale
        0, {min: 0, max: 0},
        // 0, {min: -Rec.spectroWidthPx(rec), max: 0}, // TODO TODO Clamp
      ),
    ]));
    log.info('panTranslateX = new Map', this.panTranslateX); // TODO TODO XXX Debug

    // // XXX Didn't work
    // this.panRefs = new Map(this.state.recs.map<[RecId, RefObject<PanGestureHandler>]>(rec => [
    //   rec.id,
    //   React.createRef<PanGestureHandler>(),
    // ]));

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
    `)(results => {
      const [{totalRecs}] = results.rows.raw();
      this.setState({
        totalRecs,
      });
    });

    // XXX Faster dev
    this.editQueryText('GREG,LASP,HOFI,NOFL,GTGR,SWTH,GHOW');
    this.submitQuery();

  }

  componentWillUnmount = async () => {
    log.debug('SearchScreen.componentWillUnmount');

    // Tell other apps we're no longer using the audio device
    Sound.setActive(false);

    // Release cached sound resources
    await this.releaseSounds();

  }

  editQueryText = (queryText: string) => {
    this.setState({
      queryText,
    });
  }

  submitQuery = async () => {
    let {queryText, query} = this.state;
    if (queryText && queryText !== query) {
      query = queryText;

      // Record query + clear previous results
      await this.releaseSounds();
      this.setState({
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
      ])(results => {
        const recs = results.rows.raw();
        this.setState({
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

    return async (pointerInside: boolean) => {
      log.debug('toggleRecPlaying');
      log.debug('rec', rec);
      log.debug('this.state.playing', this.state.playing);

      // FIXME Races? Tap a lot of spectros really quickly and watch the "Playing rec" logs pile up

      const {playing} = this.state;

      // Stop any recs that are currently playing
      if (playing) {
        const {rec, sound} = playing;

        // Stop sound playback
        log.debug('Stopping playing rec', rec.id);
        this.setState({
          playing: undefined,
        });
        await sound.stopAsync();

        global.sound = sound; // XXX Debug

      }

      // If touched rec was the currently playing rec, then we're done (it's stopped)
      // Else, play the (new) touched rec
      if (!this.recIsPlaying(rec.id, playing)) {

        const sound = await soundAsync;

        global.sound = sound; // XXX Debug

        // Play rec
        log.debug('Playing rec', rec.id);
        this.setState({
          playing: {rec, sound},
        });
        finallyAsync(sound.playAsync(), () => {
          // Promise fulfills after playback completes / is stopped / fails
          log.debug('Done playing rec', rec.id);
          this.setState({
            playing: undefined,
          });
        });

      }

      // log.debug('toggleRecPlaying: done');
    };
  }

  recIsPlaying = (recId: RecId, playing: undefined | {rec: Rec}): boolean => {
    return !playing ? false : playing.rec.id === recId;
  }

  onLongPress = (rec: Rec) => async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    if (state === Gesture.State.ACTIVE) {
      log.debug('onLongPress');
    }
  }

  onBottomControlsLongPress = async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    match(state,
      [Gesture.State.ACTIVE, () => this.setState({showHelp: true})],
      [Gesture.State.END,    () => this.setState({showHelp: false})],
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
      <Text>Filters</Text>
      <Text>- quality</Text>
      <Text>- month</Text>
      <Text>- species likelihood [bucketed ebird priors]</Text>
      <Text>- rec text search [conflate fields]</Text>
      {/* XXX For reference
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
      */}
      <RectButton onPress={() => this.setState({showFilters: false})}>
        <View style={{padding: 10, backgroundColor: iOSColors.blue}}>
          <Text>Done</Text>
        </View>
      </RectButton>
    </View>
  );

  cycleMetadata = async (settings: Settings) => {
    const next = (showMetadata: ShowMetadata) => match<ShowMetadata, ShowMetadata, ShowMetadata>(showMetadata,
      ['none',    'oneline'],
      ['oneline', 'full'],
      ['full',    'none'],
    );
    await settings.set('showMetadata', next(settings.showMetadata));
  }

  zoomSpectroHeight = async (settings: Settings, step: number) => {
    const before = settings.spectroScaleY;
    const after = _.clamp(
      before + step,
      this.props.spectroScaleYClamp.min,
      this.props.spectroScaleYClamp.max,
    );
    if (after !== before) {
      await settings.set('spectroScaleY', after);
    }
  }

  BottomControls = (props: {}) => (
    <Settings.Context.Consumer children={settings => (
      <View style={styles.bottomControls}>
        {/* Filters */}
        <this.BottomControlsButton
          help='Filters'
          onPress={() => this.setState({showFilters: true})}
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
          onPress={() => this.cycleMetadata(settings)}
          iconProps={{name: 'file-text'}}
          // iconProps={{name: 'credit-card', style: Styles.flipVertical}}
          // iconProps={{name: 'sidebar', style: Styles.rotate270}}
        />
        {/* Toggle editing controls for rec/species */}
        <this.BottomControlsButton
          help='Edit'
          onPress={() => settings.toggle('editing')}
          iconProps={{name: 'sliders'}}
          // iconProps={{name: 'edit'}}
          // iconProps={{name: 'edit-2'}}
          // iconProps={{name: 'edit-3'}}
          // iconProps={{name: 'layout', style: Styles.flipBoth}}
        />
        {/* Zoom more/fewer recs (spectro height) */}
        {/* - TODO Disable when spectroScaleY is min/max */}
        <this.BottomControlsButton
          help='Denser'
          onPress={() => this.zoomSpectroHeight(settings, -1)}
          iconProps={{name: 'align-justify'}} // 4 horizontal lines
        />
        <this.BottomControlsButton
          help='Taller'
          onPress={() => this.zoomSpectroHeight(settings, +1)}
          iconProps={{name: 'menu'}}          // 3 horizontal lines
        />
      </View>
    )}/>
  );

  BottomControlsButton = (props: {
    help: string,
    onPress?: (pointerInside: boolean) => void,
    iconProps: IconProps,
  }) => {
    const {style: iconStyle, ...iconProps} = props.iconProps;
    return (
      <LongPressGestureHandler onHandlerStateChange={this.onBottomControlsLongPress}>
        <BorderlessButton style={styles.bottomControlsButton} onPress={props.onPress}>
          {this.state.showHelp && (
            <Text style={styles.bottomControlsButtonHelp}>{props.help}</Text>
          )}
          <Feather style={[styles.bottomControlsButtonIcon, iconStyle]} {...iconProps} />
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
    <Settings.Context.Consumer children={settings => (
      <View style={styles.container}>

        <PinchGestureHandler
          ref={this.pinchRef}
          onGestureEvent={this.pinchScaleX.onPinchGesture}
          onHandlerStateChange={this.pinchScaleX.onPinchState}
        >
          <Animated.View style={{flex: 1}}>

              <SectionList
                style={styles.recList}
                sections={this.sectionsForRecs(this.state.recs)}
                keyExtractor={(rec, index) => rec.id.toString()}
                initialNumToRender={20}
                renderSectionHeader={({section: {species_com_name, species_sci_name, recs_for_sp}}) => (
                  settings.showMetadata === 'none' ? null : (
                    <View style={styles.sectionSpecies}>
                      {!settings.editing ? null : (
                        <this.SpeciesEditingButtons />
                      )}
                      <Text numberOfLines={1} style={styles.sectionSpeciesText}>
                        {species_com_name}
                      </Text>
                      {settings.showDebug && (
                        <Text numberOfLines={1} style={[{marginLeft: 'auto', alignSelf: 'center'}, settings.debugText]}>
                          ({recs_for_sp} recs)
                        </Text>
                      )}
                    </View>
                  )
                )}
                renderItem={({item: rec, index}) => (
                  <Animated.View style={styles.recRow}>

                    {/* TODO Flex image width so we can show these on the right (as is, they'd be pushed off screen) */}
                    {!settings.editing ? null : (
                      <this.RecEditingButtons />
                    )}

                    <Animated.View style={[styles.recRowInner,
                      // HACK Visual feedback for playing rec. Kill this after
                      (!this.recIsPlaying(rec.id, this.state.playing)
                        ? {borderColor: iOSColors.gray}
                        : {borderColor: iOSColors.red, borderTopWidth: 1}
                      ),
                    ]}>

                      <PanGestureHandler
                        // [Why do these trigger undefined.onPanGesture on init?]
                        onGestureEvent       = {(this.panTranslateX.get(rec.id) || {onPanGesture: undefined}).onPanGesture}
                        onHandlerStateChange = {(this.panTranslateX.get(rec.id) || {onPanState:   undefined}).onPanState}
                        minDeltaX={10} // Else rec pans eat all the up/down scroll events (copied from Swipeable)
                        // FIXME How to pinch instead of 2x pan when fingers are on different spectros?
                        // waitFor={this.pinchRef} // XXX Close! -- but makes one-finger pans slow to register [why?]
                        // maxPointers={1}         // XXX Nope, doesn't control for multiple pointers on separate spectros
                        // XXX Nope, doesn't prevent multiple pointers on separate spectros [why not?]
                        // ref={this.panRefs.get(rec.id)} waitFor={[
                        //   ...tap((
                        //       Array.from(this.panRefs.entries())
                        //       .filter(([recId, ref]) => recId < rec.id)
                        //       .map(([recId, ref]) => ref)
                        //   ), xs => puts(rec.xc_id, xs))
                        // ]}
                      >
                        <Animated.View>

                          <LongPressGestureHandler onHandlerStateChange={this.onLongPress(rec)}>
                            <BaseButton
                              onPress={this.toggleRecPlaying(rec)}
                              style={{
                                // ...(this.recIsPlaying(rec.id, this.state.playing) && {
                                //   paddingVertical: 1, backgroundColor: 'red',
                                // }),
                              }}
                            >

                              <Animated.View style={{flexDirection: 'row'}} collapsable={false}>

                                {settings.showMetadata !== 'none' ? null : (
                                  <View style={styles.recSpeciesSidewaysView}>
                                    <View style={styles.recSpeciesSidewaysViewInner}>
                                      <Text numberOfLines={1} style={[styles.recSpeciesSidewaysText, {
                                        fontSize: match<number, number, number>(settings.spectroScaleY,
                                          [1,             6], // Compact species label to fit within tiny rows
                                          [match.default, 11],
                                        ),
                                      }]}>
                                        {rec.species}
                                      </Text>
                                    </View>
                                  </View>
                                )}

                                <Animated.Image
                                  style={[{
                                    // XXX Can't animate height
                                    //  - "Error: Style property 'height' is not supported by native animated module"
                                    // height: this.pinchScaleX.outScaleY.interpolate({
                                    //   inputRange: [0, 1],
                                    //   outputRange: [0, this.pinchScaleX.spectroBase.h],
                                    // }),
                                    width:  this.pinchScaleX.spectroBase.w,
                                    height: this.pinchScaleX.spectroBase.h * settings.spectroScaleY,
                                    transform: [
                                      ...this.pinchScaleX.transform,
                                      ...(this.panTranslateX.get(rec.id) || {transform: []}).transform,
                                    ],
                                  }]}
                                  resizeMode='stretch'
                                  source={{uri: Rec.spectroPath(rec)}}
                                />

                              </Animated.View>

                            </BaseButton>
                          </LongPressGestureHandler>

                        </Animated.View>
                      </PanGestureHandler>

                      {match(settings.showMetadata,
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
                )}
              />

          </Animated.View>
        </PinchGestureHandler>

        <View style={settings.debugView}>
          <Text style={settings.debugText}>{this.state.status} ({this.state.totalRecs || '?'} total)</Text>
          <Text style={settings.debugText}>{JSON.stringify(this.state.queryConfig)}</Text>
        </View>

        <this.BottomControls/>
        <this.ModalsAndActionSheets/>

      </View>
    )}/>
  );

}

// HACK This is a big poop
class PinchScaleX {

  inBase:         Animated.Value;
  inScale:        Animated.Value;
  outScale:       Animated.Value;
  outTranslate:   Animated.Animated;
  transform:      Array<object>;
  onPinchGesture: (...args: Array<any>) => void;

  constructor(
    public readonly spectroBase: WidthHeight<number>,
    public          base:        number,
    public readonly baseClamp:   Clamp<number>,
  ) {
    this.inBase       = new Animated.Value(base);
    this.inScale      = new Animated.Value(1);
    this.outScale     = animated`${Animated.diffClamp(this.inBase, baseClamp.min, baseClamp.max)} * ${this.inScale}`;
    this.outTranslate = this.outScale.interpolate({
      inputRange:  [0, 1],
      outputRange: [-spectroBase.w / 2, 0],
    });
    this.transform = [
      {translateX: this.outTranslate},
      {scaleX: this.outScale},
    ];
    this.onPinchGesture = Animated.event(
      [{nativeEvent: {scale: this.inScale}}],
      {useNativeDriver: config.useNativeDriver},
    );
  }

  onPinchState = (event: Gesture.PinchGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {oldState, scale}} = event;
    if (oldState === Gesture.State.ACTIVE) {
      this.base = _.clamp( // TODO Wat. Why do we have to clamp twice?
        this.base * scale,
        this.baseClamp.min,
        this.baseClamp.max,
      );
      this.inBase.setValue(this.base);
      this.inScale.setValue(1);
    }
  }

}

// HACK This is a slightly less big poop
class PanTranslateX {

  listenTranslationX: number;
  inTranslationX:  Animated.Value;
  outTranslationX: Animated.Animated;
  transform:       Array<object>;
  onPanGesture:    (...args: Array<any>) => void;

  constructor(
    public readonly scale:             Animated.Animated,
    public          translationX:      number,
    public readonly translationXClamp: Clamp<number>, // TODO TODO
  ) {

    this.listenTranslationX = 0;
    this.inTranslationX = new Animated.Value(0);
    this.outTranslationX = (

      // Works (mostly)
      this.inTranslationX

      // TODO TODO Why does this do something funny?
      // animated`${this.inTranslationX} / ${this.scale}`

      // TODO TODO Clamp
      // Animated.diffClamp(this.inTranslationX,
      //   // -500,
      //   -Dimensions.get('window').width * 1.5,
      //   0,
      // )

    );
    this.transform = [
      {translateX: this.outTranslationX},
    ];
    this.onPanGesture = Animated.event(
      [{nativeEvent: {translationX: this.inTranslationX}}],
      {useNativeDriver: config.useNativeDriver},
    );

    // HACK Must call addListener else .{_value,_offset} don't update on the js side
    //  - We rely on ._value below, to workaround a race-condition bug
    this.inTranslationX.addListener(() => {});

    // this._log = (log_f, desc, keys = [], values = []) => { // XXX Debug
    //   log_f(sprintf(
    //     ['%21s :', 'inTranslationX[{_value[%7.2f], _offset[%7.2f]}]', 'listenTranslationX[%7.2f]', 'translationX[%7.2f]',
    //      ...keys].join(' '),
    //     desc, this.inTranslationX._value, this.inTranslationX._offset, this.listenTranslationX, this.translationX, ...values,
    //   ));
    // }

  }

  onPanState = (event: Gesture.PanGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {oldState, translationX, velocityX}} = event;
    if (oldState === Gesture.State.ACTIVE) {

      // log.info('-----------------------------');
      // this._log(log.info, 'onPanState', ['e.translationX[%7.2f]', 'e.velocityX[%7.2f]'], [translationX, velocityX]);

      // Flatten offset -> value so that .decay can use offset for momentum
      //  - {value, offset} -> {value: value+offset, offset: 0}
      this.inTranslationX.flattenOffset();

      // HACK Save ._value for workaround below
      //  - WARNING This only works if we've called .addListener (else it's always 0)
      // @ts-ignore (._value isn't exposed)
      const _valueBefore = this.inTranslationX._value;

      // this._log(log.info, 'onPanState', ['_value[%7.2f]', 'e.translationX[%7.2f]', 'e.velocityX[%7.2f]'], [_value, translationX, velocityX]);

      // Scale velocityX waaaaay down, else it's ludicrous speed [Why? Maybe a unit mismatch?]
      const scaleVelocity = 1/1000;

      Animated.decay(this.inTranslationX, {
        velocity: velocityX * scaleVelocity,
        // deceleration: .997, // Very light, needs twiddling
        deceleration: .98,     // Very heavy, good for debug
        useNativeDriver: config.useNativeDriver,
      }).start(({finished}) => {
        // this._log(log.info, 'decay.finished', ['_value[%7.2f]', 'e.finished[%5s]'], [_value, finished]);

        // Bug: .decay resets ._value to 0 if you swipe multiple times really fast and make multiple .decay's race
        //  - HACK Workaround: if .decay moved us the wrong direction, reset to the _value before .decay
        //  - When you do trip the bug the animation displays incorrectly, but ._value ends up correct
        //  - Without the workaround you'd reset to .value=0 anytime you trip the bug
        if (_valueBefore !== undefined) {
          // @ts-ignore (._value isn't exposed)
          const _valueAfter = this.inTranslationX._value;
          const sgn = Math.sign(velocityX);
          if (sgn * _valueAfter < sgn * _valueBefore) {
            this.inTranslationX.setValue(_valueBefore);
            // this._log(log.info, 'decay.finished', ['_value[%7.2f]', 'e.finished[%5s]'], [_value, finished]);
          }
        }

        // Extract offset <- value now that .decay is done using offset for momentum
        //  - {value, offset} -> {value: 0, offset: value+offset}
        //  - Net effect: (0, offset) -[flatten]> (offset, 0) -[decay]> (offset, momentum) -[extract]> (0, offset+momentum)
        this.inTranslationX.extractOffset();

        // this._log(log.info, 'decay.finished', ['_value[%7.2f]', 'e.finished[%5s]'], [_value, finished]);
      });

    }
  }

}

type Clamp<X> = {
  min: X,
  max: X,
};

type WidthHeight<X> = {
  w: X,
  h: X,
};

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
    width: 14,                       // HACK Manually shrink outer view width to match height of sideways text
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
