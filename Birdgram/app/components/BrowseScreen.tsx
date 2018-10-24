// @ts-ignore
import animated from "animated.macro";
import _ from 'lodash';
import React from 'React';
import { Component, ReactNode, RefObject } from 'react';
import {
  Animated, Dimensions, FlatList, GestureResponderEvent, Image, ImageStyle, Platform, SectionList, SectionListData,
  Text, TextInput, TouchableHighlight, View, WebView,
} from 'react-native';
import RNFB from 'rn-fetch-blob';
import FastImage from 'react-native-fast-image';
import * as Gesture from 'react-native-gesture-handler';
import {
  BorderlessButton, LongPressGestureHandler, PanGestureHandler, PinchGestureHandler, RectButton, TapGestureHandler,
  // FlatList, ScrollView, Slider, Switch, TextInput, // TODO Needed?
} from 'react-native-gesture-handler';
import DrawerLayout from 'react-native-gesture-handler/DrawerLayout';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import KeepAwake from 'react-native-keep-awake';
import SQLite from 'react-native-sqlite-storage';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
const fs = RNFB.fs;

import { config } from '../config';
import { log } from '../log';
import { Places } from '../places';
import Sound from '../sound';
import { querySql } from '../sql';
import { StyleSheet } from '../stylesheet';
import { finallyAsync, global, match } from '../utils';

const SearchRecs = {

  // TODO Test asset paths on android (see notes in README)
  dbPath: 'search_recs/search_recs.sqlite3',

  // TODO After verifying that asset dirs are preserved on android, simplify the basenames back to `${xc_id}.${format}`
  assetPath: (kind: string, species: string, xc_id: number, format: string): string => (
    `search_recs/${kind}/${species}/${kind}-${species}-${xc_id}.${format}`
  ),

};

type Quality = 'A' | 'B' | 'C' | 'D' | 'E' | 'no score';
type RecId = string;

type Rec = {
  id: RecId,
  xc_id: number,
  species: string,             // (From ebird)
  species_taxon_order: string, // (From ebird)
  species_com_name: string,    // (From xc)
  species_sci_name: string,    // (From xc)
  recs_for_sp: number,
  quality: Quality,
  lat: number,
  lng: number,
  month_day: string,
  place: string,
  place_only: string,
  state: string,
  state_only: string,
  recordist: string,
  license_type: string,
}

const Rec = {

  spectroPath: (rec: Rec): string => SearchRecs.assetPath('spectro', rec.species, rec.xc_id, 'png'),
  audioPath:   (rec: Rec): string => SearchRecs.assetPath('audio',   rec.species, rec.xc_id, 'mp4'),

  placeNorm: (rec: Rec): string => {
    return rec.place.split(', ').reverse().map(x => Rec.placePartAbbrev(x)).join(', ');
  },
  placePartAbbrev: (part: string): string => {
    const ret = (
      Places.countryCodeFromName[part] ||
      Places.stateCodeFromName[part] ||
      part
    );
    return ret;
  },

};

type WidthHeight<X> = {
  w: X,
  h: X,
};

type State = {
  totalRecs?: number,
  queryText: string,
  query?: string,
  queryConfig: {
    quality: Array<Quality>,
    limit: number,
  },
  status: string,
  recs: Array<Rec>,
  spectroScaleY: number,
  currentlyPlaying?: {
    rec: Rec,
    sound: Sound,
  }
};

type Props = {
  spectroScaleYMin: number,
  spectroScaleYMax: number,
  spectroBase:      WidthHeight<number>,
};

type SwipeButton = {name: string, backgroundColor: string, onPress: () => void};

class SpectroAnim {

  // outScaleY:      Animated.Value;
  inBase:         Animated.Value;
  inScale:        Animated.Value;
  outScale:       Animated.Value;
  outTranslate:   Animated.Animated;
  transform:      Array<object>;
  onPinchGesture: (...args: Array<any>) => void;

  constructor(
    public readonly spectroBase:    WidthHeight<number>,
    // public          scaleY:         number,
    public          base:           number,
    public readonly baseMin:        number,
    public readonly baseMax:        number,
  ) {
    // This is a big poop
    // this.outScaleY    = new Animated.Value(scaleY);
    this.inBase       = new Animated.Value(base);
    this.inScale      = new Animated.Value(1);
    this.outScale     = animated`${Animated.diffClamp(this.inBase, baseMin, baseMax)} * ${this.inScale}`;
    this.outTranslate = this.outScale.interpolate({
      inputRange:  [0, 1],
      outputRange: [-spectroBase.w / 2, 0],
    });
    this.transform = [
      {translateX: this.outTranslate},
      {scaleX: this.outScale},
      // {scaleY: this.outScaleY},
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
        this.baseMin,
        this.baseMax,
      );
      this.inBase.setValue(this.base);
      this.inScale.setValue(1);
    }
  }

  // onZoomY = (step: number) => {
  //   this.scaleY = _.clamp(
  //     this.scaleY * Math.pow(2, step),
  //     1,
  //     8,
  //   );
  //   this.outScaleY.setValue(this.scaleY);
  // }

}

export class BrowseScreen extends Component<Props, State> {

  static defaultProps = {
    spectroBase: {
      w: Dimensions.get('window').width,
      h: 25,
    },
    spectroScaleYMin: 1,
    spectroScaleYMax: 8,
  };

  db?: SQLiteDatabase;
  soundsCache: Map<RecId, Promise<Sound> | Sound>;
  drawerLayoutRef: RefObject<DrawerLayout>;
  spectroAnim: SpectroAnim;

  constructor(props: Props) {
    super(props);

    this.state = {
      queryText: '',
      queryConfig: { // TODO Move to (global) SettingsScreen.state
        quality: ['A', 'B'],
        limit: 100,
      },
      status: '',
      recs: [],
      spectroScaleY: 2,
    };

    this.soundsCache = new Map();
    this.drawerLayoutRef = React.createRef();
    this.spectroAnim = new SpectroAnim(this.props.spectroBase, 2, 1, 10);

    global.BrowseScreen = this; // XXX Debug
  }

  componentDidMount = async () => {
    log.debug('BrowseScreen.componentDidMount');

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
    log.debug('BrowseScreen.componentWillUnmount');

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

  onPress = (rec: Rec) => {

    // Eagerly allocate Sound resource for rec
    //  - TODO How eagerly should we cache this? What are the cpu/mem costs and tradeoffs?
    const soundAsync = this.getOrAllocateSoundAsync(rec);

    return async (pointerInside: boolean) => {
      log.debug('onPress');
      log.debug('rec', rec);
      log.debug('this.state.currentlyPlaying', this.state.currentlyPlaying);

      // FIXME Races? Tap a lot of spectros really quickly and watch the "Playing rec" logs pile up

      const {currentlyPlaying} = this.state;

      // Stop any recs that are currently playing
      if (currentlyPlaying) {
        const {rec, sound} = currentlyPlaying;

        // Stop sound playback
        log.debug('Stopping currentlyPlaying rec', rec.id);
        this.setState({
          currentlyPlaying: undefined,
        });
        await sound.stopAsync();

        global.sound = sound; // XXX Debug

      }

      // If touched rec was the currently playing rec, then we're done (it's stopped)
      // Else, play the (new) touched rec
      if (!currentlyPlaying || currentlyPlaying.rec.id !== rec.id) {

        const sound = await soundAsync;

        global.sound = sound; // XXX Debug

        // Play rec
        log.debug('Playing rec', rec.id);
        this.setState({
          currentlyPlaying: {rec, sound},
        });
        finallyAsync(sound.playAsync(), () => {
          // Promise fulfills after playback completes / is stopped / fails
          log.debug('Done playing rec', rec.id);
          this.setState({
            currentlyPlaying: undefined,
          });
        });

      }

      // log.debug('onPress: done');
    };
  }

  onLongPress = (rec: Rec) => async (event: Gesture.LongPressGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {state}} = event; // Unpack SyntheticEvent (before async)
    if (state === Gesture.State.ACTIVE) {
      log.debug('onLongPress');
    }
  }

  onMockPress = (rec: Rec) => async () => {
    console.log('renderLeftAction.onMockPress');
  }

  renderLeftActions = (rec: Rec) => {
    return this.renderSwipeButtons(rec, 'left', [
      {name: 'Hide',   backgroundColor: iOSColors.red,    onPress: this.onMockPress(rec)},
      {name: 'Show',   backgroundColor: iOSColors.orange, onPress: this.onMockPress(rec)},
      {name: 'Search', backgroundColor: iOSColors.yellow, onPress: this.onMockPress(rec)},
      {name: 'More',   backgroundColor: iOSColors.green,  onPress: this.onMockPress(rec)},
    ]);
  }

  renderRightActions = (rec: Rec) => {
    return this.renderSwipeButtons(rec, 'right', [
      {name: 'Select',  backgroundColor: iOSColors.blue,   onPress: this.onMockPress(rec)},
      {name: 'Correct', backgroundColor: iOSColors.purple, onPress: this.onMockPress(rec)},
      {name: 'More',    backgroundColor: iOSColors.pink,   onPress: this.onMockPress(rec)},
    ]);
  }

  renderSwipeButtons = (
    rec: Rec,
    side: 'left' | 'right',
    buttons: Array<SwipeButton>,
  ) => (progress: Animated.Value): ReactNode => {

    const width = 200;
    const x = (i: number) => (side === 'right' ?
      width - width / buttons.length * i :
      width / buttons.length * i
    );

    const renderButton = (button: SwipeButton, x: number, key: string): ReactNode => {
      const {name, backgroundColor, onPress} = button;
      const transform = [{translateX: progress.interpolate({
        inputRange:  [0, 1],
        // outputRange: [x, 0],
        outputRange: [width, 0],
      })}];
      return (
        <Animated.View key={key} style={{flex: 1, transform}}>
          <RectButton style={[styles.swipeButton, {backgroundColor}]} onPress={onPress}>
            <Text style={styles.swipeButtonText}>{name}</Text>
          </RectButton>
        </Animated.View>
      );
    };

    return (
      <View style={{width, flexDirection: 'row'}}>
        {buttons.map((button, i) => renderButton(button, x(i), i.toString()))}
      </View>
    );

  }

  renderDrawer = (): ReactNode => {
    return (
      <View style={styles.drawer}>
        <Text>The drawer</Text>
      </View>
    );
  }

  openDrawer = () => {
    this.drawerLayoutRef.current && this.drawerLayoutRef.current.openDrawer();
  }

  zoomSpectroHeight = (step: number) => {
    // TODO TODO Hook up buttons to zoom spectro height
    // this.spectroAnim.onZoomY(step);
    this.setState((state, props) => ({
      spectroScaleY: _.clamp(
        // state.spectroScaleY * Math.pow(2, step),
        state.spectroScaleY + step,
        props.spectroScaleYMin,
        props.spectroScaleYMax,
      ),
    }));
  }

  // TODO Disable when spectroScaleY is min/max
  TopControlsButton = (props: {name: string, onPress: () => void}) => (
    <BorderlessButton onPress={props.onPress}>
      <FontAwesome5 style={styles.topControlsIcon} name={props.name} />
    </BorderlessButton>
  )

  render = () => {
    return (
      <View style={styles.containerOutsideDrawer}>
        <DrawerLayout
          ref={this.drawerLayoutRef}
          drawerWidth={250}
          drawerPosition='left'
          drawerType='slide'
          drawerBackgroundColor={iOSColors.gray}
          overlayColor='rgba(0,0,0,0)'
          renderNavigationView={this.renderDrawer}
        >
          <PinchGestureHandler onGestureEvent={this.spectroAnim.onPinchGesture} onHandlerStateChange={this.spectroAnim.onPinchState}>
            <Animated.View style={styles.containerInsideDrawer}>

              {__DEV__ && <KeepAwake/>}

              <View style={styles.topControls}>
                <this.TopControlsButton name='bars'  onPress={this.openDrawer} />
                <this.TopControlsButton name='minus' onPress={() => this.zoomSpectroHeight(-1)} />
                <this.TopControlsButton name='plus'  onPress={() => this.zoomSpectroHeight(+1)} />
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
              </View>

              <Text style={styles.summaryText}>{this.state.status} ({this.state.totalRecs || '?'} total)</Text>
              <Text style={styles.summaryText}>{JSON.stringify(this.state.queryConfig)}</Text>

              <SectionList
                style={styles.recList}
                sections={sectionsForRecs(this.state.recs)}
                keyExtractor={(rec, index) => rec.id.toString()}
                initialNumToRender={20}
                renderSectionHeader={({section: {species_com_name, species_sci_name, recs_for_sp}}) => (
                  <Text style={styles.recSectionHeader}>{species_com_name} ({recs_for_sp} total recs)</Text>
                )}
                renderItem={({item: rec, index}) => (
                  <View style={styles.recRow}>

                    <Swipeable
                      renderLeftActions={this.renderLeftActions(rec)}
                      renderRightActions={this.renderRightActions(rec)}
                      // @ts-ignore [TODO PR react-native-gesture-handler/react-native-gesture-handler.d.ts]
                      waitFor={this.drawerLayoutRef}
                    >
                      <LongPressGestureHandler onHandlerStateChange={this.onLongPress(rec)}>
                        <BorderlessButton onPress={this.onPress(rec)}>

                          <Animated.View collapsable={false}>
                            <Animated.Image
                              style={[{
                                width:     this.spectroAnim.spectroBase.w,
                                height:    this.spectroAnim.spectroBase.h * this.state.spectroScaleY,
                                transform: this.spectroAnim.transform,
                              }]}
                              resizeMode='stretch'
                              source={{uri: Rec.spectroPath(rec)}}
                            />
                          </Animated.View>

                        </BorderlessButton>
                      </LongPressGestureHandler>

                      <View style={styles.recCaption}>
                        <RecText flex={3}>{rec.xc_id}</RecText>
                        <RecText flex={1}>{rec.quality}</RecText>
                        <RecText flex={2}>{rec.month_day}</RecText>
                        <RecText flex={4}>{Rec.placeNorm(rec)}</RecText>
                        {ccIcon({style: styles.recTextFont})}
                        <RecText flex={4}> {rec.recordist}</RecText>
                      </View>

                    </Swipeable>

                  </View>
                )}
              />

            </Animated.View>
          </PinchGestureHandler>
        </DrawerLayout>
      </View>
    );
  }

}

function sectionsForRecs(recs: Array<Rec>): Array<SectionListData<Rec>> {
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

function RecText<X extends {children: any, flex?: number}>(props: X) {
  const flex = props.flex || 1;
  return (<Text
    style={[styles.recText, {flex}]}
    numberOfLines={1}
    ellipsizeMode='tail'
    {...props}
  />);
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
  containerOutsideDrawer: {
    flex: 1,
  },
  drawer: {
    paddingTop: 20, // TODO How to not overlap with ios status bar?
  },
  containerInsideDrawer: {
    flex: 1,
    paddingTop: 20, // TODO How to not overlap with ios status bar?
  },
  topControls: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  topControlsIcon: {
    ...material.headlineObject,
    paddingHorizontal: 15,
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
  recSectionHeader: {
    // ...material.body1Object, backgroundColor: iOSColors.customGray, // Black on white
    ...material.body1WhiteObject, backgroundColor: iOSColors.gray, // White on black
  },
  recRow: {
    borderWidth: 1, borderColor: 'gray',
    flex: 1, flexDirection: 'column',
  },
  recCaption: {
    flex: 2, flexDirection: 'row', // TODO Eh...
  },
  recText: {
    ...material.captionObject,
  },
  recTextFont: {
    ...material.captionObject,
  },
  recSpectro: {
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
