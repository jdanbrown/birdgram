import React from 'React';
import { Component } from 'react';
import {
  Dimensions, FlatList, GestureResponderEvent, Image, ImageStyle, Platform, StyleSheet, Text, TextInput,
  TouchableHighlight, View, WebView,
} from 'react-native';
import RNFB from 'rn-fetch-blob';
import KeepAwake from 'react-native-keep-awake';
import SQLite from 'react-native-sqlite-storage';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
const fs = RNFB.fs;

import { log } from '../log';
import Sound from '../sound';
import { querySql } from '../sql';
import { finallyAsync, global } from '../utils';

const SearchRecs = {

  // TODO Test asset paths on android (see notes in README)
  dbPath: 'search_recs/search_recs.sqlite3',

  // TODO After verifying that asset dirs are preserved on android, simplify the basenames back to `${xc_id}.${format}`
  assetPath: (kind: string, species: string, xc_id: number, format: string): string => (
    `search_recs/${kind}/${species}/${kind}-${species}-${xc_id}.${format}`
  ),

};

type Quality = 'A' | 'B' | 'C' | 'D' | 'E' | 'no score'

type Rec = {
  id: string,
  xc_id: number,
  species: string,
  species_com_name: string,
  species_sci_name: string,
  quality: Quality,
  lat: number,
  lng: number,
  month_day: string,
  state_only: string,
  recordist: string,
  license_type: string,
}

const Rec = {
  spectroPath: (rec: Rec): string => SearchRecs.assetPath('spectro', rec.species, rec.xc_id, 'png'),
  audioPath:   (rec: Rec): string => SearchRecs.assetPath('audio',   rec.species, rec.xc_id, 'mp4'),
};

type State = {
  totalRecs?: number,
  query: string,
  queryConfig: {
    quality: Array<Quality>,
    limit: number,
  },
  status: string,
  recs: Array<Rec>,
  currentlyPlaying?: {
    rec: Rec,
    sound: Sound,
  }
};

type Props = {};

export class BrowseScreen extends Component<Props, State> {

  db?: SQLiteDatabase

  constructor(props: Props) {
    super(props);

    this.state = {
      query: '',
      queryConfig: { // TODO Move to (global) SettingsScreen.state
        quality: ['A', 'B'],
        limit: 100,
      },
      status: '',
      recs: [],
    };

    global.BrowseScreen = this; // XXX Debug

  }

  componentDidMount = async () => {
    log.debug('componentDidMount', this);

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
    this.editQuery('SOSP');
    this.updateQueryResults();

  }

  editQuery = (query: string) => {
    this.setState({
      query,
    });
  }

  updateQueryResults = async () => {
    if (this.state.query) {
      const query = this.state.query;

      // Clear previous results
      this.setState({
        recs: [],
        status: '[Loading...]',
      });

      // TODO Add param: quality
      log.debug('query', query);
      await querySql<Rec>(this.db!, `
        select *
        from search_recs
        where
          species = upper(trim(?)) and
          quality in (?) and
          true
        limit ?
      `, [
        query,
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

  onTouch = (rec: Rec) => async (event: GestureResponderEvent) => {
    log.debug('onTouch');
    log.debug('rec', rec);
    log.debug('this.state.currentlyPlaying', this.state.currentlyPlaying);

    // Configure react-native-sound
    //  - QUESTION What "playback mode" / "audio session mode" do we actually want?
    //  - https://github.com/zmxv/react-native-sound/wiki/API#soundsetcategoryvalue-mixwithothers-ios-only
    //  - https://apple.co/2q2osEd
    //  - https://developer.apple.com/documentation/avfoundation/avaudiosession/audio_session_modes
    //  - https://apple.co/2R22tcg
    Sound.setCategory(
      'Playback', // Enable playback in silence mode [cargo-culted from README]
      true,       // mixWithOthers
    );
    //  - https://developer.apple.com/documentation/avfoundation/avaudiosession/audio_session_modes
    Sound.setMode(
      'Default', // "The default audio session mode"
    );

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

      // Tell other apps we're using the audio device
      Sound.setActive(true);

      // Allocate Sound
      //  - QUESTION How heavy is this? Can we do it more eagerly / more of them upfront?
      const sound = await Sound.newAsync(
        Rec.audioPath(rec),
        Sound.MAIN_BUNDLE,
      );

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

        // Tell other apps we're no longer using the audio device
        Sound.setActive(false);

        // Release resources
        //  - QUESTION Retain resources for more responsive playback on next touch?
        sound.release();

      });

    }

    log.debug('onTouch: done');
  }

  render = () => {
    return (
      <View style={styles.container}>

        {__DEV__ && <KeepAwake/>}

        <TextInput
          style={styles.queryInput}
          value={this.state.query}
          onChangeText={this.editQuery}
          onSubmitEditing={this.updateQueryResults}
          autoCorrect={false}
          autoCapitalize={'characters'}
          enablesReturnKeyAutomatically={true}
          placeholder={'Species'}
          returnKeyType={'search'}
        />

        <Text>{this.state.status} ({this.state.totalRecs || '?'} total)</Text>
        <Text>{JSON.stringify(this.state.queryConfig)}</Text>

        <FlatList
          style={styles.resultsList}
          data={this.state.recs}
          keyExtractor={(rec, index) => rec.xc_id.toString()}
          renderItem={({item: rec, index}) => (
            <View style={styles.resultsRow}>

              <TouchableHighlight onPress={this.onTouch(rec)}>
                <Image style={styles.resultsRecSpectro as ImageStyle /* HACK Avoid weird type error */}
                  source={{uri: Rec.spectroPath(rec)}}
                />
              </TouchableHighlight>

              <View style={styles.resultsUpper}>
                <Text style={styles.resultsCell}>{index + 1}</Text>
                <Text style={styles.resultsCell}>{rec.xc_id}</Text>
              </View>
              <View style={styles.resultsLower}>
                <Text style={styles.resultsCell}>{rec.species_com_name}</Text>
                <Text style={styles.resultsCell}>{rec.quality}</Text>
              </View>
              <View style={styles.resultsLower}>
                <Text style={styles.resultsCell}>{rec.month_day}</Text>
                <Text style={styles.resultsCell}>{rec.state_only.slice(0, 15)}</Text>
              </View>
              <View style={styles.resultsLower}>
                <Text style={styles.resultsCell}>{rec.recordist}</Text>
                <Text style={styles.resultsCell}>{rec.license_type}</Text>
              </View>

            </View>
          )}
        />

      </View>
    );
  }

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: Dimensions.get('window').width,
  },
  queryInput: {
    marginTop: 20, // HACK ios status bar
    // width: Dimensions.get('window').width,
    borderWidth: 1, borderColor: 'gray',
    fontSize: 30,
    width: Dimensions.get('window').width, // TODO flex
  },
  resultsList: {
    // borderWidth: 1, borderColor: 'gray',
    width: Dimensions.get('window').width, // TODO flex
  },
  resultsRow: {
    borderWidth: 1, borderColor: 'gray',
    flex: 1, flexDirection: 'column',
  },
  resultsUpper: {
    flex: 2, flexDirection: 'row', // TODO Eh...
  },
  resultsLower: {
    flex: 1, flexDirection: 'row',
  },
  resultsCell: {
    flex: 1,
  },
  resultsRecSpectro: {
    width: Dimensions.get('window').width!, // TODO flex
    height: 50,
    resizeMode: 'stretch',
  },
});
