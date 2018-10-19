import React from 'React';
import { Component } from 'react';
import { Dimensions, FlatList, Image, Platform, StyleSheet, Text, TextInput, View, WebView } from 'react-native';
import RNFB from 'rn-fetch-blob';
import KeepAwake from 'react-native-keep-awake';
import SQLite from 'react-native-sqlite-storage';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
const fs = RNFB.fs;

import { global, querySql } from '../utils';

// TODO Test asset paths on android
const dbFilename = 'search_recs/search_recs.sqlite3';

type Rec = {
  xc_id: number,
  species: string,
  species_com_name: string,
  species_sci_name: string,
  lat: number,
  lng: number,
  month_day: string,
  state_only: string,
  recordist: string,
  license_type: string,
}

type State = {
  totalRecs?: number,
  query: string,
  recs: Rec[],
  status: string,
};

type Props = {};

export class BrowseScreen extends Component<Props, State> {

  db?: SQLiteDatabase

  constructor(props: Props) {
    super(props);
    this.state = {
      query: '',
      recs: [],
      status: '',
    };
  }

  componentDidMount = async () => {
    console.log('componentDidMount', this);

    // Open db conn
    const dbExists = await fs.exists(`${fs.dirs.MainBundleDir}/${dbFilename}`);
    if (!dbExists) {
      console.error(`DB file not found: ${dbFilename}`);
    } else {
      const dbPath = `~/${dbFilename}`; // Relative to app bundle (copied into the bundle root by react-native-asset)
      this.db = await SQLite.openDatabase({
        name: dbFilename,           // Just for SQLite bookkeeping, I think
        readOnly: true,             // Else it will copy the (huge!) db file from the app bundle to the documents dir
        createFromLocation: dbPath, // Else readOnly will silently not work
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
        status: 'loading',
      });

      // TODO sql params instead of raw string subst
      console.log('query', query);
      await querySql<Rec>(this.db!, `
        select *
        from search_recs
        where species = upper(trim(?))
        -- limit 100
        limit 1000
      `, [
        query,
      ])(results => {
        const recs = results.rows.raw();
        this.setState({
          recs,
          status: `${recs.length} recs`,
        });
      });

    }
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

        <Text>
          {this.state.status} ({this.state.totalRecs || '?'} total)
        </Text>

        <FlatList
          style={styles.resultsList}
          data={this.state.recs}
          keyExtractor={(item, index) => item.xc_id.toString()}
          renderItem={({item, index}) => (
            <View style={styles.resultsRow}>
              <View style={styles.resultsUpper}>
                <Text style={styles.resultsCell}>{index + 1}</Text>
                <Text style={styles.resultsCell}>{item.xc_id}</Text>
              </View>
              <View style={styles.resultsLower}>
                <Text style={styles.resultsCell}>{item.species_com_name}</Text>
              </View>
              <View style={styles.resultsLower}>
                <Text style={styles.resultsCell}>{item.month_day}</Text>
                <Text style={styles.resultsCell}>{item.state_only.slice(0, 15)}</Text>
              </View>
              <View style={styles.resultsLower}>
                <Text style={styles.resultsCell}>{item.recordist}</Text>
                <Text style={styles.resultsCell}>{item.license_type}</Text>
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
});
