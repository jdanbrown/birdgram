import React from 'React';
import { Component } from 'react';
import { Dimensions, FlatList, Image, Platform, StyleSheet, Text, TextInput, View, WebView } from 'react-native';
import KeepAwake from 'react-native-keep-awake';
import SQLite from 'react-native-sqlite-storage';
import { SQLiteDatabase } from 'react-native-sqlite-storage';

import { global, querySql } from '../utils';

// TODO Test asset paths on android
const dbFilename = 'foo.db';

type Row = {
  xc_id: number,
  month_day: string,
  state_only: string,
  recordist: string,
  license_type: string,
}

type State = {
  query: string,
  rows: Row[],
};

type Props = {};

export class BrowseScreen extends Component<Props, State> {

  db?: SQLiteDatabase

  constructor(props: Props) {
    super(props);
    this.state = {
      query: '',
      rows: [],
    };
  }

  componentDidMount = async () => {
    console.log('componentDidMount', this);

    // Open db conn
    const dbPath = `~/${dbFilename}`; // Relative to app bundle (copied into the bundle root by react-native-asset)
    this.db = await SQLite.openDatabase({
      name: dbFilename,           // Just for SQLite bookkeeping, I think
      readOnly: true,             // Else it will copy the (huge!) db file from the app bundle to the documents dir
      createFromLocation: dbPath, // Else readOnly will silently not work
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

      // TODO sql params instead of raw string subst
      console.log('query', query);
      await querySql(this.db!, `
        select *
        from search_recs
        where species = upper(trim(?))
        limit 100
      `, [
        query,
      ])(results => {
        const rows = results.rows.raw();
        console.log('rows.length', rows.length);
        rows.slice(0, 3).forEach((row, i) => {
          console.log(`row[${i}]`, row);
        });
        this.setState({
          rows: rows as unknown as Row[],
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
        />

        <FlatList
          style={styles.resultsList}
          data={this.state.rows}
          keyExtractor={(item, index) => item.xc_id.toString()}
          renderItem={({item, index}) => (
            <View style={styles.resultsRow}>
              <View style={styles.resultsUpper}>
                <Text style={styles.resultsCell}>{item.xc_id}</Text>
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
