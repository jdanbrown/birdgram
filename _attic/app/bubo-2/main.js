// Exported from snack.expo.io
import Expo from 'expo';
import React, { Component } from 'react';
import { Text, View, StyleSheet, ListView } from 'react-native';
import { Constants } from 'expo';

export default class App extends Component {

  constructor() {
    super();
    const ds = new ListView.DataSource({rowHasChanged: (r1, r2) => r1 !== r2});
    this.state = {
      dataSource: ds.cloneWithRows([
        'turkey jay',
        'stellars jey',
        'owls',
        'more owls',
        'balloon bird sp.',
      ]),
    };
  }

  render() {
    console.log('Rendering');
    return (
      <View style={styles.container}>
        <Text style={styles.paragraph}>
          You found some birds:
        </Text>
        <ListView
          dataSource={this.state.dataSource}
          renderRow={(rowData) => <Text>{rowData}</Text>}
        />
      </View>
    );
  }

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: Constants.statusBarHeight,
    backgroundColor: '#ecf0f1',
  },
  paragraph: {
    margin: 24,
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#34495e',
  },
});

Expo.registerRootComponent(App);
