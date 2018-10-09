import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Text
} from 'react-native';
import {
  Header,
  Left,
} from 'native-base';
import { ExpoLinksView } from '@expo/samples';

export default class CompareScreen extends React.Component {
  static navigationOptions = {
    title: 'Compare',
  };

  render() {
    return (
      <Header>
        <Text>
          TODO
        </Text>
      </Header>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 15,
    backgroundColor: '#fff',
  },
});
