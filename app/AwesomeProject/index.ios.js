/**
 * @flow
 */

import React, { Component } from 'react';
import {
  AppRegistry,
  StyleSheet,
  Text,
  TouchableHighlight,
  View,
} from 'react-native';

type AwesomeProjectProps = {
};

export default class AwesomeProject extends Component {

  state: {
    t: Date,
  };

  constructor(props: AwesomeProjectProps) {
    super(props);
    this.state = {t: new Date()};
  }

  render() {
    return (
      <View style={styles.container}>
        {
          // TODO See README
          //<Canvas id="viewport" height="512" width="700" />
        }
        <Instruct style={styles.welcome}>
          Welcome to React Native! ({this.state.t.toISOString()})
        </Instruct>
        <Instruct color="red">
          To get started, edit index.ios.js {'\n'}
          foo bar foo
        </Instruct>
        <Instruct color="blue">
          Press Cmd+R to reload,{'\n'}
          Cmd+D or shake for dev menu
        </Instruct>
        <TouchableHighlight style={styles.button} onPress={() => this.onPress()}>
          <Text>
            Go
          </Text>
        </TouchableHighlight>
      </View>
    );
  }

  onPress() {
    console.log('go!');
    this.setState({t: new Date()});
  }

}

class Instruct extends Component {
  render() {
    return (
      <Text style={StyleSheet.flatten([styles.instructions, this.props.style, {color: this.props.color}])}>
        {this.props.text}
        {this.props.children}
      </Text>
    )
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  welcome: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  instructions: {
    textAlign: 'center',
    color: '#333333',
    marginBottom: 5,
  },
  button: {
  },
});

AppRegistry.registerComponent('AwesomeProject', () => AwesomeProject);
