import React from 'React';
import { Component } from 'react';
import { Alert, Dimensions, Image, Platform, StyleSheet, Text, View, WebView } from 'react-native';
import KeepAwake from 'react-native-keep-awake';
import SettingsList from 'react-native-settings-list';

import { global } from '../utils';

type Props = {};

// TODO Share this globally
type State = {
  thing: boolean,
};

export class SettingsScreen extends Component<Props, State> {

  constructor(props: Props) {
    super(props);
    this.state = {
      thing: false,
    };
  }

  render = () => {
    // TODO https://github.com/evetstech/react-native-settings-list#a-more-realistic-example
    return (
      <View style={styles.container}>
        {__DEV__ && <KeepAwake/>}
        <SettingsList>

          <SettingsList.Item
            title='Subthings'
            onPress={() => Alert.alert('Need some subthings')}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

          <SettingsList.Item
            title='Thing'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.state.thing}
            switchOnValueChange={x => this.setState({thing: x})}
          />

        </SettingsList>
      </View>
    );
  }

}

const styles = StyleSheet.create({
  container: {
    marginTop: 20, // TODO -> https://github.com/ovr/react-native-status-bar-height
  },
});
