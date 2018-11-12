import _ from 'lodash';
import React, { Component } from 'react';
import {
  Alert, AsyncStorage, Dimensions, Easing, Image, Modal, Platform, ScrollView, Text, TouchableHighlight, View, WebView,
} from 'react-native';
import { BorderlessButton, RectButton } from 'react-native-gesture-handler';
import SettingsList from 'react-native-settings-list';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { getStatusBarHeight } from 'react-native-status-bar-height';

import { log } from '../log';
import { Settings } from '../settings';
import { StyleSheet } from '../stylesheet';
import { global } from '../utils';

type Props = {
  settings: Settings;
};

type State = {
  showModal: boolean;
};

export class SettingsScreen extends Component<Props, State> {

  constructor(props: Props) {
    super(props);
    this.state = {
      showModal: false,
    };
  }

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);
  }

  componentWillUnmount = async () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
  }

  // TODO https://github.com/evetstech/react-native-settings-list#a-more-realistic-example
  render = () => (
    <View style={{
      flex: 1,
      backgroundColor: iOSColors.white,
    }}>

      <View style={{
        borderBottomWidth: 1,
        borderColor: iOSColors.midGray,
      }}>
        <Text style={{
          alignSelf: 'center',
          marginTop: 30 - getStatusBarHeight(), // No status bar
          marginBottom: 10,
          ...material.titleObject,
        }}>
          Settings
        </Text>
      </View>

      <View style={{
        // flexGrow: 1, // TODO Do we need flexGrow i/o flex? Used to have flexGrow
        flex: 1,
        backgroundColor: iOSColors.customGray,
      }}>
        <SettingsList
          defaultItemSize={50}
          borderColor={iOSColors.midGray}
        >

          <SettingsList.Header headerStyle={{marginTop: 15}} />

          <SettingsList.Item
            id='Allow uploads'
            title='Allow uploads'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.props.settings.allowUploads}
            switchOnValueChange={async x => await this.props.settings.set('allowUploads', x)}
          />

          <SettingsList.Item
            id='Test modal'
            title='Test modal'
            onPress={() => this.setState({showModal: true})}
          />

          <SettingsList.Item
            id='Playback progress (high cpu)'
            title='Playback progress (high cpu)'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.props.settings.playingProgressEnable}
            switchOnValueChange={async x => await this.props.settings.set('playingProgressEnable', x)}
          />

          {/* FIXME Horrible UX. I think we'll need to redo react-native-settings-list ourselves... */}
          <SettingsList.Item
            id='Playback progress interval (ms)'
            title='Playback progress interval (ms)'
            isEditable={true}
            hasNavArrow={false}
            value={(this.props.settings.playingProgressInterval || '').toString()}
            onTextChange={async str => {
              const x = parseInt(str);
              await this.props.settings.set('playingProgressInterval', _.isNaN(x) ? 0 : x);
            }}
          />

          <SettingsList.Header headerStyle={{marginTop: 15}} />

          <SettingsList.Item
            id='Debug: Show debug info'
            title='Debug: Show debug info'
            hasNavArrow={false}
            hasSwitch={true}
            switchState={this.props.settings.showDebug}
            switchOnValueChange={async x => await this.props.settings.set('showDebug', x)}
          />

        </SettingsList>
      </View>

      <Modal
        animationType='none' // 'none' | 'slide' | 'fade'
        transparent={true}
        visible={this.state.showModal}
      >
        <View style={{
          flex: 1,
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          marginTop: 300,
          marginBottom: 50,
          backgroundColor: iOSColors.midGray,
        }}>
          <View>
            <Text>This is a modal</Text>
            <RectButton onPress={() => this.setState({showModal: !this.state.showModal})}>
              <View style={{padding: 20, backgroundColor: iOSColors.orange}}>
                <Text>Close</Text>
              </View>
            </RectButton>
          </View>
        </View>
      </Modal>

      {this.props.settings.showDebug && (
        <View style={this.props.settings.debugView}>
          <Text style={this.props.settings.debugText}>DEBUG INFO</Text>
        </View>
      )}

    </View>
  );

}

const styles = StyleSheet.create({
});
