import React, { Component } from 'react';
import {
  Alert, AsyncStorage, Dimensions, Easing, Image, Modal, Platform, ScrollView, Text, TouchableHighlight, View, WebView,
} from 'react-native';
import SettingsList from 'react-native-settings-list';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { BorderlessButton, RectButton } from 'react-native-gesture-handler';

import { Settings } from './Settings';
import { StyleSheet } from '../stylesheet';
import { global } from '../utils';

type Props = {};
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

  // TODO https://github.com/evetstech/react-native-settings-list#a-more-realistic-example
  render = () => (
    <Settings.Context.Consumer children={settings => (
      <View style={styles.container}>

        <View style={styles.settingsList}>
          <SettingsList>

            <SettingsList.Item
              title='Allow uploads'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={settings.allowUploads}
              switchOnValueChange={async x => await settings.set('allowUploads', x)}
            />

            <SettingsList.Item
              title='Test modal'
              onPress={() => this.setState({showModal: true})}
            />

            {/* FIXME Well, this is a pretty horrible UX. Looks like we'll need to redo react-native-settings-list ourselves. */}
            <SettingsList.Item
              isEditable={true}
              hasNavArrow={false}
              id='Debug text color'
              title='Debug text color'
              value={settings.debugTextColor}
              onTextChange={async x => await settings.set('debugTextColor', x)}
            />

            <SettingsList.Item
              title='Show debug info'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={settings.showDebug}
              switchOnValueChange={async x => await settings.set('showDebug', x)}
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

        {settings.showDebug && (
          <View style={settings.debugView}>
            <Text style={settings.debugText}>DEBUG INFO</Text>
          </View>
        )}

      </View>
    )}/>
  );

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'column',
  },
  settingsList: {
    flexGrow: 1,
  },
});
