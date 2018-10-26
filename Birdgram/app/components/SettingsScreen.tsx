import React, { Component } from 'react';
import { Alert, AsyncStorage, Dimensions, Image, Platform, ScrollView, Text, View, WebView } from 'react-native';
import KeepAwake from 'react-native-keep-awake';
import SettingsList from 'react-native-settings-list';
import { getStatusBarHeight } from 'react-native-status-bar-height';

import { Settings } from './Settings';
import { StyleSheet } from '../stylesheet';
import { global } from '../utils';

type Props = {};
type State = {};

export class SettingsScreen extends Component<Props, State> {

  // TODO https://github.com/evetstech/react-native-settings-list#a-more-realistic-example
  render = () => (
    <Settings.Context.Consumer children={settings => (
      <View style={styles.container}>

        {__DEV__ && <KeepAwake/>}

        <View style={styles.settingsList}>
          <SettingsList>

            <SettingsList.Item
              title='Subthings'
              onPress={() => Alert.alert('Need some subthings')}
            />

            <SettingsList.Item
              title='Show debugging info'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={settings.showDebug}
              switchOnValueChange={async showDebug => {
                await settings.set('showDebug', showDebug);
              }}
            />

            {/* FIXME Well, this is a pretty horrible UX. Looks like we'll need to redo react-native-settings-list ourselves. */}
            <SettingsList.Item
              isEditable={true}
              hasNavArrow={false}
              id='Debug text color'
              title='Debug text color'
              value={settings.debugTextColor}
              onTextChange={async color => {
                await settings.set('debugTextColor', color);
              }}
            />

          </SettingsList>
        </View>

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
    marginTop: getStatusBarHeight(),
  },
  settingsList: {
    flexGrow: 1,
  },
});
