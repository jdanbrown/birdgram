import _ from 'lodash';
import React, { PureComponent } from 'react';
import {
  Alert, AsyncStorage, Dimensions, Easing, Image, Modal, Platform, ScrollView, Text, TouchableHighlight, View, WebView,
} from 'react-native';
import { BorderlessButton, RectButton } from 'react-native-gesture-handler';
import SettingsList from 'react-native-settings-list';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { getStatusBarHeight } from 'react-native-status-bar-height';

import { log } from '../log';
import { SettingsWrites } from '../settings';
import { Styles } from '../styles';
import { StyleSheet } from '../stylesheet';
import { global, json, pretty, shallowDiffPropsState, yaml, yamlPretty } from '../utils';

type Props = {
  // Settings
  settings: SettingsWrites;
  showDebug: boolean;
  allowUploads: boolean;
  playingProgressEnable: boolean;
  playingProgressInterval: number;
};

type State = {
  showModal: boolean;
};

export class SettingsScreen extends PureComponent<Props, State> {

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

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));
  }

  // TODO https://github.com/evetstech/react-native-settings-list#a-more-realistic-example
  render = () => {
    log.info(`${this.constructor.name}.render`);
    return (
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
              switchState={this.props.allowUploads}
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
              switchState={this.props.playingProgressEnable}
              switchOnValueChange={async x => await this.props.settings.set('playingProgressEnable', x)}
            />

            {/* FIXME Horrible UX. I think we'll need to redo react-native-settings-list ourselves... */}
            <SettingsList.Item
              id='Playback progress interval (ms)'
              title='Playback progress interval (ms)'
              isEditable={true}
              hasNavArrow={false}
              value={(this.props.playingProgressInterval || '').toString()}
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
              switchState={this.props.showDebug}
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

        {this.props.showDebug && (
          <View style={Styles.debugView}>
            <Text style={Styles.debugText} children={yamlPretty({
              showDebug: this.props.showDebug,
              // WARNING __DEV__ must be a computed key else it gets replaced with its boolean value [how?] in the
              // Release build (but not the Debug build!), which causes the build to fail, which Xcode only _sometimes_
              // surfaces as a build error, and if it doesn't then you have a Release app that's silently stuck on stale
              // js code even though your Debug app has the latest js code. UGH.
              ['__DEV__']: __DEV__,
            })}/>
          </View>
        )}

      </View>
    );
  }

}

const styles = StyleSheet.create({
});
