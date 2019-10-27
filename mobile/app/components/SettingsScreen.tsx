import _ from 'lodash';
import React, { PureComponent } from 'react';
import {
  Alert, AsyncStorage, Dimensions, Easing, Image, Modal, Platform, ScrollView, Text, TouchableHighlight, View,
} from 'react-native';
import { BorderlessButton, RectButton } from 'react-native-gesture-handler';
import SettingsList from 'react-native-settings-list';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import { getStatusBarHeight } from 'react-native-status-bar-height';

import { config } from 'app/config';
import { ServerConfig } from 'app/datatypes';
import { Log, rich } from 'app/log';
import { DEFAULTS, SettingsWrites } from 'app/settings';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import { global, json, pretty, shallowDiffPropsState, yaml, yamlPretty } from 'app/utils';

const log = new Log('SettingsScreen');

export const refreshRateMin = 1;
export const refreshRateMax = 64;

type Props = {
  // Settings
  serverConfig: ServerConfig;
  settings: SettingsWrites;
  showDebug: boolean;
  allowUploads: boolean;
  maxHistory: number;
  f_bins: number;
  // Geo
  geoHighAccuracy: boolean;
  geoWarnIfNoCoords: boolean;
  // RecordScreen
  refreshRate: number;
  doneSpectroChunkWidth: number
  spectroChunkLimit: number;
  // SearchScreen
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
    log.info('componentDidMount');
  }

  componentWillUnmount = async () => {
    log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  // TODO https://github.com/evetstech/react-native-settings-list#a-more-realistic-example
  render = () => {
    log.info('render');
    return (
      <View style={{
        flex: 1,
      }}>

        <View style={{
          backgroundColor:   Styles.tabBar.backgroundColor,
          borderBottomWidth: Styles.tabBar.borderTopWidth,
          borderBottomColor: Styles.tabBar.borderTopColor,
        }}>
          <Text style={{
            alignSelf: 'center',
            marginTop: Styles.tabBarText.marginTop,
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
              id='Debug: Show debug info'
              title='Debug: Show debug info'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={this.props.showDebug}
              switchOnValueChange={async showDebug => await this.props.settings.set({showDebug})}
            />

            <SettingsList.Header headerStyle={{marginTop: 15}} />

            {/* TODO After we do background uploads */}
            {/* <SettingsList.Item
              id='Allow uploads'
              title='Allow uploads'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={this.props.allowUploads}
              switchOnValueChange={async allowUploads => await this.props.settings.set({allowUploads})}
            /> */}

            <SettingsList.Item
              id='geoHighAccuracy'
              title='geoHighAccuracy (requires app restart)'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={this.props.geoHighAccuracy}
              switchOnValueChange={async geoHighAccuracy => await this.props.settings.set({geoHighAccuracy})}
            />

            <SettingsList.Item
              id='geoWarnIfNoCoords'
              title='geoWarnIfNoCoords'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={this.props.geoWarnIfNoCoords}
              switchOnValueChange={async geoWarnIfNoCoords => await this.props.settings.set({geoWarnIfNoCoords})}
            />

            {/* FIXME Various (small) issues when changing this: */}
            {/* - <Image>'s warn on file not found for intermediate f_bins values (e.g. 80 -> 4 -> 40) */}
            {/* - RecordScreen scales really wacky on f_bins!=80 (e.g. 40 very small, 160 very large) */}
            {/* <SettingsList.Item
              id='f_bins (requires restart)'
              title='f_bins (requires restart)'
              isEditable={true}
              hasNavArrow={false}
              value={(this.props.f_bins || '').toString()}
              onTextChange={async str => {
                const x = parseInt(str);
                const f_bins = _.isNaN(x) ? DEFAULTS.f_bins : x;
                await this.props.settings.set({f_bins});
              }}
            /> */}

            {/* FIXME Horrible UX. I think we'll need to redo react-native-settings-list ourselves... */}
            <SettingsList.Item
              id='maxHistory (0 for unlimited)'
              title='maxHistory (0 for unlimited)'
              isEditable={true}
              hasNavArrow={false}
              value={(this.props.maxHistory || '').toString()}
              onTextChange={async str => {
                const x = parseInt(str);
                const maxHistory = _.isNaN(x) ? DEFAULTS.maxHistory : x;
                await this.props.settings.set({maxHistory});
              }}
            />

            {/* FIXME Horrible UX. I think we'll need to redo react-native-settings-list ourselves... */}
            <SettingsList.Item
              id='Recording spectro refresh rate (/sec)'
              title='Recording spectro refresh rate (/sec)'
              isEditable={true}
              hasNavArrow={false}
              value={(this.props.refreshRate || '').toString()}
              onTextChange={async str => {
                const x = parseInt(str);
                const refreshRate = _.clamp(_.isNaN(x) ? 1 : x, refreshRateMin, refreshRateMax);
                await this.props.settings.set({refreshRate});
              }}
            />

            {/* FIXME Horrible UX. I think we'll need to redo react-native-settings-list ourselves... */}
            <SettingsList.Item
              id='doneSpectroChunkWidth'
              title='doneSpectroChunkWidth'
              isEditable={true}
              hasNavArrow={false}
              value={(this.props.doneSpectroChunkWidth || '').toString()}
              onTextChange={async str => {
                const x = parseInt(str);
                const doneSpectroChunkWidth = _.isNaN(x) ? DEFAULTS.doneSpectroChunkWidth : x;
                await this.props.settings.set({doneSpectroChunkWidth});
              }}
            />

            {/* FIXME Horrible UX. I think we'll need to redo react-native-settings-list ourselves... */}
            <SettingsList.Item
              id='spectroChunkLimit (0 for unlimited)'
              title='spectroChunkLimit (0 for unlimited)'
              isEditable={true}
              hasNavArrow={false}
              value={(this.props.spectroChunkLimit || '').toString()}
              onTextChange={async str => {
                const x = parseInt(str);
                const spectroChunkLimit = _.isNaN(x) ? 1 : x;
                await this.props.settings.set({spectroChunkLimit});
              }}
            />

            <SettingsList.Item
              id='Playback progress (high cpu)'
              title='Playback progress (high cpu)'
              hasNavArrow={false}
              hasSwitch={true}
              switchState={this.props.playingProgressEnable}
              switchOnValueChange={async playingProgressEnable => await this.props.settings.set({playingProgressEnable})}
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
                await this.props.settings.set({playingProgressInterval: _.isNaN(x) ? 0 : x});
              }}
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
              // WARNING __DEV__ must be a computed key else it gets replaced with its boolean value [how?] in the
              // Release build (but not the Debug build!), which causes the build to fail, which Xcode only _sometimes_
              // surfaces as a build error, and if it doesn't then you have a Release app that's silently stuck on stale
              // js code even though your Debug app has the latest js code. UGH.
              ['__DEV__']: __DEV__,
              app: {
                name:      `${config.env.APP_NAME}`,
                bundle:    `${config.env.APP_BUNDLE_ID}`,
                version:   `${config.env.APP_VERSION} (${config.env.APP_VERSION_BUILD})`,
                buildDate: `${config.env.BUILD_DATE}`,
              },
              config: _.omit(config, 'env'),
              // TODO Very tall, show somewhere less disruptive (e.g. a sub-page within Settings)
              //  - And then show all of:
              //    - this.props.serverConfig.server_globals.sg_load.search
              //    - this.props.serverConfig.server_globals.sg_load.xc_meta
              //    - this.props.serverConfig.api.recs.search_recs.params
              payload: {
                ...this.props.serverConfig.server_globals.sg_load.xc_meta,
                ..._.pick(this.props.serverConfig.server_globals.sg_load.xc_meta, ['limit']),
                ..._.pick(this.props.serverConfig.api.recs.search_recs.params, ['audio_s']),
                ..._.pick(this.props.serverConfig.server_globals.sg_load.search, ['cv_str']),
              },
            })}/>
          </View>
        )}

      </View>
    );
  }

}

const styles = StyleSheet.create({
});
