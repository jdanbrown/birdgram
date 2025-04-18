import _ from 'lodash';
import React, { PureComponent, ReactNode } from 'react';
import { Linking, Text, TextProps, View, ViewProps } from 'react-native';
import {
  BaseButton, BorderlessButton, BorderlessButtonProperties, LongPressGestureHandler, PanGestureHandler,
  PinchGestureHandler, RectButton, TapGestureHandler,
  // FlatList, ScrollView, Slider, Switch, TextInput, // TODO Needed?
} from 'react-native-gesture-handler';
import { SafeAreaContext } from 'react-native-safe-area-context';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';

import * as Colors from 'app/colors';
import { debug_print, Log, logErrors, logErrorsAsync, rich, puts, tap } from 'app/log';
import { SettingsWrites } from 'app/settings';
import { Styles } from 'app/styles';
import {
  enumerate, global, into, json, local, mapNil, mapNull, mapUndefined, match, matchNull,
  matchUndefined, mergeArraysWith, objectKeysTyped, pretty, shallowDiffPropsState, showDate,
  showDateNoTime, showTime, throw_, typed, yaml,
} from 'app/utils';

//
// TitleBar
//

export function TitleBar(props: {
  title: string,
  left?: ReactNode,
  right?: ReactNode,
  viewProps?: ViewProps,
  textProps?: TextProps,
}) {
  return (
    <View style={{
      flexDirection:     'row',
      alignItems:        'center',
      justifyContent:    'space-between',
      width:             '100%',
      backgroundColor:   Styles.tabBar.backgroundColor,
      borderBottomWidth: Styles.tabBar.borderTopWidth,
      borderBottomColor: Styles.tabBar.borderTopColor,
      ...props.viewProps,
    }}>

      {/* .left */}
      <View style={{
        // backgroundColor: 'green', // XXX Debug
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'flex-start', // Against left side
      }}>
        {props.left}
      </View>

      {/* .title */}
      <View style={{
        // backgroundColor: 'yellow', // XXX Debug
        flex: 5,
        alignItems: 'center', // (horizontal)
        justifyContent: 'center',
        paddingTop: 5,
        paddingBottom: 5,
      }}>
        <Text style={{
          // borderWidth: 1, borderColor: 'red', // XXX Debug
          // ...material.titleObject, // XXX Too big (e.g. in SearchScreen)
          ...material.subheadingObject, fontWeight: '600', // (Override 'normal') Between 'normal' (400) and 'bold' (700)
          // fontSize: 16, // XXX Not helpful
          ...props.textProps,
        }}>
          {props.title}
        </Text>
      </View>

      {/* .right */}
      <View style={{
        // backgroundColor: 'green', // XXX Debug
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'flex-end', // Against right side
      }}>
        {props.right}
      </View>

    </View>
  );
}

//
// TitleBarWithHelp
//

interface TitleBarWithHelpProps {
  title: string,
  settings: SettingsWrites,
  showHelp: boolean,
  help: ReactNode,
}

interface TitleBarWithHelpState {
}

export class TitleBarWithHelp extends PureComponent<TitleBarWithHelpProps, TitleBarWithHelpState> {

  log = new Log('TitleBarWithHelp');

  state = {
  };

  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
    this.log.info('componentDidMount');
  });

  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    this.log.info('componentWillUnmount');
  });

  componentDidUpdate = async (
    prevProps: TitleBarWithHelpProps,
    prevState: TitleBarWithHelpState,
  ) => logErrorsAsync('componentDidUpdate', async () => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  });

  render = () => (
    <View style={{
      flexDirection: 'column',
    }}>
      <TitleBar
        title={this.props.title}
        right={(
          this.props.help && ( // Hide help button if there's no help to show
            <BorderlessButton
              style={{
                flex: 1,
                paddingHorizontal: 10,
                justifyContent: 'center',
              }}
              onPress={() => {
                this.props.settings.toggle('showHelp');
              }}
            >
              <Feather
                style={{
                  ...material.buttonObject,
                  fontSize: 20,
                  ...(!this.props.showHelp ? {} : {
                    color: Styles.help.color,
                    // TODO Make this a backgroundColor like the blue/red/yellow buttons on BrowseScreen
                    //  - Will require some style twiddling in TitleBar
                  }),
                }}
                name='help-circle'
              />
            </BorderlessButton>
          )
        )}
      />
      {this.props.showHelp && this.props.help}
    </View>
  );

}

//
// HelpText
//

interface HelpTextProps {
  children: ReactNode;
}

interface HelpTextState {
}

export class HelpText extends PureComponent<HelpTextProps, HelpTextState> {

  log = new Log('HelpText');

  state = {
  };

  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
    this.log.info('componentDidMount');
  });

  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    this.log.info('componentWillUnmount');
  });

  componentDidUpdate = async (
    prevProps: HelpTextProps,
    prevState: HelpTextState,
  ) => logErrorsAsync('componentDidUpdate', async () => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  });

  render = () => (
    <Text style={{
      padding: 10,
      // borderTopWidth: 1, borderBottomWidth: 1, // XXX Not supported for Text [https://github.com/facebook/react-native/issues/29]
      borderWidth: 3,
      borderColor: Styles.help.color,
      backgroundColor: Styles.help.backgroundColor,
    }}>
      {/* <Text style={{fontWeight: 'bold'}}>Help</Text>{'\n'} */}
      {this.props.children}
    </Text>
  );

}
