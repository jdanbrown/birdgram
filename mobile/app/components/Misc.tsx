import _ from 'lodash';
import React, { ReactNode } from 'react';
import { Linking, Text, TextProps, View, ViewProps } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';

import { Styles } from 'app/styles';

export function TitleBar(props: {
  title: string,
  viewProps?: ViewProps,
  textProps?: TextProps,
}) {
  return (
    <View style={{
      width:             '100%',
      backgroundColor:   Styles.tabBar.backgroundColor,
      borderBottomWidth: Styles.tabBar.borderTopWidth,
      borderBottomColor: Styles.tabBar.borderTopColor,
      ...props.viewProps,
    }}>
      <Text style={{
        alignSelf: 'center',
        marginTop: 25 - Styles.statusBarHeight,
        marginBottom: 5,
        ...material.subheadingObject,
        fontWeight: '500', // (Override 'normal') Between 'normal' (400) and 'bold' (700)
        ...props.textProps,
      }}>
        {props.title}
      </Text>
    </View>
  );
}

export function Hyperlink(props: {url: null | string} & TextProps) { // TODO Return type?
  const {url} = props; // (Unpack for null type checking)
  return !url ? (
    <Text
      {..._.omit(props, 'url')}
    />
  ) : (
    <Text
      style={{
        color: iOSColors.blue,
        ...(props.style as object), // TODO Handle non-object styles (e.g. Style[])
      }}
      onPress={() => Linking.openURL(url)}
      {..._.omit(props, 'url', 'style')}
    />
  );
}

export function CCIcon(props?: object): ReactNode {
  const [icon] = LicenseTypeIcons('cc', props);
  return icon;
}

// TODO Make into a proper component (push licenseType into props)
export function LicenseTypeIcons(licenseType: string, props?: object): Array<ReactNode> {
  return ['cc', ...licenseType.split('-')].map(k => k && (
    <FontAwesome5
      key={k}
      name={k === 'cc' ? 'creative-commons' : `creative-commons-${k}`}
      {...props}
    />
  ));
}
