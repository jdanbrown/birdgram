import _ from 'lodash';
import React, { ReactNode } from 'react';
import { Linking, Text, TextProps } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';

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

// TODO Element -> ReactNode
export function CCIcon(props?: object): Element {
  const [icon] = LicenseTypeIcons('cc', props);
  return icon;
}

// TODO Element -> ReactNode
// TODO Make into a proper component (push licenseType into props)
export function LicenseTypeIcons(licenseType: string, props?: object): Array<Element> {
  licenseType = `cc-${licenseType}`;
  return licenseType.split('-').map(k => (<FontAwesome5
    key={k}
    name={k === 'cc' ? 'creative-commons' : `creative-commons-${k}`}
    {...props}
  />));
}
