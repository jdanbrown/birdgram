import React, { ReactNode } from 'react';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';

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
