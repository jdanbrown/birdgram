import { Dimensions } from 'react-native';

// TODO Switch to a more featureful library
//  - https://github.com/yamill/react-native-orientation -- defunct?
//  - https://github.com/wonday/react-native-orientation-locker -- reliable?

export type Orientation = 'portrait' | 'landscape';
export function matchOrientation<X>(orientation: Orientation, cases: {
  portrait:  (orientation: 'portrait')  => X,
  landscape: (orientation: 'landscape') => X,
}): X {
  switch (orientation) {
    case 'portrait':  return cases.portrait(orientation);
    case 'landscape': return cases.landscape(orientation);
  }
}

export function getOrientation(): Orientation {
  const {width, height} = Dimensions.get('screen');
  return width < height ? 'portrait' : 'landscape';
}

// Based on https://shellmonger.com/2017/07/26/handling-orientation-changes-in-react-native/
export function isTablet(): boolean {
  const {width, height, scale} = Dimensions.get('screen');

  // [What's all this about?]
  const limit = scale < 2 ? 1000 : 1900;
  return scale * width >= limit || scale * height >= limit;

};
