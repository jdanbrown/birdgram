import _ from 'lodash';
import { TextProps, ViewProps } from 'react-native'
import DeviceInfo from 'react-native-device-info';
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { StyleSheet } from 'app/stylesheet';

const iOSDefaultBackgroundColor = '#f7f7f7'; // Default background color (in iOS 10)

// Generic styles
//  - TODO How to put in a StyleSheet.create without losing type info?
//    - Forced into ViewStyle | TextStyle | ImageStyle, which is too lossy for e.g. TopControlsButtonProps / Feather (icon)
export const Styles = {

  fill: {
    flex: 1,
    width: '100%',
    height: '100%',
  } as ViewProps,
  center: {
    justifyContent: 'center', // Main axis
    alignItems:     'center', // Cross axis (unless wrapped)
  } as ViewProps,
  rotate45:       {transform: [{rotate: '45deg'}]},
  rotate90:       {transform: [{rotate: '90deg'}]},
  rotate180:      {transform: [{rotate: '180deg'}]},
  rotate270:      {transform: [{rotate: '270deg'}]},
  flipHorizontal: {transform: [{scaleX: -1}]},
  flipVertical:   {transform: [{scaleY: -1}]},
  flipBoth:       {transform: [{scaleX: -1}, {scaleY: -1}]},

  app: {
    backgroundColor: iOSDefaultBackgroundColor, // Match app bg to tabBar bg (only matters on notched ios)
  },

  statusBarHeight: (!DeviceInfo.hasNotch()
    ? getStatusBarHeight() // Adjust up b/c we hide the status bar
    : 0                    // HACK Fudge the adjustment on notched ios [why?]
  ),

  // For TabRoutes, and reused by others (e.g. screen headers)
  tabBar: {
    backgroundColor: iOSDefaultBackgroundColor, // Match app bg to tabBar bg (only matters on notched ios)
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(0,0,0,.3)',
  },

  // Debug styles
  debug: {
    color: iOSColors.green,
    backgroundColor: iOSColors.black,
  },
  debugView: {
    padding: 3,
    backgroundColor: iOSColors.black,
  },
  debugText: {
    color: iOSColors.green,
  },

};

// (Not sure about this type)
export function normalizeStyle<X extends {}>(style: undefined | null | X | Array<X>): Array<X> {
  return (
    !style ? [] :
    style instanceof Array ? style :
    [style]
  );
}

//
// Labels
//

export interface LabelStyle {
  color: string;
  backgroundColor: string;
}

// [Autogen]
/*
print('export const labelStyles: Array<LabelStyle> = [')
for color, backgroundColors in [
    ('white', islice(mpl.cm.tab20.colors, 0, None, 2)),
    ('black', islice(mpl.cm.tab20.colors, 1, None, 2)),
    ('black', mpl.cm.Pastel1.colors),
]:
    print()
    for backgroundColor in backgroundColors:
        backgroundColor = mpl.colors.rgb2hex(backgroundColor)
        color = {
            '#bcbd22': 'black',
            '#17becf': 'black',
        }.get(backgroundColor) or color
        print("  {color: '%s', backgroundColor: '%s'}," % (
            color,
            mpl.colors.rgb2hex(backgroundColor),
        ))
print()
print('];')
*/
export const labelStyles: Array<LabelStyle> = [

  {color: 'white', backgroundColor: '#1f77b4'},
  {color: 'white', backgroundColor: '#ff7f0e'},
  {color: 'white', backgroundColor: '#2ca02c'},
  {color: 'white', backgroundColor: '#d62728'},
  {color: 'white', backgroundColor: '#9467bd'},
  {color: 'white', backgroundColor: '#8c564b'},
  {color: 'white', backgroundColor: '#e377c2'},
  {color: 'white', backgroundColor: '#7f7f7f'},
  {color: 'black', backgroundColor: '#bcbd22'},
  {color: 'black', backgroundColor: '#17becf'},

  {color: 'black', backgroundColor: '#aec7e8'},
  {color: 'black', backgroundColor: '#ffbb78'},
  {color: 'black', backgroundColor: '#98df8a'},
  {color: 'black', backgroundColor: '#ff9896'},
  {color: 'black', backgroundColor: '#c5b0d5'},
  {color: 'black', backgroundColor: '#c49c94'},
  {color: 'black', backgroundColor: '#f7b6d2'},
  {color: 'black', backgroundColor: '#c7c7c7'},
  {color: 'black', backgroundColor: '#dbdb8d'},
  {color: 'black', backgroundColor: '#9edae5'},

  {color: 'black', backgroundColor: '#fbb4ae'},
  {color: 'black', backgroundColor: '#b3cde3'},
  {color: 'black', backgroundColor: '#ccebc5'},
  {color: 'black', backgroundColor: '#decbe4'},
  {color: 'black', backgroundColor: '#fed9a6'},
  {color: 'black', backgroundColor: '#ffffcc'},
  {color: 'black', backgroundColor: '#e5d8bd'},
  {color: 'black', backgroundColor: '#fddaec'},
  {color: 'black', backgroundColor: '#f2f2f2'},

];
