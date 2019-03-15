declare module 'react-native-icon-badge' {

// Copied from https://github.com/yanqiw/react-native-icon-badge/blob/ccca39b/index.d.ts

import * as React from "react";
import { StyleProp, ViewStyle } from "react-native";

export interface IconBadgeProps {
  /**
   * The background element.
   */
  MainElement: JSX.Element;

  /**
   * The badge element, normally it is a Text.
   */
  BadgeElement: JSX.Element;

  /**
   * Customized container (main view) style.
   */
  MainViewStyle?: StyleProp<ViewStyle>;

  /**
   * Customized badge style.
   */
  IconBadgeStyle?: StyleProp<ViewStyle>;

  /**
   * Hides badge.
   * @default false
   */
  Hidden?: boolean;
}

export default class IconBadge extends React.Component<IconBadgeProps> {}

}
