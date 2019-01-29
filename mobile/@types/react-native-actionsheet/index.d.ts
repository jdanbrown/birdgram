// https://github.com/beefe/react-native-actionsheet
declare module 'react-native-actionsheet' {

  import { Component, ReactNode } from 'react';

  type ActionSheetProps = {
    title?: string;
    message?: string;
    options: Array<string | ReactNode>; // TODO Is ReactNode the right type for "PropTypes.element"?
    tintColor?: string;
    cancelButtonIndex?: number;
    destructiveButtonIndex?: number;
    onPress?: (index: number) => void;
    styles?: any; // TODO
  };

  class ActionSheet extends Component<ActionSheetProps> {
    show: () => void;
  }

  export default ActionSheet;

}
