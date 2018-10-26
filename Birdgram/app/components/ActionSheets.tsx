// Utils for https://github.com/beefe/react-native-actionsheet

import React, { Component, Ref, RefObject } from 'react';

// Careful how you import, else mysterious "Element type is invalid" errors (ugh!)
//  - https://github.com/react-navigation/react-navigation/issues/1374#issuecomment-299183535
// import { ActionSheet } from 'react-native-actionsheet'; // XXX Bad [no idea why]
import ActionSheet from 'react-native-actionsheet'; // Good

type ActionSheetBasicProps = {
  title?: string,
  message?: string,
  innerRef: RefObject<ActionSheet>;
  options: Array<[string, () => void]>;
};

export function ActionSheetBasic(props: ActionSheetBasicProps) {
  const ks = props.options.map(([k, f]) => k);
  const fs = props.options.map(([k, f]) => f);
  return (
    <ActionSheet
      ref={props.innerRef}
      title={props.title}
      message={props.message}
      options={[...ks, 'Cancel']}
      cancelButtonIndex={ks.length}
      onPress={(i: number) => i < ks.length && fs[i]()}
    />
  );
}
