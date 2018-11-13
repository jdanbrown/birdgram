import { Location, MemoryHistory } from 'history';
import React, { PureComponent, ReactNode } from 'react';
import { Alert } from 'react-native';
import { MemoryRouterProps } from 'react-router';
import { Route, Router } from 'react-router-native';

import { log } from './log';
import { global, shallowDiffPropsState } from './utils';

//
// RouterWithHistory
//

export interface RouterWithHistoryProps {
  history: MemoryHistory;
}

export interface RouterWithHistoryState {
}

// A decomposed version of NativeRouter (+ MemoryRouter) that exposes its history as a prop for the caller
export class RouterWithHistory extends PureComponent<RouterWithHistoryProps, RouterWithHistoryState> {

  // Like NativeRouter
  static defaultProps = {
    getUserConfirmation: (message: string, callback: (ok: boolean) => void) => {
      Alert.alert("Confirm", message, [
        { text: "Cancel", onPress: () => callback(false) },
        { text: "OK", onPress: () => callback(true) }
      ]);
    }
  };

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);
  }

  componentWillUnmount = async () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
  }

  componentDidUpdate = async (prevProps: RouterWithHistoryProps, prevState: RouterWithHistoryState) => {
    // log.info(`${this.constructor.name}.componentDidUpdate`, shallowDiffPropsState(prevProps, prevState, this.props, this.state));
  }

  // Like MemoryRouter
  render = () => (
    <Router {...this.props} />
  );

}

//
// Utils
//

export function HistoryConsumer(props: {
  children: (props: {location: Location, history: MemoryHistory}) => ReactNode,
}) {
  return (
    <Route children={({location, history}: {location: Location, history: MemoryHistory}) => (
      props.children({location, history})
    )}/>
  );
}

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
export function ObserveHistory(props: {globalKey: string}) {
  return (
    <Route children={({location, history}: {location: Location, history: MemoryHistory}) => {

      // XXX Obviated by App.state.histories
      // if (!global.histories) {
      //   global.histories = {}
      // }
      // global.histories[props.globalKey] = {location, history}; // (Exclude match since it's meaningless)

      // Noisy, but maybe useful later
      // log.info(`histories.${props.globalKey}: location`, json(location));
      // log.debug('histories.Route: history', history.entries.map((entry, i) => {
      //   const current = history.index === i;
      //   return `\n${current ? ' -> ' : '    '}${json(entry)}`;
      // }).join());

      return null;
    }}/>
  );
}
