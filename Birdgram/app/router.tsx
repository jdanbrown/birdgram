import _ from 'lodash';
import { Action, createMemoryHistory, Location, MemoryHistory } from 'history';
import React, { PureComponent, ReactNode } from 'react';
import { Alert, AsyncStorage } from 'react-native';
import { MemoryRouterProps } from 'react-router';
import { Route, Router } from 'react-router-native';

import { log } from './log';
import { Settings } from './settings';
import { global, shallowDiffPropsState } from './utils';

//
// Utils
//

export type Go = (tab: TabName, path: string) => void;

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

//
// Histories
//

export interface TabHistories {
  record:   MemoryHistory;
  search:   MemoryHistory;
  recent:   MemoryHistory;
  saved:    MemoryHistory;
  settings: MemoryHistory;
}

export interface Histories extends TabHistories {
  tabs: MemoryHistory;
}

export type TabName = keyof TabHistories;
export type HistoryName = keyof Histories;

export async function saveHistories(histories: Histories): Promise<void> {
  log.debug('router.saveHistories', histories);
  await AsyncStorage.setItem('router.histories', JSON.stringify(histories));
}

export async function loadHistories(): Promise<null | Histories> {
  // FIXME Types: distinguish hydrated Histories vs. unhydrated Record<HistoryName, HistoryNonFunctionProps>
  const x = await Settings._getItemFromJson('router.histories') as null | Histories;
  const histories = !x ? null : (
    // Rehydrate via createMemoryHistory()
    _.mapValues(x, ({
      entries, index,
      // location, action, length, // Ignored
    }) => createMemoryHistory({
      initialIndex: index,
      initialEntries: entries as unknown as string[], // HACK Work around bunk type (string i/o Location)
      // keyLength: ...           // Use default [maybe dangerous if we ever decide to vary keyLength]
      // getUserConfirmation: ... // Use default
    })) as unknown as Histories // HACK Work around bunk type (Dictionary<X> instead of Record<K,V>)
  );
  // log.debug('router.loadHistories', histories);
  return histories;
}

export function createDefaultHistories(): Histories {
  return {
    tabs:     createMemoryHistory(),
    record:   createMemoryHistory(),
    search:   createMemoryHistory(),
    recent:   createMemoryHistory(),
    saved:    createMemoryHistory(),
    settings: createMemoryHistory(),
  };
}

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
