import _ from 'lodash';
import * as _history from 'history';
import React, { PureComponent, ReactNode } from 'react';
import { Alert, AsyncStorage } from 'react-native';
import { MemoryRouterProps } from 'react-router';
import { Route, Router } from 'react-router-native';

import { Log, rich } from './log';
import { Settings } from './settings';
import { global, into, json, local, shallowDiffPropsState, yaml } from './utils';

const log = new Log('router');

//
// Utils
//

export type Go = (tab: TabName, to: GoTo) => void;
export type GoTo = {path?: string, index?: number}; // TODO Make this a proper union type

export function HistoryConsumer(props: {
  children: (props: {location: Location, history: History}) => ReactNode,
}) {
  return (
    <Route children={({location, history}: {location: Location, history: History}) => (
      props.children({location, history})
    )}/>
  );
}

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
export function ObserveHistory(props: {globalKey: string}) {
  return (
    <Route children={({location, history}: {location: Location, history: History}) => {

      // XXX Obviated by App.state.histories
      // if (!global.histories) {
      //   global.histories = {}
      // }
      // global.histories[props.globalKey] = {location, history}; // (Exclude match since it's meaningless)

      // Noisy, but maybe useful later
      // log.info(`histories.${props.globalKey}: location`, yaml(location));
      // log.debug('histories.Route: history', history.entries.map((entry, i) => {
      //   const current = history.index === i;
      //   return `\n${current ? ' -> ' : '    '}${yaml(entry)}`;
      // }).join());

      return null;
    }}/>
  );
}

//
// Histories
//

// Redefine standard history types with our own LocationState
export type Location = _history.Location<LocationState>;
export type History  = _history.MemoryHistory<LocationState>;
export interface LocationState {
  timestamp: Date;
}

export function createHistory(opts: {
  initialEntries?: Array<Location>;
  initialIndex?:   number;
  keyLength?:      number;
} = {}): History {
  return _history.createMemoryHistory({
    initialEntries: opts.initialEntries as unknown as Array<string>, // HACK Work around bunk type (string i/o Location)
    initialIndex:   opts.initialIndex,
    keyLength:      opts.keyLength,
  });
}

export interface TabHistories {
  record:   History;
  search:   History;
  recent:   History;
  saved:    History;
  places:   History;
  settings: History;
  help:     History;
}

// export const tabHistoriesKeys: Array<keyof TabHistories> = [ // XXX Else .includes() complains for string input
export const tabHistoriesKeys: Array<string> = [
  'record',
  'search',
  'recent',
  'saved',
  'places',
  'settings',
  'help',
];

export interface Histories extends TabHistories {
  tabs: History;
}

export type TabName = keyof TabHistories;
export type HistoryName = keyof Histories;

// Prefix keys in AsyncStorage
const _prefix = 'router_v3.'; // Bump version to wipe out storage on incompat code changes
function prefixKey(key: string): string {
  return `${_prefix}${key}`;
};

export async function saveHistories(histories: Histories): Promise<void> {
  log.debug('saveHistories', rich(histories));
  await AsyncStorage.setItem(prefixKey('histories'), JSON.stringify(histories));
}

export async function loadHistories(): Promise<null | Histories> {
  // TODO Eliminate the big unsafe type assertions so that unsound json->data loading can be caught by type errors
  // FIXME Types: distinguish hydrated Histories vs. unhydrated Record<HistoryName, HistoryNonFunctionProps>
  const x = await Settings._getItemFromJson(prefixKey('histories')) as null | Histories;
  var histories: null | Histories = null;
  if (x) {
    // Rehydrate via createHistory()
    histories = (
      _.mapValues(x, ({
        entries, index,
        // location, action, length, // Ignored
      }) => createHistory({
        initialIndex: index,
        initialEntries: (entries as Array<_history.Location<{}>>).map(location => ({
          ...location,
          state: into((location.state || {}) as {[key: string]: any}, state => ({
            // Provide defaults for state fields for back compact (e.g. after code change)
            timestamp: new Date(state.timestamp || '1970'),
          })),
        })),
        // keyLength: ...           // Use default [maybe dangerous if we ever decide to vary keyLength]
        // getUserConfirmation: ... // Use default
      })) as unknown as Histories // HACK Work around bunk type (Dictionary<X> instead of Record<K,V>)
    )
    // Add a fresh createHistory() for any missing keys (e.g. we added a new tab)
    tabHistoriesKeys.forEach(k => {
      if (!(k in histories!)) {
        (histories! as unknown as {[key: string]: History})[k] = createHistory(); // HACK Types
      }
    });
  }
  log.debug('loadHistories', rich(histories));
  return histories;
}

// For debugging, e.g. to throw away malformed history locations after changing a new sourceId format
//  - NOTE Requires app restart after success, to reload histories from AsyncStorage
export async function _dev_trimHistoriesToCurrentIndex(histories: Histories): Promise<void> {
  _.toPairs(histories).forEach(([k, history]) => {
    const before = json(_.pick(history, 'index', 'length'));
    history.entries = history.entries.slice(0, history.index + 1);
    history.length  = history.entries.length;
    const after = json(_.pick(history, 'index', 'length'));
    log.info(`${k}: ${before} -> ${after}`);
  });
  await saveHistories(histories);
}

export function createDefaultHistories(): Histories {
  return {
    tabs:     createHistory(),
    record:   createHistory(),
    search:   createHistory(),
    recent:   createHistory(),
    saved:    createHistory(),
    places:   createHistory(),
    settings: createHistory(),
    help:     createHistory(),
  };
}

//
// RouterWithHistory
//

export interface RouterWithHistoryProps {
  history: History;
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
        { text: "OK",     onPress: () => callback(true) }
      ]);
    }
  };

  log = new Log(this.constructor.name);

  componentDidMount = async () => {
    this.log.info('componentDidMount');
  }

  componentWillUnmount = async () => {
    this.log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: RouterWithHistoryProps, prevState: RouterWithHistoryState) => {
    // this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  // Like MemoryRouter
  render = () => (
    <Router {...this.props} />
  );

}
