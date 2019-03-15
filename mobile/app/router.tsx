import _ from 'lodash';
import * as _history from 'history';
import React, { PureComponent, ReactNode } from 'react';
import { Alert, AsyncStorage } from 'react-native';
import { MemoryRouterProps } from 'react-router';
import { Route, Router } from 'react-router-native';

import { debug_print, Log, puts, rich } from 'app/log';
import { Settings } from 'app/settings';
import { global, into, json, local, mapNull, Nil, pretty, shallowDiffPropsState, typed, yaml } from 'app/utils';

const log = new Log('router');

//
// Utils
//

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
// Go
//

export type Go = (tab: TabName, to?: GoTo) => void;
export type GoTo = {path?: string, index?: number}; // TODO Make this a proper union type

export function go(histories: Histories, tab: TabName, to?: GoTo) {
  log.info('go', {tab, to});
  // Update tab location (async, can't await)
  const history = histories[tab];
  if (to === undefined) {
    // Noop location, just switch to tab
  } else if (to.path !== undefined) {
    // Push new location on top of most recent location (else we lose items)
    //  - HACK Mutate history.index i/o calling history.go() to avoid an unnecessary update
    //    - Ref: https://github.com/ReactTraining/history/blob/v4.7.2/modules/createMemoryHistory.js#L61
    history.index = history.length - 1;
    // Dedupe contiguous history.entries: replace if .pathname isn't changing
    //  - NOTE .state isn't compared; it should be determined by .pathname (modulo important, e.g. timestamp isn't)
    //  - NOTE Compare using locationPathIsEqual b/c === on .pathname isn't reliable (see locationPathIsEqual)
    if (locationPathIsEqual({pathname: to.path}, history.entries[history.index])) {
      // log.debug('go: Dedupe', {to, historyLocation: history.location}); // XXX Debug
    } else {
      // log.debug('go: Push', {to, historyLocation: history.location}); // XXX Debug
      history.push(to.path, {
        timestamp: new Date(),
      });
    }
  } else if (to.index !== undefined) {
    // Jump back/forward to existing location in history (by index)
    history.go(to.index - history.index);
  }
  // Switch to tab
  histories.tabs.replace('/' + tab); // (Leading '/' for absolute i/o relative)
}

//
// Histories
//

// Redefine standard history types with our own LocationState
//  - Need `| undefined` for locations that are created from just a path (e.g. <Redirect to={path}/> i/o <Redirect to={location}/>)
//  - TODO If we update _all_ uses of Redirect [any other location producers?] then we can remove `| undefined`
export type Location = _history.Location<LocationState | undefined>;
export type History  = _history.MemoryHistory<LocationState | undefined>;
export interface LocationState {
  timestamp: Date;
}

// TODO(location_equality): How should Location equality work?
//  - Current expectations
//    - .pathname -- determines identity (eg. it embeds a SourceId)
//    - .state    -- determined by .pathname (but difficult/infeasible to compute from .pathname)
//    - .search   -- unused and ignored (prefer one of .pathname/.state)
//    - .hash     -- unused and ignored (prefer one of .pathname/.state)
//    - .key      -- unused and ignored [TODO But is it treated specially by react-router? e.g. to determine equality?]
//  - Current usage
//    - RecentScreen was comparing like: x.key === y.key
//    - SavedScreen builds mock locations from just a pathname, where both .key and even .state are meaningless
export function locationKeyIsEqual(
  x: Nil<Pick<Location, 'key'>>,
  y: Nil<Pick<Location, 'key'>>,
): boolean {
  return (
    _.get(x, 'key') ===
    _.get(y, 'key')
  );
}
export function locationPathIsEqual(
  x: Nil<Pick<Location, 'pathname'>>,
  y: Nil<Pick<Location, 'pathname'>>,
): boolean {
  return (
    // HACK decodeURIComponent b/c something in history/react-router messes up uri encodings (YOU HAD ONE JOB)
    //  - x === '/edit/user:user-20180101-abcdef'   <- wrong, mangled after going through History/Location/router
    //  - y === '/edit/user%3Auser-20180101-abcdef' <- right, via encodeURIComponent in e.g. SavedScreen.recordSavedFromSource
    mapNull(_.get(x, 'pathname', null), decodeURIComponent) ===
    mapNull(_.get(y, 'pathname', null), decodeURIComponent)
  );
}

export function locationStateOrEmpty(state?: LocationState): LocationState | {[key in keyof LocationState]: null} {
  return state || {
    timestamp: null,
  };
}

export function createHistory(opts: {
  initialEntries?: Array<Location>;
  initialIndex?:   number;
  keyLength?:      number;
} = {}): History {
  const history = _history.createMemoryHistory({
    initialEntries: opts.initialEntries as unknown as Array<string>, // HACK Work around bunk type (string i/o Location)
    initialIndex:   opts.initialIndex,
    keyLength:      opts.keyLength,
  });
  // Add initial state (undefined by default -- which is only observable on first launch after app install!)
  history.location.state = typed<LocationState>({
    timestamp: new Date(),
  });
  return history;
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

export type TabLocations<K extends string = TabName> = {[key in K]: Location};
export function getTabLocations(histories: TabHistories): TabLocations {
  return {
    record:   histories.record.location,
    search:   histories.search.location,
    recent:   histories.recent.location,
    saved:    histories.saved.location,
    places:   histories.places.location,
    settings: histories.settings.location,
    help:     histories.help.location,
    // tabs:    histories.tabs.location, // Omit .tabs (else we'd trigger componentDidUpdate on each tab switch)
  };
}

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
