//
// Utils
//

import reactFastCompare from 'react-fast-compare';

// Export global:any, which would have otherwise come from DOM but we disable DOM for react-native (tsconfig -> "lib")
//  - Fallback to a mock `{}` for release builds, which run in jsc instead of chrome v8 and don't have window.global
//  - https://facebook.github.io/react-native/docs/javascript-environment
// @ts-ignore
export const global: any = window.global || {};

// `X0 extends X` so that x0 can't (quietly) generalize the type of the case patterns (e.g. to include null)
//  - e.g. fail on `match(X | null, ...)` if the case patterns don't include null
export function match<X, X0 extends X, Y>(x0: X0, ...cases: Array<[X | Match, Y]>): Y {
  for (let [x, y] of cases) {
    if (x === x0 || x === Match.Default) {
      return y;
    }
  }
  throw new Error(`No cases matched: ${x0} not in ${cases}`);
}

// Singleton match.default
enum Match { Default }
match.default = Match.Default;

export function getOrSet<K, V>(map: Map<K, V>, k: K, v: () => V): V {
  if (!map.has(k)) {
    map.set(k, v());
  }
  return map.get(k)!;
}

// Typesafe wrapper around react-fast-compare
export function deepEqual<X, Y extends X>(x: X, y: Y): boolean {
  return reactFastCompare(x, y);
}

// Nonstandard shorthands (apologies for breaking norms, but these are too useful and too verbose by default)
export const json   = JSON.stringify;
export const pretty = (x: any) => JSON.stringify(x, null, 2);
export const unjson = JSON.parse;

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.json   = json;
global.pretty = pretty;
global.unjson = unjson;

//
// Promise
//

// TODO How to polyfill Promise.finally in react-native?
//  - Maybe: https://github.com/facebook/fbjs/pull/293
export async function finallyAsync<X>(p: Promise<X>, f: () => Promise<void>): Promise<X> {
  try {
    return await p;
  } finally {
    await f();
  }
}

//
// chance
//

import Chance from 'chance';

// Instantiate a global Chance
export const chance = new Chance();

//
// react
//

import { Component } from 'react';

export function setStateAsync<P, S, K extends keyof S>(
  component: Component<P, S>,
  state: ((prevState: Readonly<S>, props: Readonly<P>) => (Pick<S, K> | S | null)) | (Pick<S, K> | S | null),
): Promise<void> {
  return new Promise((resolve, reject) => {
    component.setState(state, () => resolve());
  });
}

//
// react-native
//

// Generic styles
//  - TODO How to put in a StyleSheet.create without losing type info?
//    - Forced into ViewStyle | TextStyle | ImageStyle, which is too lossy for e.g. TopControlsButtonProps / Feather (icon)
export const Styles = {
  rotate90:       {transform: [{rotate: '90deg'}]},
  rotate180:      {transform: [{rotate: '180deg'}]},
  rotate270:      {transform: [{rotate: '270deg'}]},
  flipHorizontal: {transform: [{scaleX: -1}]},
  flipVertical:   {transform: [{scaleY: -1}]},
  flipBoth:       {transform: [{scaleX: -1}, {scaleY: -1}]},
};

// Copy hard-coded params for react-navigation tab bar
//  - https://github.com/react-navigation/react-navigation-tabs/blob/v0.5.1/src/views/BottomTabBar.js#L199-L201
//  - https://github.com/react-navigation/react-navigation-tabs/blob/v0.5.1/src/views/BottomTabBar.js#L261-L266
export const TabBarBottomConstants = {
  DEFAULT_HEIGHT: 49,
  COMPACT_HEIGHT: 29,
}

export type Point = {
  x: number;
  y: number;
};

export type Dim<X> = {
  width:  X;
  height: X;
};

export type Clamp<X> = {
  min: X;
  max: X;
};
