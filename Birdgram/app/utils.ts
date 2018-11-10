//
// Utils
//

// Export global:any, which would have otherwise come from DOM but we disable DOM for react-native (tsconfig -> "lib")
//  - Fallback to a mock `{}` for release builds, which run in jsc instead of chrome v8 and don't have window.global
//  - https://facebook.github.io/react-native/docs/javascript-environment
// @ts-ignore
export const global: any = window.global || {};

export function all(...xs: Array<any>): boolean { return xs.every(Boolean); }
export function any(...xs: Array<any>): boolean { return xs.some(Boolean); }

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

export function round(x: number, prec: number = 0): number {
  return Math.round(x * 10**prec) / 10**prec;
}

export function mapMapKeys<K, V, L>(map: Map<K, V>, f: (k: K) => L): Map<L, V> {
  return new Map(Array.from(map).map<[L, V]>(([k, v]) => [f(k), v]));
}

export function mapMapValues<K, V, U>(map: Map<K, V>, f: (v: V) => U): Map<K, U> {
  return new Map(Array.from(map).map<[K, U]>(([k, v]) => [k, f(v)]));
}

export function mapMapEntries<K, V, L, U>(map: Map<K, V>, f: (k: K, v: V) => [L, U]): Map<L, U> {
  return new Map(Array.from(map).map<[L, U]>(([k, v]) => f(k, v)));
}

export class Timer {
  constructor(
    public readonly startTime: Date = new Date(),
  ) {}
  time = (): number => {
    return (new Date().getTime() - this.startTime.getTime()) / 1000; // Seconds
  };
}

// When you want Object.keys(x): Array<keyof typeof x> i/o Array<string>
//  - https://github.com/Microsoft/TypeScript/pull/12253#issuecomment-263132208
export function objectKeysTyped<X extends {}>(x: X): Array<keyof X> {
  return Object.keys(x) as unknown as Array<keyof X>;
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

// Noop marker to indicate that we didn't forget to await a promise
export function noawait<X>(p: Promise<X>): Promise<X> {
  return p;
}

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
// lodash
//

import _, { List } from 'lodash';

// Like _.zip, but refuse arrays of different lengths, and coerce _.zip return element types from (X | undefined) back to X
export function zipSame<X, Y>(xs: List<X>, ys: List<Y>): Array<[X, Y]> {
  if (xs.length !== ys.length) throw `zipSameLength: lengths don't match: xs.length[${xs.length}] !== ys.length[${ys.length}]`;
  return _.zip(xs, ys) as unknown as Array<[X, Y]>;
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
import reactFastCompare from 'react-fast-compare';
import { ImageStyle, RegisteredStyle, TextStyle, ViewStyle } from 'react-native';

// Evolving approach to how to pass StyleSheet parts around
export type Style = RegisteredStyle<ViewStyle | TextStyle | ImageStyle>

// Sounds like this is an anti-pattern
//  - https://github.com/facebook/react/pull/9989#issuecomment-309141521
//  - https://github.com/facebook/react/issues/2642#issuecomment-66676469
//  - https://github.com/facebook/react/issues/2642#issuecomment-309142005
//  - https://github.com/facebook/react/issues/2642#issuecomment-352135607
// export function setStateAsync<P, S, K extends keyof S>(
//   component: Component<P, S>,
//   state: ((prevState: Readonly<S>, props: Readonly<P>) => (Pick<S, K> | S | null)) | (Pick<S, K> | S | null),
// ): Promise<void> {
//   return new Promise((resolve, reject) => {
//     component.setState(state, () => resolve());
//   });
// }

// Typesafe wrapper around react-fast-compare
export function deepEqual<X, Y extends X>(x: X, y: Y | null | undefined): boolean {
  return reactFastCompare(x, y);
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

//
// fs (via rn-fetch-blob)
//

import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

export async function readJsonFile<X extends {}>(path: string): Promise<X> {
  const json = await fs.readFile(path, 'utf8');
  return JSON.parse(json);
}
