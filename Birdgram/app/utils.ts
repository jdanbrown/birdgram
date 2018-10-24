//
// Utils
//

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

//
// Promise
//

// TODO How to polyfill Promise.finally in react-native?
//  - Maybe: https://github.com/facebook/fbjs/pull/293
export async function finallyAsync<X>(p: Promise<X>, f: () => void): Promise<X> {
  try {
    return await p;
  } finally {
    f();
  }
}

//
// chance
//

import Chance from 'chance';

// Instantiate a global Chance
export const chance = new Chance();

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
