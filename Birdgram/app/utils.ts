//
// Utils
//

import _ from 'lodash';

// Export global:any, which would have otherwise come from DOM but we disable DOM for react-native (tsconfig -> "lib")
//  - Fallback to a mock `{}` for release builds, which run in jsc instead of chrome v8 and don't have window.global
//  - https://facebook.github.io/react-native/docs/javascript-environment
// @ts-ignore
export const global: any = window.global || {};

export function assert(value: any, msg?: string) {
  if (!value) {
    throw new Error(`Assertion failed: ${msg}`);
  }
}

// Useful for pattern matching in an expression context
//  - Example usage: into(<complex-expression>, x => ...)
export function into<X, Y>(x: X, f: (x: X) => Y): Y {
  return f(x);
}

export function all(...xs: Array<any>): boolean { return xs.every(Boolean); }
export function any(...xs: Array<any>): boolean { return xs.some(Boolean); }

export function tryElse<Z, X extends Z = Z, Y extends Z = Z>(x: X, f: () => Y): Z {
  try {
    return f();
  } catch {
    return x;
  }
}

export async function tryElseAsync<Z, X extends Z = Z, Y extends Z = Z>(x: X, f: () => Promise<Y>): Promise<Z> {
  try {
    return await f();
  } catch {
    return x;
  }
}

// TODO Change cases to functions, like the other ADT matchFoo functions
// `X0 extends X` so that x0 can't (quietly) generalize the type of the case patterns (e.g. to include null)
//  - e.g. fail on `match(X | null, ...)` if the case patterns don't include null
export function match<X, X0 extends X, Y>(x0: X0, ...cases: Array<[X | Match, () => Y]>): Y {
  for (let [x, f] of cases) {
    if (x === x0 || x === Match.Default) {
      return f();
    }
  }
  throw new Error(`No cases matched: ${x0} not in ${cases}`);
}

// Singleton match.default
enum Match { Default }
match.default = Match.Default;

export function matchNull<X, Y>(x: null | X, cases: {x: (x: X) => Y, null: () => Y}): Y {
  switch (x) {
    case null: return cases.null();
    default:   return cases.x(x);
  }
}

export function matchUndefined<X, Y>(x: undefined | X, cases: {x: (x: X) => Y, undefined: () => Y}): Y {
  return (x === undefined
    ? cases.undefined()
    : cases.x(x)
  )
}

export function matchNil<X, Y>(x: undefined | null | X, cases: {x: (x: X) => Y, nil: (nil: undefined | null) => Y}): Y {
  return (_.isNil(x)
    ? cases.nil(x)
    : cases.x(x)
  )
}

export function mapNull<X, Y>(x: null | X, f: (x: X) => Y): null | Y {
  return matchNull(x, {null: () => null, x: x => f(x)});
}

export function mapUndefined<X, Y>(x: undefined | X, f: (x: X) => Y): undefined | Y {
  return matchUndefined(x, {undefined: () => undefined, x: x => f(x)});
}

export function mapNil<X, Y>(x: undefined | null | X, f: (x: X) => Y): undefined | null | Y {
  return matchNil(x, {nil: nil => nil, x: x => f(x)});
}

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
    public startTime: Date = new Date(),
  ) {}
  time = (): number => {
    return (new Date().getTime() - this.startTime.getTime()) / 1000; // Seconds
  };
  reset = () => {
    this.startTime = new Date();
  }
  lap = (): number => {
    const time = this.time();
    this.reset();
    return time;
  }
}

// When you want Object.keys(x): Array<keyof typeof x> i/o Array<string>
//  - https://github.com/Microsoft/TypeScript/pull/12253#issuecomment-263132208
export function objectKeysTyped<X extends {}>(x: X): Array<keyof X> {
  return Object.keys(x) as unknown as Array<keyof X>;
}

// Useful for debugging shouldComponentUpdate
export function shallowDiff(x: object, y: object): {[key: string]: boolean} {
  return _.assignWith<{[key: string]: boolean}>(
    _.clone(x),
    y,
    (a: any, b: any) => a === b,
  );
}

export class ExpWeightedMean {
  constructor(
    public readonly alpha: number,
    public value: number = 0,
  ) {
    if (!(0 <= alpha && alpha <= 1)) {
      throw `ExpWeightedMean: alpha[${alpha}] must be in [0,1]`;
    }
  }
  add = (x: number) => {
    this.value = this.alpha*x + (1 - this.alpha)*this.value;
  }
}

export class ExpWeightedRate {
  _timer: Timer;
  _mean:  ExpWeightedMean;
  constructor(
    public readonly alpha: number,
  ) {
    this._timer = new Timer();
    this._mean  = new ExpWeightedMean(alpha);
  }
  reset = () => { this._timer.reset(); }
  mark  = () => { this._mean.add(this._timer.lap()); }
  get value(): number { return 1 / this._mean.value }
}

export function timed<X>(f: () => X): number {
  return _timed(f).time;
}

export function _timed<X>(f: () => X): {x: X, time: number} {
  const timer = new Timer();
  const x = f();
  return {x, time: timer.time()};
}

export function times<X>(n: number, f: () => X) {
  while (n > 0) {
    f();
    n -= 1;
  }
}

// HACK Globals for dev
global.timed = timed;
global.times = times;
global.str   = (x: any) => x.toString(); // e.g. for nj.array, which have a not useful console.log but a useful .toString()

//
// json
//

// Nonstandard shorthands (apologies for breaking norms, but these are too useful and too verbose by default)
export const json   = JSON.stringify;
export const pretty = (x: any) => JSON.stringify(x, null, 2);
export const unjson = JSON.parse;

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.json   = json;
global.pretty = pretty;
global.unjson = unjson;

//
// yaml
//

import Yaml from 'js-yaml';

export function yaml(x: any, opts?: Yaml.DumpOptions): string {
  return Yaml.safeDump(x, {
    flowLevel: 0,                     // Single line (pass -1 for multi-line pretty print)
    schema: Yaml.DEFAULT_FULL_SCHEMA, // Don't barf on undefined [also allows all kinds of rich types through as !!foo]
    lineWidth: 1e9,                   // Don't wrap long strings (default: 80)
    ...(opts || {}),
  }).trim(); // Remove trailing newline (only when flowLevel > 0)
}

export function yamlPretty(x: any, opts?: Yaml.DumpOptions): string {
  return yaml(x, {
    flowLevel: -1,
    ...(opts || {}),
  });
}

export function unyaml(x: string, opts?: Yaml.LoadOptions): any {
  return Yaml.safeLoad(x, {
    schema: Yaml.DEFAULT_SAFE_SCHEMA, // Do barf on undefined, to avoid loading unsafe code (e.g. !!js/function)
    ...(opts || {}),
  });
}

global.Yaml       = Yaml;
global.yaml       = yaml;
global.yamlPretty = yamlPretty;
global.unyaml     = unyaml;

//
// Typescript
//

export type Omit<X, K> = Pick<X, Exclude<keyof X, K>>;

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

import { List } from 'lodash';

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

export function shallowDiffPropsState<Props, State>(prevProps: Props, prevState: State, props: Props, state: State): object {
  const diff = {};
  [
    {prefix: 'props', prevObj: prevProps, obj: props},
    {prefix: 'state', prevObj: prevState, obj: state},
  ].forEach(({prefix, prevObj, obj}) => {
    const changed = _.assignWith(_.clone(prevObj), obj, (x: any, y: any) => x === y);
    _.uniq([..._.keys(prevObj), ..._.keys(obj)]).forEach(k => {
      const diffKey = `${prefix}.${k}`;
      const prev = (prevObj as Record<string, any>)[k];
      const curr = (obj     as Record<string, any>)[k];
      if (prev !== curr) {
        (diff as Record<string, any>)[diffKey] = {prev, curr};
      }
    })
  });
  return diff;
}

//
// react-native
//

// Like the RCT_IF_DEV() objc macro, to distinguish Xcode Debug vs. Release build
//  - Structured as a higher-order function for js
//  - Example usage: `f = __F_IF_DEV__(x => ...)`
//  - See log.ts for examples
export function __F_IF_DEV__<F extends (...args: any[]) => void>(f: F): F {
  const noop = ((...args: any[]) => {}) as F; // HACK How to typecheck this properly?
  return __DEV__ ? f : noop;
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

//
// path
//  - Based on https://github.com/LittoCats/react-native-path/blob/master/index.js
//

// TODO Need more normalize()?
//  - https://github.com/LittoCats/react-native-path/blob/master/index.js#L127-L129
export function dirname(path: string): string {
  return path.split(/\//g).slice(0, -1).join('/');
}
export function basename(path: string): string {
  return path.split(/\//g).slice(-1)[0]!;
}

//
// URL
//

import urlParse from 'url-parse';

export function parseUrl(url: string): {
  protocol: string,
  host:     string,
  port:     string, // TODO number
  pathname: string,
  query:    {[key: string]: any},
  hash:     string,
} {
  return urlParse(
    url,
    // @ts-ignore (Bad d.ts)
    true,
  );
}

//
// fs (via rn-fetch-blob)
//

import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

export async function ensureParentDir(path: string): Promise<string> {
  await ensureDir(dirname(path));
  return path;
}

export async function ensureDir(path: string): Promise<string> {
  await mkdir_p(path);
  return path;
}

export async function mkdir_p(path: string): Promise<void> {
  // Tricky
  //  - Constraints
  //    - fs.mkdir throws if path exists as file
  //    - fs.mkdir throws if path exists as dir
  //    - fs.mkdir succeeds and creates intermediate dirs if path doesn't exist
  //    - And we need to be concurrency safe
  //  - Approach
  //    - Blindly do fs.mkdir()
  //    - If it succeeds, we're done
  //    - If it throws, then fail only if we can detect that path doesn't exist as a dir (via fs.isDir)
  //      - This final check is still open to races, but the behavior is equivalent to a trace where the mkdir did
  //        succeed before someone else rmdir'd it and then did whatever, so it's sound to claim success
  try {
    await fs.mkdir(path);
  } catch {
    if (!await fs.isDir(path)) {
      throw Error(`mkdir_p: path already exists as a file: ${path}`);
    } else {
      // Already exists as dir (which is good)
    }
  }
}

export async function readJsonFile<X extends {}>(path: string): Promise<X> {
  const json = await fs.readFile(path, 'utf8');
  return JSON.parse(json);
}
