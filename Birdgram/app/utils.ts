//
// Utils
//

import _ from 'lodash';
import moment from 'moment';
import { sprintf } from 'sprintf-js';

import { debug_print, puts } from './log';

// Export global:any, which would have otherwise come from DOM but we disable DOM for react-native (tsconfig -> "lib")
//  - Fallback to a mock `{}` for release builds, which run in jsc instead of chrome v8 and don't have window.global
//  - https://facebook.github.io/react-native/docs/javascript-environment
// @ts-ignore
export const global: any = window.global || {};

// throw in expression context
export function throw_(e: any): never {
  throw e;
}

export function assert(value: any, msg?: string | (() => string)) {
  if (!value) assertFalse(msg);
}

export function assertFalse(msg?: string | (() => string)): never {
  if (msg instanceof Function) msg = msg(); // Lazy if requested, to avoid perf bottlenecks
  throw new Error(`Assertion failed: ${msg}`);
}

// Simple idiom for local block scope in expression context, like: (() => {...})()
export function local<X>(f: () => X): X {
  return f();
}

// Useful for pattern matching in an expression context
//  - Example usage: into(<complex-expression>, x => ...)
export function into<X, Y>(x: X, f: (x: X) => Y): Y {
  return f(x);
}

// Type annotation (not type assertion)
export function typed<X>(x: X): X {
  return x;
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

export function catchTry<X>(
  c: (e: Error) => X,
  f: () => X,
): X {
  try {
    return f();
  } catch (e) {
    return c(e);
  }
}

export async function catchTryAsync<X>(
  c: (e: Error) => Promise<X>,
  f: () => Promise<X>,
): Promise<X> {
  try {
    return await f();
  } catch (e) {
    return await c(e);
  }
}

// TODO Change cases to functions, like the other ADT matchFoo functions
// `X0 extends X` so that x0 can't (quietly) generalize the type of the case patterns (e.g. to include null)
//  - e.g. fail on `match(X | null, ...)` if the case patterns don't include null
export function match<X, Y = X, X0 extends X = X>(x0: X0, ...cases: Array<[X | Match, (x0: X0) => Y]>): Y {
  for (let [x, f] of cases) {
    if (
      x === x0 || // Shallow equality
      x === Match.Default
    ) {
      return f(x0);
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
  );
}

export function matchNil<X, Y>(x: undefined | null | X, cases: {x: (x: X) => Y, nil: (nil: undefined | null) => Y}): Y {
  return (_.isNil(x)
    ? cases.nil(x)
    : cases.x(x)
  );
}

export function matchEmpty<X, Y>(x: X, cases: {nonEmpty: (x: X) => Y, empty: (x: X) => Y}): Y {
  return (_.isEmpty(x)
    ? cases.empty(x)
    : cases.nonEmpty(x)
  );
}

export function mapNull<X, Y>(x: null | X, f: (x: X) => Y): null | Y {
  return matchNull(x, {null: () => null, x: f});
}

export function mapUndefined<X, Y>(x: undefined | X, f: (x: X) => Y): undefined | Y {
  return matchUndefined(x, {undefined: () => undefined, x: f});
}

export function mapNil<X, Y>(x: undefined | null | X, f: (x: X) => Y): undefined | null | Y {
  return matchNil(x, {nil: nil => nil, x: f});
}

export function mapEmpty<X>(x: X, f: (x: X) => X): X {
  return matchEmpty(x, {empty: x => x, nonEmpty: f});
}

export function ifNull<X>(x: null | X, f: () => X): X {
  return matchNull(x, {null: f, x: x => x});
}

export function ifUndefined<X>(x: undefined | X, f: () => X): X {
  return matchUndefined(x, {undefined: f, x: x => x});
}

export function ifNil<X>(x: undefined | null | X, f: () => X): X {
  return matchNil(x, {nil: f, x: x => x});
}

export function ifEmpty<X>(x: X, f: () => X): X {
  return matchEmpty(x, {empty: f, nonEmpty: x => x});
}

export function getOrSet<K, V>(map: Map<K, V>, k: K, v: () => V): V {
  if (!map.has(k)) {
    map.set(k, v());
  }
  return map.get(k)!;
}

// js is ridiculous
//  - HACK This is `safeParseInt` i/o `safeParseNumber` b/c the String(x) check isn't safe for floats...
export function safeParseInt(s: string): number {
  const x = Number(s);
  if (_.isNaN(x) || String(x) !== s) throw `Failed to parse int: ${s}`;
  return x;
}

export function safeParseIntOrNull(s: string): number | null {
  try {
    return safeParseInt(s);
  } catch {
    return null;
  }
}

export function round(x: number, prec: number = 0): number {
  return Math.round(x * 10**prec) / 10**prec;
}

export function splitFirst(s: string, sep: string): Array<string> {
  const [first, ...rest] = s.split(sep);
  return [first, rest.join(sep)];
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

export function enumerate<X>(xs: Array<X>): Array<{x: X, i: number}> {
  return xs.map((x, i) => ({x, i}));
}

export function mergeArraysWith<X, Y>(
  f: (x: X) => NonNullable<Y>, // NonNullable else errors inside _.maxBy (which lodash types don't catch)
  ...xss: Array<Array<X>>
): Array<X> {
  xss = xss.map(xs => xs.slice()); // Copy each array so we can mutate
  const result: X[] = [];
  while (true) {
    // Filter out empty xs (else need more complexity below)
    xss = xss.filter(xs => xs.length);
    // Stop when no nonempty xs remain
    if (!xss.length) break;
    // Pick max xs[0]
    const {x, i} = _.maxBy(
      xss.map((xs, i) => ({x: xs[0], i})),
      ({x, i}) => f(x),
    )!;
    // Pop xs[0] from xss[i]
    xss[i].shift();
    // Add xs[0] to result
    result.push(x);
  }
  return result;
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
// Typescript
//

export type NoKind<X>  = Omit<X, 'kind'>; // For ADTs
export type Omit<X, K> = Pick<X, Exclude<keyof X, K>>;

//
// stringify / parse / show
//

export function showDate(d: Date): string {
  const nowYear = new Date().getFullYear();
  return (d.getFullYear() === nowYear
    ? moment(d).format('ddd M/D h:mm:ssa')   // e.g. 'Thu 1/10 5:39:49pm'
    : moment(d).format('ddd Y/M/D h:mm:ssa') // e.g. 'Thu 2019/1/10 5:39:49pm'
    // ? moment(d).format('ddd MMM D h:mm:ssa')   // e.g. 'Thu Jan 10 5:39:49pm'
    // : moment(d).format('Y ddd MMM D h:mm:ssa') // e.g. '2019 Thu Jan 10 5:39:49pm'
  );
}

export function showSuffix<X>(sep: string, x: X | undefined, show: (x: X) => string): string {
  return mapEmpty(mapUndefined(x, show) || '', s => sep + s);
}

//
// Interval
//

// Closed-open intervals
export class Interval {

  constructor(
    public lo: number,
    public hi: number,
  ) {}

  static bottom      = new Interval(Infinity,  -Infinity);
  static top         = new Interval(-Infinity, Infinity);
  static nonNegative = new Interval(0, Infinity);
  static nonPositive = new Interval(-Infinity, 0);

  contains = (x: number | Interval): boolean => {
    const {lo, hi} = this;
    return (!(x instanceof Interval) ? (
      // Point: closed-open
      (lo === Infinity || lo <= x) &&
      (hi === Infinity || x < hi)
    ) : (
      // Interval: compare endpoints like closed-closed
      (lo === Infinity || lo <= x.lo) &&
      (hi === Infinity || x.hi <= hi)
    ));
  }

  overlaps = (x: Interval): boolean => {
    return this.contains(x.lo) || this.contains(x.hi);
  }

  clamp = (x: number): number => {
    return _.clamp(x, this.lo, this.hi);
  }

  union = (x: Interval): Interval => {
    return new Interval(
      Math.min(this.lo, x.lo),
      Math.max(this.hi, x.hi),
    );
  }

  intersect = (x: Interval): Interval | null => {
    const lo = Math.max(this.lo, x.lo);
    const hi = Math.min(this.hi, x.hi);
    return lo > hi ? null : new Interval(lo, hi);
  }

  // Can't simply json/unjson because we have to JsonSafeNumber to handle Infinity/-Infinity
  //  - And along the way we simplify '{"lo":x,"hi":y}' -> '[x,y]'
  stringify = (): string => {
    return json(this.jsonSafe());
  }
  static parse = (s: string): Interval => {
    return Interval.unjsonSafe(unjson(s));
  }

  jsonSafe = (): any => {
    return {
      lo: JsonSafeNumber.safe(this.lo),
      hi: JsonSafeNumber.safe(this.hi),
    };
  }
  static unjsonSafe = (x: any): Interval => {
    return new Interval(
      JsonSafeNumber.unsafe(x.lo),
      JsonSafeNumber.unsafe(x.hi),
    );
  }

  show = (): string => {
    return sprintf('[%s–%s]',
      this._showNumber(this.lo),
      this._showNumber(this.hi),
    );
  }

  _showNumber = (x: number): string => {
    const y = JsonSafeNumber.safe(x);
    if (typeof y === 'string') {
      // HACK A bit specialized for concerns in {Edit->DraftEdit->Clip}.show
      return match(y,
        ['NaN',         () => 'NaN'],
        ['-Infinity',   () => ''],
        ['Infinity',    () => ''],
        [match.default, y  => y],
      );
    } else {
      return sprintf('%.2f', y);
    }
  }

};

//
// json
//

import jsonStableStringify from 'json-stable-stringify';

// Nonstandard shorthands (apologies for breaking norms, but these are too useful and too verbose by default)
export const json   = JSON.stringify;
export const pretty = (x: any) => JSON.stringify(x, null, 2);
export const unjson = JSON.parse;

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.json   = json;
global.pretty = pretty;
global.unjson = unjson;

export type JsonSafeNumber = number | 'NaN' | 'Infinity' | '-Infinity';

export const JsonSafeNumber = {

  safe: (x: number): JsonSafeNumber => {
    // WARNING Can't match() on NaN, must _.isNaN
    if (_.isNaN(x)) {
      return 'NaN';
    } else {
      return match<number, JsonSafeNumber>(x,
        [Infinity,      () => 'Infinity'],
        [-Infinity,     () => '-Infinity'],
        [match.default, x  => x],
      );
    }
  },

  unsafe: (x: JsonSafeNumber): number => {
    return match<JsonSafeNumber, number>(x,
      ['NaN',            () => NaN],
      ['Infinity',       () => Infinity],
      ['-Infinity',      () => -Infinity],
      [match.default, x  => x as number],
    );
  },

};

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

// Avoid: sounds like this is an anti-pattern
//  - https://github.com/facebook/react/pull/9989#issuecomment-309141521
//  - https://github.com/facebook/react/issues/2642#issuecomment-66676469
//  - https://github.com/facebook/react/issues/2642#issuecomment-309142005
//  - https://github.com/facebook/react/issues/2642#issuecomment-352135607
export function setStateAsync<P, S, K extends keyof S>(
  component: Component<P, S>,
  state: ((prevState: Readonly<S>, props: Readonly<P>) => (Pick<S, K> | S | null)) | (Pick<S, K> | S | null),
): Promise<void> {
  return new Promise((resolve, reject) => {
    component.setState(state, () => resolve());
  });
}

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

//
// path
//  - Based on https://github.com/LittoCats/react-native-path/blob/master/index.js
//  - Based on https://nodejs.org/api/path.html
//

// TODO Need more normalize()?
//  - https://github.com/LittoCats/react-native-path/blob/master/index.js#L127-L129
export function basename(path: string, ext?: string): string {
  const basename = path.split(/\//g).pop()!;
  if (ext) {
    const tmp = basename.split(/\./g);
    const _ext = tmp.pop();
    if (ext === _ext || ext.slice(1) === _ext) {
      return tmp.join('.')
    }
  }
  return basename;
}
export function dirname(path: string): string {
  return path.split(/\//g).slice(0, -1).join('/');
}
export function extname(path: string): string {
  path = path.replace(/^[\.]+/, '');
  if (/\./.test(path)) {
    return path.match(/\.[^.]*$/)![0];
  } else {
    return '';
  }
}
export function isAbsolute(path: string): boolean {
  return path[0] === '/';
}

export function stripExt(path: string): string {
  return basename(path, extname(path));
}

// Replace unsafe chars in path
//  - (See requireSafePath for details)
export function safePath(path: string, to: string = '-'): string {
  return path.replace(/[:%]/g, to);
}

// Fail if unsafe chars in path
//  - ':' maps to dir separator on ios
//  - '%xx' escapes are mishandled by many react-native libs (e.g. they uri decode too early which garbles 'path/a%3Ab' -> 'path/a:b')
//    - Broken: new Sound(path) -> file not found
//    - Broken: RNFB.fs.createFile(path) -> can't create file
//    - Broken: <Image source={{uri}} /> -> file not found [did I observe the issue with Image or FastImage?]
export function requireSafePath(path: string): string {
  if (/[:%]/.test(path)) throw `Unsafe chars in path[${path}]`;
  return path;
}

//
// URL
//

// TODO Do we need url-parse in addition to query-string?
//  - e.g. queryString.parse in datatypes, queryString.stringify in ebird
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

export function parseUrlNoQuery<X>(url: string): {
  protocol: string,
  host:     string,
  port:     string, // TODO number
  pathname: string,
  query:    string,
  hash:     string,
} {
  return urlParse(
    url,
    // @ts-ignore (Bad d.ts)
    false,
  );
}

export function parseUrlWithQuery<X>(url: string, parseQuery: (s: string) => X): {
  protocol: string,
  host:     string,
  port:     string, // TODO number
  pathname: string,
  query:    X,
  hash:     string,
} {
  const xs = parseUrlNoQuery(url);
  return {
    ...xs,
    query: parseQuery(xs.query),
  };
}

// A parsed query string
export type QueryString = {[key: string]: string};

//
// qs
//  - https://github.com/ljharb/qs
//

import qs from 'qs';
import traverse from 'traverse';

// Wrap qs to provide sane defaults (i/o some quite insane defaults)
export const qsSane = {
  stringify: (x: any, opts?: qs.IStringifyOptions): string => {
    return qs.stringify(
      traverse(x).map(x => !_.isArray(x) ? x : Object.assign({}, x)), // Render {a:['b']} as 'a.0=b' i/o 'a[0]=b'
      {...(opts || {}),
        allowDots:     true,            // Render {a:{b:'c'}}  as 'a.b=c' i/o 'a[b]=c'
        sort:          compare, // Stable key order (by string compare)
        serializeDate: (d: Date) => { throw `Serialize your own dates (because I can't parse them for you): ${d}`; },
      },
    );
  },
  parse: (s: string, opts?: qs.IParseOptions): any => {
    return qs.parse(
      s,
      {...(opts || {}),
        allowDots:         true,     // Parse 'a.b=c' like 'a[b]=c'
        ignoreQueryPrefix: true,     // Ignore leading '?' (default: include in first key)
        arrayLimit:        Infinity, // Always return arrays (default: return object i/o array if size > arrayLimit)
        depth:             Infinity, // Always unroll object (default: produce flat objects with weird keys after depth)
        parameterLimit:    Infinity, // Parse all params (default: drop params beyond parameterLimit)
      },
    );
  },
};

export function compare<X>(x: X, y: X): number {
  return x < y ? -1 : x > y ? 1 : 0;
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
