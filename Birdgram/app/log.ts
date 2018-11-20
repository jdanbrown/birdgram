// Polyfill console.{time,timeEnd} for jsc (works in rndebugger v8)
import 'react-native-console-time-polyfill';
import { sprintf } from 'sprintf-js';

import { global, match, pretty } from './utils';

// Config
const defaultLevel: Level = 'debug';

// Levels
export type Level = 'debug' | 'info' | 'warn' | 'error';
export function levelOrder(level: Level): number {
  return {
    'debug': 0,
    'info':  1,
    'warn':  2,
    'error': 3,
  }[level];
}
export function levelStyle(level: Level): string {
  return {
    'debug': 'padding: 0 1px; color: white; background: #1f78b4', // Blue
    'info':  'padding: 0 1px; color: white; background: #33a02c', // Green
    'warn':  'padding: 0 1px; color: white; background: #fec44f', // Yellow/orange
    'error': 'padding: 0 1px; color: white; background: #e31a1c', // Red
  }[level];
}

// FIXME These Do<X> types are busted
//  - e.g. log.time(1, 2, 3) typechecks but shouldn't
//  - console spec: https://console.spec.whatwg.org/
//  - Maybe use conditional types + infer? https://www.typescriptlang.org/docs/handbook/advanced-types.html

// Shorthands for below
type Do<X extends any[]> = (...args: X) => void;
const noop: Do<any[]> = (...args) => {};

// Logger
//  - Noop if not dev [https://facebook.github.io/react-native/docs/performance#common-sources-of-performance-problems]
//  - Accept log level to filter logging calls (log/debug/info/warn/error)
export class Log {

  constructor(
    public level: Level = defaultLevel,
  ) {}

  // Level methods
  get log   (): Do<any[]> { return this.ifDev(this.logIfLevel('debug')); } // Same as log.debug
  get debug (): Do<any[]> { return this.ifDev(this.logIfLevel('debug')); }
  get info  (): Do<any[]> { return this.ifDev(this.logIfLevel('info'));  }
  get warn  (): Do<any[]> { return this.ifDev(this.logIfLevel('warn'));  }
  get error (): Do<any[]> { return this.ifDev(this.logIfLevel('error')); }

  // Non-level methods
  get time    (): Do<any[]> { return this.ifDev(console.time    .bind(console)); }
  get timeEnd (): Do<any[]> { return this.ifDev(console.timeEnd .bind(console)); }

  // Log if level is enabled
  //  - Return a (bound) console method so that the caller's filename:lineno comes through instead of ours
  //  - Assume caller does ifDev
  logIfLevel = (level: Level): Do<any[]> => {
    const console_f = levelOrder(level) < levelOrder(this.level) ? noop : (
      match(level, // (Instead of console[level] because types)
        ['debug', () => console.debug],
        ['info',  () => console.info],
        ['warn',  () => console.warn],
        ['error', () => console.error],
      ).bind(console)
    );
    return (...args: any[]) => console_f(
      '%c%s', levelStyle(level), sprintf('%-5s', level.toUpperCase()),
      ...args,
    );
  }

  // Do f if not dev
  ifDev = <X extends any[]>(f: Do<X>): (Do<X>) => {
    return __DEV__ ? f : noop;
  }

}

// Global logger (with defaults as per above)
export const log = new Log();

//
// Utils (that rely on log.*)
//

export function tap<X>(x: X, f: (x: X) => void): X {
  f(x);
  return x;
}

export function puts<X>(x: X, ...args: any[]): X {
  log.debug('puts', x, ...args);
  return x;
}

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.log   = log;
global.tap   = tap;
global.puts  = puts;
global.print = log.info.bind(log);
global.pp    = (...xs: any[]) => log.info(...xs.map(pretty));

// Greppable + tsc-unhappy shorthand for trace debugging [copy of puts, but different printed string]
global.debugPrint = <X>(x: X, ...args: any[]): X => {
  log.debug('debugPrint', x, ...args);
  return x;
};
