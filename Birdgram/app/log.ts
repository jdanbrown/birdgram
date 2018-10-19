import { global, match } from './utils';

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
    return levelOrder(level) < levelOrder(this.level) ? noop : (
      match(level, // (Instead of console[level] because types)
        ['debug', console.debug],
        ['info',  console.info],
        ['warn',  console.warn],
        ['error', console.error],
      ).bind(console)
    );
  }

  // Do f if not dev
  ifDev = <X extends any[]>(f: Do<X>): (Do<X>) => {
    return __DEV__ ? f : noop;
  }

}

// Global logger (with defaults as per above)
export const log = new Log();

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.log   = log;
global.print = log.info.bind(log); // Shorthand for interactive dev
