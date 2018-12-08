// Polyfill console.{time,timeEnd} for jsc (works in rndebugger v8)
import 'react-native-console-time-polyfill';
import { sprintf } from 'sprintf-js';

import { __F_IF_DEV__, global, json, match, pretty, Timer } from './utils';

// Config
const defaultLevel: LevelSetting = 'debug';

// Levels
export type Level        = 'debug' | 'info' | 'warn' | 'error';
export type LevelSetting = Level | 'off';
export function levelOrder(level: LevelSetting): number {
  return {
    'debug': 0,
    'info':  1,
    'warn':  2,
    'error': 3,
    'off':   4,
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

export type LogF = (
  // Label each log line as '${name}.${label}', omitting either/both if falsy
  label:   string,
  // Json format any args that aren't: (1) marked as rich(...) or (2) already a String
  // Format each arg:
  //  - Call if function, so callers can noop expensive computations in Release builds
  //  - json() if not a string or wrapped with rich(...), as sane default formatting with escape hatches
  ...data: any[]
) => void;

// Mark a log arg to not json format
export function rich<X>(x: X): Rich<X> { return new Rich(x); }
export class Rich<X> { constructor(public value: X) {} }

// Logger
//  - Noop if not dev [https://facebook.github.io/react-native/docs/performance#common-sources-of-performance-problems]
//  - Accept log level to filter logging calls (log/debug/info/warn/error)
export class Log {

  constructor(
    public name:  string | null = null,
    public level: LevelSetting  = defaultLevel,
  ) {}

  // Log if level is enabled
  //  - FIXME Any feasible way for console.foo() to see the caller's filename:lineno i/o ours?
  logIfLevel = (level: Level): LogF => {
    return (label: string, ...data: any[]): void => {
      if (levelOrder(level) >= levelOrder(this.level)) {
        // Label each log line as '${name}.${label}', omitting either/both if falsy
        const nameLabel = [
          ...(this.name ? [this.name] : []),
          ...(label     ? [label]     : []),
        ].join('.');
        // Format each arg:
        //  - Call if function, so callers can noop expensive computations in Release builds
        //  - json() if not Rich or String, as sane default formatting with escape hatches
        data = data.map(x => {
          if (x instanceof Function) x = x(); // Unwrap functions first so we can treat its result like non-functions
          return (
            x instanceof Rich      ? x.value :
            typeof(x) === 'string' ? x : // (typeof i/o instanceof for primitive types)
            json(x)
          );
        });
        console[level](
          '%c%s', levelStyle(level), sprintf('%-5s', level.toUpperCase()),
          ...(nameLabel ? [nameLabel] : []),
          ...data,
        );
      }
    };
  }

  // Level methods
  log:   LogF = __F_IF_DEV__(this.logIfLevel('debug')); // Alias
  debug: LogF = __F_IF_DEV__(this.logIfLevel('debug'));
  info:  LogF = __F_IF_DEV__(this.logIfLevel('info'));
  warn:  LogF = __F_IF_DEV__(this.logIfLevel('warn'));
  error: LogF = __F_IF_DEV__(this.logIfLevel('error'));

  // Non-level methods
  time:    typeof console.time    = __F_IF_DEV__(console.time    .bind(console));
  timeEnd: typeof console.timeEnd = __F_IF_DEV__(console.timeEnd .bind(console));

  // Non-level methods
  //  - timed/timedAsync as a nicer alternative to time/timeEnd
  timed = <X>(msg: string, f: () => X): X => {
    const timer = new Timer();
    try {
      return f();
    } finally {
      this.debug(sprintf(`timed: [%.3fs] %s`, timer.time(), msg)); // HACK 'timed:' to place nicely with name prefix
    }
  };
  timedAsync = async <X>(msg: string, f: () => Promise<X>): Promise<X> => {
    const timer = new Timer();
    try {
      return await f();
    } finally {
      this.debug(sprintf(`timed: [%.3fs] %s`, timer.time(), msg)); // HACK 'timed:' to place nicely with name prefix
    }
  };

  // Alias, to allow log.rich() i/o extra top-level import
  rich = rich;

}

// Global logger (with defaults as per above)
export const log = new Log(null);

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

export const print = log.info.bind(log);
export const pp    = (...xs: any[]) => log.info('', ...xs.map(pretty));

// HACK Globals for dev (rely on type checking to catch improper uses of these in real code)
global.log   = log;
global.tap   = tap;
global.puts  = puts;
global.print = print;
global.pp    = pp;

// Greppable + tsc-unhappy shorthand for trace debugging [copy of puts, but different printed string]
global.debugPrint = <X>(x: X, ...args: any[]): X => {
  log.debug('debugPrint', x, ...args);
  return x;
};
