import _ from 'lodash';
import React from 'react';
import { AsyncStorage } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { InlineMetadataColumn, InlineMetadataColumns } from './datatypes';
import { log, puts } from './log';
import { debugStyle } from './styles';
import { json } from './utils';

export type ShowMetadata = 'none' | 'inline' | 'full';

// All 5 of these lists of attrs (4 here + constructor) must be kept in sync, else load/setItem/etc. aren't typesafe
export interface Props {
  // NOTE Keep attrs in sync (1/5)
  readonly allowUploads: boolean;
  readonly showDebug: boolean;
  // For SearchScreen
  readonly showMetadata: ShowMetadata;
  readonly inlineMetadataColumns: Array<InlineMetadataColumn>;
  readonly editing: boolean;
  readonly seekOnPlay: boolean;
  readonly playingProgressEnable: boolean;
  readonly playingProgressInterval: number;
  readonly spectroScale: number;
}
export const DEFAULTS: Props = {
  // NOTE Keep attrs in sync (2/5)
  allowUploads: true,
  showDebug: false,
  // For SearchScreen
  showMetadata: 'inline',
  inlineMetadataColumns: ['state', 'month_day'] as Array<InlineMetadataColumn>,
  editing: false,
  seekOnPlay: false,
  playingProgressEnable: true,
  // playingProgressInterval: 16, // ~frame rate (60fps), but kills rndebugger in dev
  playingProgressInterval: 250,   // Usable in dev
  spectroScale: 2,
};
export const TYPES: {[key: string]: string} = {
  // NOTE Keep attrs in sync (3/5)
  allowUploads: 'boolean',
  showDebug: 'boolean',
  // For SearchScreen
  showMetadata: 'string',
  inlineMetadataColumns: 'object',
  editing: 'boolean',
  seekOnPlay: 'boolean',
  playingProgressEnable: 'boolean',
  playingProgressInterval: 'number',
  spectroScale: 'number',
};
export const KEYS = [
  // NOTE Keep attrs in sync (4/5)
  //  - Keys in the order expected by the constructor
  'allowUploads',
  'showDebug',
  // For SearchScreen
  'showMetadata',
  'inlineMetadataColumns',
  'editing',
  'seekOnPlay',
  'playingProgressEnable',
  'playingProgressInterval',
  'spectroScale',
];

export class Settings implements Props {
  // WARNING Declare all functions as methods i/o attrs, else they will sneak into serdes

  constructor(
    // Callback to trigger App.setState when settings change
    public readonly appSetState: (settings: Settings) => void,
    // NOTE Keep attrs in sync (4/5)
    public readonly allowUploads: boolean,
    public readonly showDebug: boolean,
  // For SearchScreen
    public readonly showMetadata: ShowMetadata,
    public readonly inlineMetadataColumns: Array<InlineMetadataColumn>,
    public readonly editing: boolean,
    public readonly seekOnPlay: boolean,
    public readonly playingProgressEnable: boolean,
    public readonly playingProgressInterval: number,
    public readonly spectroScale: number,
  ) {}

  withProps(props: object): Settings {
    props = _.assign({}, this, props);
    // @ts-ignore (Possible to do this typesafe-ly?)
    return new Settings(
      this.appSetState,
      ...KEYS.map(key => (props as {[key: string]: any})[key]),
    );
  }

  static async load(appSetState: (settings: Settings) => void): Promise<Settings> {

    // Load whatever junk we might have saved last
    //  - Fallback to DEFAULTS (via {}) if we can't load it at all
    let saved: {[key: string]: any};
    try {
      saved = (
        _.chain(await AsyncStorage.multiGet(KEYS.map(Settings.prefixKey)))
        .map(([k, v]) => [Settings.unprefixKey(k), JSON.parse(v)])
        .fromPairs()
        .value()
      );
    } catch (e) {
      log.warn(`Settings: Failed to load saved state, using defaults: keys[${KEYS}]`, e);
      saved = {};
    }

    // Convert to a Settings, taking care to avoid introducing any values with invalid types
    //  - Careful: iterate over KEYS, not keys from saved, else we might load an outdated set of keys
    //  - Fallback to default values for any values that are missing or have the wrong type
    const values: Array<any> = KEYS.map(key => {
      const value = saved[key];
      const def = (DEFAULTS as {[key: string]: any})[key];
      if (value === null) {
        // Quietly use default for missing keys (e.g. we added a new setting since the last app run)
        return def;
      } else if (!Settings.keyHasType(key, typeof value)) {
        // Warn and use default for saved values with the wrong type (probably a bug)
        //  - TODO Make sure this warning doesn't show in Release builds (e.g. when we change the type for an existing key)
        log.warn(
          `Settings: Dropping saved value with invalid type: {${key}: ${value}} has type ${typeof value} != ${TYPES[key]}`,
        );
        return def;
      } else {
        // Else this key:value should be safe to use
        return value;
      }
    })

    // @ts-ignore (Possible to do this typesafe-ly?)
    const settings = new Settings(
      appSetState,
      ...values,
    );

    log.info('Settings.load', settings);
    return settings;

  }

  async set<K extends keyof Props>(key: K, value: Props[K]): Promise<void> {
    log.info('Settings.set', json({key, value}));
    // Persist in AsyncStorage
    await Settings.setItem(key, value);
    // Set locally (only if persist worked)
    this.appSetState(this.withProps({
      [key]: value,
    }));
  }

  async get<K extends keyof Props>(key: K): Promise<Props[K]> {
    // Read from AsyncStorage
    const value = await Settings.getItem(key)
    if (value === null) {
      throw `Settings key not found: ${key}`;
    } else {
      return value;
    }
  }

  async update<K extends keyof Props>(key: K, f: (v: Props[K]) => Props[K]): Promise<void> {
    await this.set(key, f(this[key]));
  }

  // Prefix keys in AsyncStorage
  static _prefix = 'Settings.';
  static prefixKey(key: string): string {
    return `${Settings._prefix}${key}`;
  };
  static unprefixKey(key: string): string {
    if (!key.startsWith(Settings._prefix)) {
      throw `Expected key[${key}] to start with prefix[${Settings._prefix}]`;
    }
    return key.substr(Settings._prefix.length);
  };

  async toggle<K extends keyof Props>(key: K): Promise<boolean> {
    Settings.assertKeyHasType(key, 'boolean');
    const value = this[key];
    await this.set(key, !value);
    return !value;
  }

  static keyHasType(key: string, type: string) {
    return TYPES[key] === type;
  }

  static assertKeyHasType(key: string, type: string) {
    if (!Settings.keyHasType(key, type)) {
      throw `Expected type[${TYPES[key]}] for key[${key}], got type[${type}]`;
    }
  }

  //
  // AsyncStorage
  //

  // Like AsyncStorage.setItem except:
  //  - Prefixes stored keys
  //  - Does json serdes
  //  - Typesafe (kind of)
  static async setItem(key: string, value: any): Promise<void> {
    Settings.assertKeyHasType(key, typeof value);
    await AsyncStorage.setItem(
      Settings.prefixKey(key),
      JSON.stringify(value),
    );
  }

  // Like AsyncStorage.getItem except:
  //  - Prefixes stored keys
  //  - Does json serdes
  static async getItem(key: string): Promise<any | null> {
    const str = await AsyncStorage.getItem(Settings.prefixKey(key));
    return str && JSON.parse(str);
  }

  // Like AsyncStorage.multiGet except:
  //  - Prefixes stored keys
  //  - Does json serdes
  static async multiGet(keys: Array<string>): Promise<Array<[string, any]>> {
    const entries = await AsyncStorage.multiGet(keys
      .map(Settings.prefixKey)
    );
    return entries.map(([k, v]: [string, string]): [string, any] => [
      Settings.unprefixKey(k),
      JSON.parse(v),
    ])
  }

  // Like AsyncStorage.multiGet, but keys defaults to AsyncStorage.getAllKeys()
  //  - For debugging
  //  - No key prefixing
  //  - No json serdes
  static async _multiGetAll(keys?: Array<string>): Promise<object> {
    keys = keys || await AsyncStorage.getAllKeys();
    return _.fromPairs(await AsyncStorage.multiGet(keys));
  }

  //
  // React
  //

  // Styles
  get debugView(): object {
    return {
      display: this.showDebug ? undefined : 'none',
      padding: 3,
      backgroundColor: debugStyle.backgroundColor,
    };
  }
  get debugText(): object {
    return {
      color: debugStyle.color,
      backgroundColor: debugStyle.backgroundColor,
    };
  }

}
