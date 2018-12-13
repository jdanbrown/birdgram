import _ from 'lodash';
import React from 'react';
import { AsyncStorage } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { MetadataColumnBelow, MetadataColumnLeft, MetadataColumnsBelow, MetadataColumnsLeft } from './components/MetadataColumns';
import { HARDCODED_PLACES } from './data/places';
import { Place } from './datatypes';
import { Log, puts, rich } from './log';
import { json, objectKeysTyped, yaml } from './utils';

const log = new Log('Settings');

// All 5 of these lists of attrs (4 here + constructor) must be kept in sync, else load/setItem/etc. aren't typesafe
export interface Props {
  // NOTE Keep attrs in sync (1/5)
  readonly showDebug: boolean;
  readonly allowUploads: boolean;
  readonly maxHistory: number;
  // For RecordScreen
  readonly refreshRate: number;
  readonly doneSpectroChunkWidth: number
  readonly spectroImageLimit: number;
  // For SearchScreen
  readonly showMetadataLeft: boolean;
  readonly showMetadataBelow: boolean;
  readonly metadataColumnsLeft: Array<MetadataColumnLeft>;
  readonly metadataColumnsBelow: Array<MetadataColumnBelow>;
  readonly editing: boolean;
  readonly seekOnPlay: boolean;
  readonly playingProgressEnable: boolean;
  readonly playingProgressInterval: number;
  readonly spectroScale: number;
  readonly place: null | Place;
  // For PlacesScreen
  readonly places: Array<Place>;
}
export const DEFAULTS: Props = {
  // NOTE Keep attrs in sync (2/5)
  showDebug: false,
  allowUploads: true,
  maxHistory: 100, // 0 for unlimited
  // For RecordScreen
  refreshRate: 8,
  doneSpectroChunkWidth: 25, // (ios dims: https://tinyurl.com/y8xsdvnk)
  spectroImageLimit: 0, // 0 for unlimited
  // For SearchScreen
  showMetadataLeft: false,
  showMetadataBelow: false,
  metadataColumnsLeft: ['com_name'] as Array<MetadataColumnLeft>,
  metadataColumnsBelow: objectKeysTyped(MetadataColumnsBelow) as Array<MetadataColumnBelow>,
  editing: false,
  seekOnPlay: false,
  playingProgressEnable: true,
  // playingProgressInterval: 16, // ~frame rate (60fps), but kills rndebugger in dev
  playingProgressInterval: 250,   // Usable in dev
  spectroScale: 2,
  place: null,
  // For PlacesScreen
  places: HARDCODED_PLACES,
};
export const TYPES: {[key: string]: Array<string>} = {
  // NOTE Keep attrs in sync (3/5)
  showDebug: ['boolean'],
  allowUploads: ['boolean'],
  maxHistory: ['number'],
  // For RecordScreen
  refreshRate: ['number'],
  doneSpectroChunkWidth: ['number'],
  spectroImageLimit: ['number'],
  // For SearchScreen
  showMetadataLeft: ['boolean'],
  showMetadataBelow: ['boolean'],
  metadataColumnsLeft: ['object'],
  metadataColumnsBelow: ['object'],
  editing: ['boolean'],
  seekOnPlay: ['boolean'],
  playingProgressEnable: ['boolean'],
  playingProgressInterval: ['number'],
  spectroScale: ['number'],
  place: ['object', 'object'], // null | Place, and typeof null -> 'object'
  // For PlacesScreen
  places: ['object'],
};
export const KEYS = [
  // NOTE Keep attrs in sync (4/5)
  //  - Keys in the order expected by the constructor
  'showDebug',
  'allowUploads',
  'maxHistory',
  // For RecordScreen
  'refreshRate',
  'doneSpectroChunkWidth',
  'spectroImageLimit',
  // For SearchScreen
  'showMetadataLeft',
  'showMetadataBelow',
  'metadataColumnsLeft',
  'metadataColumnsBelow',
  'editing',
  'seekOnPlay',
  'playingProgressEnable',
  'playingProgressInterval',
  'spectroScale',
  'place',
  // For PlacesScreen
  'places',
];

export interface SettingsWrites {
  set<K extends keyof Props>(key: K, value: Props[K]): Promise<void>;
  get<K extends keyof Props>(key: K): Promise<Props[K]>;
  update<K extends keyof Props>(key: K, f: (v: Props[K]) => Props[K]): Promise<void>;
  toggle<K extends keyof Props>(key: K): Promise<boolean>;
}

export class SettingsProxy implements SettingsWrites {
  constructor(
    public getProxy: () => SettingsWrites,
  ) {}
  async set<K extends keyof Props>(key: K, value: Props[K]): Promise<void> {
    return await this.getProxy().set(key, value);
  }
  async get<K extends keyof Props>(key: K): Promise<Props[K]> {
    return await this.getProxy().get(key);
  }
  async update<K extends keyof Props>(key: K, f: (v: Props[K]) => Props[K]): Promise<void> {
    return await this.getProxy().update(key, f);
  }
  async toggle<K extends keyof Props>(key: K): Promise<boolean> {
    return await this.getProxy().toggle(key);
  }
}

export class Settings implements SettingsWrites, Props {
  // WARNING Declare all functions as methods i/o attrs, else they will sneak into serdes

  constructor(
    // Callback to trigger App.setState when settings change
    public readonly appSetState: (settings: Settings) => void,
    // NOTE Keep attrs in sync (4/5)
    public readonly showDebug: boolean,
    public readonly allowUploads: boolean,
    public readonly maxHistory: number,
    // For RecordScreen
    public readonly refreshRate: number,
    public readonly doneSpectroChunkWidth: number,
    public readonly spectroImageLimit: number,
    // For SearchScreen
    public readonly showMetadataLeft: boolean,
    public readonly showMetadataBelow: boolean,
    public readonly metadataColumnsLeft: Array<MetadataColumnLeft>,
    public readonly metadataColumnsBelow: Array<MetadataColumnBelow>,
    public readonly editing: boolean,
    public readonly seekOnPlay: boolean,
    public readonly playingProgressEnable: boolean,
    public readonly playingProgressInterval: number,
    public readonly spectroScale: number,
    public readonly place: null | Place,
    // For PlacesScreen
    public readonly places: Array<Place>,
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
      log.warn(`load: Failed to load saved state, using defaults: keys[${KEYS}]`, e);
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
        log.warn(`load: Dropping saved value with invalid type: {${key}: ${value}} has type ${typeof value} != ${TYPES[key]}`);
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

    log.info('load', rich(settings));
    return settings;

  }

  async set<K extends keyof Props>(key: K, value: Props[K]): Promise<void> {
    log.info('set', {key, value});
    // Set locally
    //  - Before persist: faster App.state response, async persist (which has high variance runtime)
    this.appSetState(this.withProps({
      [key]: value,
    }));
    // Persist in AsyncStorage (async wrt. App.state)
    await Settings.setItem(key, value);
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

  async toggle<K extends keyof Props>(key: K): Promise<boolean> {
    Settings.assertKeyHasType(key, 'boolean');
    const value = this[key];
    await this.set(key, !value);
    return !value;
  }

  //
  // AsyncStorage (Settings.*)
  //

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

  static keyHasType(key: string, type: string) {
    return TYPES[key].includes(type);
  }

  static assertKeyHasType(key: string, type: string) {
    if (!Settings.keyHasType(key, type)) {
      throw `Expected type[${TYPES[key]}] for key[${key}], got type[${type}]`;
    }
  }

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
    return Settings._getItemFromJson(Settings.prefixKey(key));
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

  //
  // AsyncStorage (raw)
  //

  // Like AsyncStorage.getItem except:
  //  - No key prefixing
  //  - Does json serdes
  //  - Returns null if key not found (like AsyncStorage.getItem)
  //  - Returns null if value can't be parsed as json
  static async _getItemFromJson(key: string): Promise<any | null> {
    const json = await AsyncStorage.getItem(key);
    // Return null if no stored value
    if (json === null) {
      return null;
    }
    // Fail gracefully if the stored value is garbage
    try {
      return JSON.parse(json);
    } catch {
      return null;
    }
  }

  // Like AsyncStorage.multiGet, but keys defaults to AsyncStorage.getAllKeys()
  //  - No key prefixing
  //  - No json serdes
  //  - For debugging
  static async _multiGetAll(keys?: Array<string>): Promise<object> {
    keys = keys || await AsyncStorage.getAllKeys();
    return _.fromPairs(await AsyncStorage.multiGet(keys));
  }

}
