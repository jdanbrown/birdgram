import _ from 'lodash';
import React from 'react';
import { AsyncStorage } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { MetadataColumnBelow, MetadataColumnLeft, MetadataColumnsBelow, MetadataColumnsLeft } from 'app/components/MetadataColumns';
import { SortListResults, SortSearchResults } from 'app/components/SearchScreen';
import { Place, Quality, Species, SpeciesGroup } from 'app/datatypes';
import { debug_print, Log, puts, rich } from 'app/log';
import { json, match, objectKeysTyped, pretty, typed, yaml } from 'app/utils';

const log = new Log('Settings');

// All 5 of these lists of attrs (4 here + constructor) must be kept in sync, else load/setItem/etc. aren't typesafe
export interface Props {
  // NOTE Keep attrs in sync (1/5)
  readonly showDebug: boolean;
  readonly allowUploads: boolean;
  readonly maxHistory: number;
  readonly f_bins: number;
  // For Geo
  readonly geoHighAccuracy: boolean;
  readonly geoWarnIfNoCoords: boolean;
  // For RecordScreen
  readonly refreshRate: number;
  readonly doneSpectroChunkWidth: number
  readonly spectroChunkLimit: number;
  // For SearchScreen
  readonly n_sp: number;
  readonly n_per_sp: number;
  readonly n_recs: number;
  readonly filterQuality: Set<Quality>;
  readonly sortListResults: SortListResults;
  readonly sortSearchResults: SortSearchResults;
  readonly showMetadataLeft: boolean;
  readonly showMetadataBelow: boolean;
  readonly metadataColumnsLeft: Array<MetadataColumnLeft>;
  readonly metadataColumnsBelow: Array<MetadataColumnBelow>;
  readonly editing: boolean;
  readonly seekOnPlay: boolean;
  readonly playOnTap: boolean;
  readonly playingProgressEnable: boolean;
  readonly playingProgressInterval: number;
  readonly spectroScale: number;
  readonly place: null | Place;
  // For BrowseScreen/SearchScreen
  readonly excludeSpecies:       Set<Species>;
  readonly excludeSpeciesGroups: Set<SpeciesGroup>;
  readonly unexcludeSpecies:     Set<Species>;
  // For PlacesScreen
  readonly places: Array<Place>;
}
export const DEFAULTS: Props = {
  // NOTE Keep attrs in sync (2/5)
  showDebug: false,
  allowUploads: true,
  maxHistory: 100, // 0 for unlimited
  f_bins: 80, // Show higher-res spectros for user recs than model uses to query xc recs (f_bins=40)
  // For Geo
  geoHighAccuracy: true,
  geoWarnIfNoCoords: true,
  // For RecordScreen
  refreshRate: 8,
  doneSpectroChunkWidth: 5, // (ios dims: https://tinyurl.com/y8xsdvnk)
  spectroChunkLimit: 0, // 0 for unlimited
  // For SearchScreen
  n_sp:     10, // For rec queries
  n_per_sp: 3,  // For rec queries
  n_recs:   30, // For non-rec queries
  filterQuality: new Set<Quality>(['A', 'B']),
  sortListResults: 'taxon_order',
  sortSearchResults: 'slp__d_pc',
  showMetadataLeft: true,
  showMetadataBelow: false,
  metadataColumnsLeft: typed<Array<MetadataColumnLeft>>([
    'com_name',
  ]),
  metadataColumnsBelow: typed<Array<MetadataColumnBelow>>([
    'species',
    'species_group',
    'id',
    'recordist',
    'quality',
    'month_day',
    'place',
    'remarks',
  ]),
  editing: false,
  seekOnPlay: true,
  playOnTap: true,
  playingProgressEnable: false, // FIXME High cpu
  // playingProgressInterval: 16, // ~frame rate (60fps), but kills rndebugger in dev
  playingProgressInterval: 250,   // Usable in dev
  spectroScale: 2,
  place: null,
  // For BrowseScreen/SearchScreen
  excludeSpecies:       new Set(),
  excludeSpeciesGroups: new Set(),
  unexcludeSpecies:     new Set(),
  // For PlacesScreen
  places: [],
};
export const TYPES: {[key: string]: Array<string | Function>} = {
  // NOTE Keep attrs in sync (3/5)
  showDebug: ['boolean'],
  allowUploads: ['boolean'],
  maxHistory: ['number'],
  f_bins: ['number'],
  // For Geo
  geoHighAccuracy: ['boolean'],
  geoWarnIfNoCoords: ['boolean'],
  // For RecordScreen
  refreshRate: ['number'],
  doneSpectroChunkWidth: ['number'],
  spectroChunkLimit: ['number'],
  // For SearchScreen
  n_sp: ['number'],
  n_per_sp: ['number'],
  n_recs: ['number'],
  filterQuality: [Set],
  sortListResults: ['string'],
  sortSearchResults: ['string'],
  showMetadataLeft: ['boolean'],
  showMetadataBelow: ['boolean'],
  metadataColumnsLeft: ['object'],
  metadataColumnsBelow: ['object'],
  editing: ['boolean'],
  seekOnPlay: ['boolean'],
  playOnTap: ['boolean'],
  playingProgressEnable: ['boolean'],
  playingProgressInterval: ['number'],
  spectroScale: ['number'],
  place: ['object', 'object'], // null | Place, and typeof null -> 'object'
  // For BrowseScreen/SearchScreen
  excludeSpecies:       [Set],
  excludeSpeciesGroups: [Set],
  unexcludeSpecies:     [Set],
  // For PlacesScreen
  places: ['object'],
};
export const KEYS = [
  // NOTE Keep attrs in sync (4/5)
  //  - Keys in the order expected by the constructor
  'showDebug',
  'allowUploads',
  'maxHistory',
  'f_bins',
  // For Geo
  'geoHighAccuracy',
  'geoWarnIfNoCoords',
  // For RecordScreen
  'refreshRate',
  'doneSpectroChunkWidth',
  'spectroChunkLimit',
  // For SearchScreen
  'n_sp',
  'n_per_sp',
  'n_recs',
  'filterQuality',
  'sortListResults',
  'sortSearchResults',
  'showMetadataLeft',
  'showMetadataBelow',
  'metadataColumnsLeft',
  'metadataColumnsBelow',
  'editing',
  'seekOnPlay',
  'playOnTap',
  'playingProgressEnable',
  'playingProgressInterval',
  'spectroScale',
  'place',
  // For BrowseScreen/SearchScreen
  'excludeSpecies',
  'excludeSpeciesGroups',
  'unexcludeSpecies',
  // For PlacesScreen
  'places',
];

export interface SettingsWrites {
  set(props: Partial<Props> | ((props: Props) => Partial<Props>)): Promise<void>;
  get<K extends keyof Props>(key: K): Promise<Props[K]>;
  update<K extends keyof Props>(key: K, f: (v: Props[K]) => Props[K]): Promise<void>;
  toggle<K extends keyof Props>(key: K): Promise<boolean>;
}

export class SettingsProxy implements SettingsWrites {
  constructor(
    public getProxy: () => SettingsWrites,
  ) {}
  async set(props: Partial<Props> | ((props: Props) => Partial<Props>)): Promise<void> {
    return await this.getProxy().set(props);
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
    public readonly f_bins: number,
    // For Geo
    public readonly geoHighAccuracy: boolean,
    public readonly geoWarnIfNoCoords: boolean,
    // For RecordScreen
    public readonly refreshRate: number,
    public readonly doneSpectroChunkWidth: number,
    public readonly spectroChunkLimit: number,
    // For SearchScreen
    public readonly n_sp: number,
    public readonly n_per_sp: number,
    public readonly n_recs: number,
    public readonly filterQuality: Set<Quality>,
    public readonly sortListResults: SortListResults,
    public readonly sortSearchResults: SortSearchResults,
    public readonly showMetadataLeft: boolean,
    public readonly showMetadataBelow: boolean,
    public readonly metadataColumnsLeft: Array<MetadataColumnLeft>,
    public readonly metadataColumnsBelow: Array<MetadataColumnBelow>,
    public readonly editing: boolean,
    public readonly seekOnPlay: boolean,
    public readonly playOnTap: boolean,
    public readonly playingProgressEnable: boolean,
    public readonly playingProgressInterval: number,
    public readonly spectroScale: number,
    public readonly place: null | Place,
    // For BrowseScreen/SearchScreen
    public readonly excludeSpecies:       Set<Species>,
    public readonly excludeSpeciesGroups: Set<SpeciesGroup>,
    public readonly unexcludeSpecies:     Set<Species>,
    // For PlacesScreen
    public readonly places: Array<Place>,
  ) {}

  withProps(props: Partial<Props>): Settings {
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
        .map(([key, value]) => [
          Settings.unprefixKey(key),
          Settings.parse(Settings.unprefixKey(key), value),
        ])
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
      } else if (!Settings.keyHasTypeOfValue(key, value)) {
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

  // XXX Replaced with set(props) (via multiSet i/o setItem)
  // async set<K extends keyof Props>(key: K, value: Props[K]): Promise<void> {
  //   log.info('set', {key, value});
  //   // Set locally
  //   //  - Before persist: faster App.state response, async persist (which has high variance runtime)
  //   this.appSetState(this.withProps({
  //     [key]: value,
  //   }));
  //   // Persist in AsyncStorage (async wrt. App.state)
  //   await Settings.setItem(key, value);
  // }

  async set(props: Partial<Props> | ((props: Props) => Partial<Props>)): Promise<void> {
    // Promote input types
    if (props instanceof Function) props = props(this);
    // Log (after promoting)
    log.info('set', rich(props));
    // Set locally
    //  - Before persist: faster App.state response, async persist (which has high variance runtime)
    this.appSetState(this.withProps(props));
    // Persist in AsyncStorage (async wrt. App.state)
    await Settings.multiSet(props);
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
    await this.set({[key]: f(this[key])});
  }

  async toggle<K extends keyof Props>(key: K): Promise<boolean> {
    Settings.assertKeyHasTypeOfValue(key, true);
    const value = this[key];
    await this.set({[key]: !value});
    return !value;
  }

  //
  // Serdes
  //

  static conversions: Array<{
    type:        Array<string | Function>,
    onStringify: (x: any) => any,
    onParse:     (x: any) => any,
  }> = [{
    type:        [Set],
    onStringify: x => Array.from(x),
    onParse:     x => new Set(x),
  }];

  static stringify(key: string, x: any): string {
    var s;
    try {
      const t = TYPES[key];
      Settings.conversions.forEach(conversion => {
        if (_.isEqual(t, conversion.type)) x = conversion.onStringify(x);
      });
      s = JSON.stringify(x);
    } catch (e) {
      const d = DEFAULTS[key as keyof Props];
      log.warn(`stringify: For key[$key], failed to stringify x[${x}], using default[${d}]`, e);
      s = JSON.stringify(d);
    }
    return s;
  }

  static parse(key: string, s: string): any {
    var x: any;
    try {
      x = JSON.parse(s);
      const t = TYPES[key];
      Settings.conversions.forEach(conversion => {
        if (_.isEqual(t, conversion.type)) x = conversion.onParse(x);
      });
    } catch (e) {
      const d = DEFAULTS[key as keyof Props];
      log.warn(`parse: For key[${key}], failed to parse s[${s}], using default[${d}]`, e);
      x = d;
    }
    return x;
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

  static keyHasTypeOfValue(key: string, value: any) {
    return _.some(TYPES[key], t => (
      t === typeof value ||   // e.g. 'boolean', 'object'
      t === value.constructor // e.g. Set
    ));
  }

  static assertKeyHasTypeOfValue(key: string, value: any) {
    if (!Settings.keyHasTypeOfValue(key, value)) {
      throw `Expected type[${TYPES[key]}] for key[${key}], got type[${typeof value}]`;
    }
  }

  // Like AsyncStorage.setItem except:
  //  - Prefixes stored keys
  //  - Does json serdes
  //  - Typesafe (kind of)
  static async setItem(key: string, value: any): Promise<void> {
    Settings.assertKeyHasTypeOfValue(key, value);
    await AsyncStorage.setItem(
      Settings.prefixKey(key),
      Settings.stringify(key, value),
    );
  }

  // Like AsyncStorage.multiSet except:
  //  - Prefixes stored keys
  //  - Does json serdes
  //  - Typesafe (kind of)
  static async multiSet(props: object): Promise<void> {
    const kvs = _.toPairs(props);
    kvs.forEach(([key, value]) => {
      Settings.assertKeyHasTypeOfValue(key, value);
    });
    await AsyncStorage.multiSet(kvs.map(([key, value]) => [
      Settings.prefixKey(key),
      Settings.stringify(key, value),
    ]));
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
    return entries.map(([key, value]: [string, string]): [string, any] => [
      Settings.unprefixKey(key),
      Settings.parse(Settings.unprefixKey(key), value),
    ])
  }

  //
  // AsyncStorage (raw)
  //

  // Like AsyncStorage.getItem except:
  //  - No key prefixing
  //  - Does serdes
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
      return Settings.parse(key, json);
    } catch {
      return null;
    }
  }

  // Like AsyncStorage.multiGet, but keys defaults to AsyncStorage.getAllKeys()
  //  - No key prefixing
  //  - No serdes
  //  - For debugging
  static async _multiGetAll(keys?: Array<string>): Promise<object> {
    keys = keys || await AsyncStorage.getAllKeys();
    return _.fromPairs(await AsyncStorage.multiGet(keys));
  }

}
