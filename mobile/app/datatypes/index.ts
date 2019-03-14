import _ from 'lodash';
import queryString from 'query-string';
import { matchPath } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
const {fs} = RNFB;

import { Filters } from 'app/components/SearchScreen';
import { Place } from 'app/datatypes/place';
import { SourceId } from 'app/datatypes/source';
import { debug_print, log, Log, rich } from 'app/log';
import { Location } from 'app/router';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchNull, matchUndefined, NoKind,
  Omit, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane, requireSafePath, safeParseInt, safeParseIntOrNull,
  safePath, showDate, showSuffix, splitFirst, stripExt, throw_, tryElse, typed, unjson,
} from 'app/utils';

export * from 'app/datatypes/edit';
export * from 'app/datatypes/place';
export * from 'app/datatypes/rec';
export * from 'app/datatypes/source';

//
// App globals
//

export class Paths {
  static payloads    = 'payloads';
  static search_recs = `${Paths.payloads}/search_recs`;
};

export const Models = {
  search: {
    path: `${Paths.search_recs}/models/search.json`,
  },
};

// See Bubo/FileProps.swift for behaviors that we haven't yet ported back to js
export interface FileProps {
  _path: string;
}

export interface ModelsSearch extends FileProps {
  classes_: Array<string>;
}

export const SearchRecs = {

  serverConfigPath:    `${Paths.search_recs}/server-config.json`,
  metadataSpeciesPath: `${Paths.search_recs}/metadata/species.json`,
  metadataXcIdsPath:   `${Paths.search_recs}/metadata/xc_ids.json`,
  dbPath:              `${Paths.search_recs}/search_recs.sqlite3`, // TODO Test asset paths on android (see notes in README)

  // TODO After verifying that asset dirs are preserved on android, simplify the basenames back to `${xc_id}.${format}`
  assetPath: (kind: string, species: string, xc_id: number, format: string): string => (
    `${fs.dirs.MainBundleDir}/${Paths.search_recs}/${kind}/${species}/${kind}-${species}-${xc_id}.${format}`
  ),

};

export interface ServerConfig {
  server_globals: {
    sg_load: {
      search: object,
      xc_meta: {
        countries_k: string | null,
        com_names_k: string | null,
        num_recs:    number | null,
      },
    },
  };
  api: {
    recs: {
      search_recs: {
        params: {
          limit: number,
          audio_s: number,
        },
      },
      spectro_bytes: {
        format: string,
        convert?: object,
        save?: object,
      },
    },
  };
  audio: {
    audio_persist: {
      audio_kwargs: {
        format: string,
        bitrate?: string,
        codec?: string,
      },
    },
  };
};

export type MetadataSpecies = Array<SpeciesMetadata>;
export type MetadataXcIds   = {[key: string]: Species};

//
// Species
//

export type Species     = string; // SpeciesMetadata.shorthand    (e.g. 'HETH')
export type SpeciesCode = string; // SpeciesMetadata.species_code (e.g. 'herthr')

export interface SpeciesMetadata {
  sci_name:       string;
  com_name:       string;
  species_code:   SpeciesCode;
  taxon_order:    string; // NOTE Should sort as number, not string (e.g. sql `cast(taxon_order as real)` in SearchScreen)
  taxon_id:       string;
  com_name_codes: string;
  sci_name_codes: string;
  banding_codes:  string;
  shorthand:      Species;
  longhand:       string;
  species_group:  string;
  family:         string;
  order:          string;
};

//
// RecordPathParams
//

export type RecordPathParams =
  | RecordPathParamsRoot
  | RecordPathParamsEdit;
export type RecordPathParamsRoot = { kind: 'root' };
export type RecordPathParamsEdit = { kind: 'edit', sourceId: SourceId };

export function matchRecordPathParams<X>(recordPathParams: RecordPathParams, cases: {
  root: (recordPathParams: RecordPathParamsRoot) => X,
  edit: (recordPathParams: RecordPathParamsEdit) => X,
}): X {
  switch (recordPathParams.kind) {
    case 'root': return cases.root(recordPathParams);
    case 'edit': return cases.edit(recordPathParams);
  }
}

export function recordPathParamsFromLocation(location: Location): RecordPathParams {
  const tryParseInt = (_default: number, s: string): number => { try { return parseInt(s); } catch { return _default; } };
  const {pathname: path, search} = location;
  const queries = queryString.parse(search);
  let match;
  match = matchPath<{}>(path, {path: '/', exact: true});
  if (match) return {kind: 'root'};
  match = matchPath<{sourceId: SourceId}>(path, {path: '/edit/:sourceId'});
  if (match) return {kind: 'edit', sourceId: decodeURIComponent(match.params.sourceId)};
  log.warn(`recordPathParamsFromLocation: Unknown location[${location}], returning {kind: root}`);
  return {kind: 'root'};
}

//
// SearchPathParams
//

// QUESTION Unify with Query? 1-1 so far
export type SearchPathParams =
  | SearchPathParamsRoot
  | SearchPathParamsRandom
  | SearchPathParamsSpecies
  | SearchPathParamsRec;
export type SearchPathParamsRoot    = { kind: 'root' };
export type SearchPathParamsRandom  = { kind: 'random',  filters: Filters, seed: number };
export type SearchPathParamsSpecies = { kind: 'species', filters: Filters, species: string };
export type SearchPathParamsRec     = { kind: 'rec',     filters: Filters, sourceId: SourceId };

export function matchSearchPathParams<X>(searchPathParams: SearchPathParams, cases: {
  root:    (searchPathParams: SearchPathParamsRoot)    => X,
  random:  (searchPathParams: SearchPathParamsRandom)  => X,
  species: (searchPathParams: SearchPathParamsSpecies) => X,
  rec:     (searchPathParams: SearchPathParamsRec)     => X,
}): X {
  switch (searchPathParams.kind) {
    case 'root':    return cases.root(searchPathParams);
    case 'random':  return cases.random(searchPathParams);
    case 'species': return cases.species(searchPathParams);
    case 'rec':     return cases.rec(searchPathParams);
  }
}

// TODO(put_all_query_state_in_location)
export function searchPathParamsFromLocation(location: Location): SearchPathParams {
  const tryParseInt = (_default: number, s: string): number => { try { return parseInt(s); } catch { return _default; } };
  const {pathname: path, search} = location;
  const queries = queryString.parse(search);
  let match;
  match = matchPath<{}>(path, {path: '/', exact: true});
  if (match) return {kind: 'root'};
  match = matchPath<{seed: string}>(path, {path: '/random/:seed'});
  if (match) return {kind: 'random',  filters: {}, seed: tryParseInt(0, decodeURIComponent(match.params.seed))};
  match = matchPath<{species: string}>(path, {path: '/species/:species'});
  if (match) return {kind: 'species', filters: {}, species: decodeURIComponent(match.params.species)};
  match = matchPath<{sourceId: SourceId}>(path, {path: '/rec/:sourceId*'});
  if (match) return {kind: 'rec',     filters: {}, sourceId: decodeURIComponent(match.params.sourceId)};
  log.warn(`searchPathParamsFromLocation: Unknown location[${location}], returning {kind: root}`);
  return {kind: 'root'};
}
