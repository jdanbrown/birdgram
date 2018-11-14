import { Places } from './places';
import { Omit } from './utils';

//
// Rec
//

export type Quality = 'A' | 'B' | 'C' | 'D' | 'E' | 'no score';
export type RecId = string;

export interface Rec {

  // bubo
  id: RecId;

  // xc
  xc_id: number;
  species: string;             // (From ebird)
  species_taxon_order: string; // (From ebird)
  species_com_name: string;    // (From xc)
  species_sci_name: string;    // (From xc)
  recs_for_sp: number;
  quality: Quality;
  lat: number;
  lng: number;
  month_day: string;
  place: string;
  place_only: string;
  state: string;
  state_only: string;
  recordist: string;
  license_type: string;
  remarks: string;

  // search_recs / search output
  slp?: number;
  d_pc?: number;

}

// If xc, abbrev long path rec.id -> short '<species>/<xc_id>'
export function shortRecId(id: RecId): RecId {
  return (
    id.startsWith('cache/audio/xc/data/') ? id.split('/').slice(4, 6).join('/') :
    id
  );
}

export interface Rec_f_preds {
  [key: string]: number;
}

export function rec_f_preds(rec: Rec): Rec_f_preds {
  return rec as unknown as Rec_f_preds;
}

export const Rec = {

  spectroPath: (rec: Rec): string => SearchRecs.assetPath('spectro', rec.species, rec.xc_id, 'png'),
  audioPath:   (rec: Rec): string => SearchRecs.assetPath('audio',   rec.species, rec.xc_id, 'mp4'),

  placeNorm: (placeLike: string): string => {
    return placeLike.split(', ').reverse().map(x => Rec.placePartAbbrev(x)).join(', ');
  },
  placePartAbbrev: (part: string): string => {
    const ret = (
      Places.countryCodeFromName[part] ||
      Places.stateCodeFromName[part] ||
      part
    );
    return ret;
  },

};

//
// SearchPathParams
//

import { matchPath } from 'react-router-native';

export type SearchPathParams =
  | SearchPathParamsNone
  | SearchPathParamsRandom
  | SearchPathParamsSpecies
  | SearchPathParamsRec;
export type SearchPathParamsNone    = { kind: 'none' };
export type SearchPathParamsRandom  = { kind: 'random', seed: number };
export type SearchPathParamsSpecies = { kind: 'species', species: string };
export type SearchPathParamsRec     = { kind: 'rec', recId: string };

export function matchSearchPathParams<X>(searchPathParams: SearchPathParams, cases: {
  none:    (searchPathParams: SearchPathParamsNone)    => X,
  random:  (searchPathParams: SearchPathParamsRandom)  => X,
  species: (searchPathParams: SearchPathParamsSpecies) => X,
  rec:     (searchPathParams: SearchPathParamsRec)     => X,
}): X {
  switch (searchPathParams.kind) {
    case 'none':    return cases.none(searchPathParams);
    case 'random':  return cases.random(searchPathParams);
    case 'species': return cases.species(searchPathParams);
    case 'rec':     return cases.rec(searchPathParams);
  }
}

export function searchPathParamsFromPath(path: string): SearchPathParams {
  const tryParseInt = (_default: number, s: string): number => { try { return parseInt(s); } catch { return _default; } };
  let match;
  match = matchPath<{}>(path, {path: '/', exact: true});
  if (match) return {kind: 'none'};
  match = matchPath<{seed: string}>(path, {path: '/random/:seed'});
  if (match) return {kind: 'random', seed: tryParseInt(0, decodeURIComponent(match.params.seed))};
  match = matchPath<{species: string}>(path, {path: '/species/:species'});
  if (match) return {kind: 'species', species: decodeURIComponent(match.params.species)};
  match = matchPath<{recId: string}>(path, {path: '/rec/:recId*'});
  if (match) return {kind: 'rec', recId: decodeURIComponent(match.params.recId)};
  throw `searchPathParamsFromPath: Unexpected path: ${path}`;
}

//
// Misc.
//

export const Models = {
  search: {
    path: `search_recs/models/search.json`,
  },
};

export interface ModelsSearch {
  classes_: Array<string>;
}

export const SearchRecs = {

  serverConfigPath: 'search_recs/server-config.json',

  // TODO Test asset paths on android (see notes in README)
  dbPath: 'search_recs/search_recs.sqlite3',

  // TODO After verifying that asset dirs are preserved on android, simplify the basenames back to `${xc_id}.${format}`
  assetPath: (kind: string, species: string, xc_id: number, format: string): string => (
    `search_recs/${kind}/${species}/${kind}-${species}-${xc_id}.${format}`
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
