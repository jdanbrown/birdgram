import _ from 'lodash';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { debug_print } from './log';
import { Places } from './places';
import { match, matchUndefined, Omit, parseUrl } from './utils';

//
// SourceId
//

export type SourceId = string;
export type Clip     = {start_s: number, end_s: number};

export function SourceId(
  type: 'xc' | 'user',
  name: string,
  opts: {
    clip?: Clip,
  } = {},
): SourceId {
  const query = matchUndefined(opts.clip, {
    undefined: ()                 => '',
    x:         ({start_s, end_s}) => `?[${start_s},${end_s}]`,
  });
  return `${type}:${name}${query}`
}

export function matchSourceId<X>(sourceId: SourceId, cases: {
  xc:   (x: {clip?: Clip, xc_id: number}) => X,
  user: (x: {clip?: Clip, name: string})  => X,
}): X {
  const {protocol, pathname, query} = parseUrl(sourceId);
  const type = protocol.split(':')[0];
  const clip = !query.clip ? undefined : JSON.parse(query.clip);
  return match(type,
    ['xc',   () => cases.xc   ({clip, xc_id: parseInt(pathname)})],
    ['user', () => cases.user ({clip, name: pathname})],
  );
}

// Human-friendly display for a sourceId, e.g.
//  - 'xc:123456' -> 'XC123456'
//  - 'user:<timestamp>-<hash>.wav?clip=[0,15]' -> [TODO More human friendly]
export function showSourceId(sourceId: SourceId): string {
  return matchSourceId(sourceId, {
    xc:   ({xc_id, clip}) => `XC${xc_id}`   + (!clip ? '' : `(${clip.start_s}-${clip.end_s})`),
    user: ({name,  clip}) => `user:${name}` + (!clip ? '' : `(${clip.start_s}-${clip.end_s})`),
  });
}

//
// Rec
//

export interface Rec {

  // bubo
  source_id: SourceId; // More appropriate than id for mobile (see python util.rec_to_source_id for details)
  // id: string;       // Hide so that we don't accidentally use it (we'll get type errors)

  // xc
  xc_id: number;
  species: string;             // (From ebird)
  species_taxon_order: string; // (From ebird)
  species_com_name: string;    // (From xc)
  species_sci_name: string;    // (From xc)
  recs_for_sp: number;
  quality: Quality;
  lat?: number;
  lng?: number;
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

// TODO Make separate subtypes XCRec, UserRec <: Rec
export interface XCRec extends Rec {
  // kind: 'xc', // TODO
}

export interface UserRec extends Rec {
  // kind: 'user', // TODO
  f_preds: Array<number>;
}

export type Quality = 'A' | 'B' | 'C' | 'D' | 'E' | 'no score';
export type F_Preds = Array<number>;

export function matchRec<X>(rec: Rec, cases: {
  xc:   (rec: XCRec,   sourceId: {xc_id: number}) => X,
  user: (rec: UserRec, sourceId: {name: string})  => X,
}): X {
  // HACK Switch on .source_id until we refactor all Rec constructors to include .kind
  return matchSourceId(rec.source_id, {
    xc:   ({xc_id}) => cases.xc   (rec as XCRec,   {xc_id}),
    user: ({name})  => cases.user (rec as UserRec, {name}),
  });
}

export function rec_f_preds(rec: Rec): Rec_f_preds {
  return matchRec(rec, {
    xc:   rec => rec as unknown as Rec_f_preds,                               // Expose .f_preds_* from sqlite
    user: rec => _.fromPairs(rec.f_preds.map((p, i) => [`f_preds_${i}`, p])), // Materialize {f_preds_*:p} from .f_preds
  });
}

export interface Rec_f_preds {
  [key: string]: number;
}

export const Rec = {

  spectroPath: (rec: Rec): string => matchRec(rec, {
    xc:   rec             => SearchRecs.assetPath('spectro', rec.species, rec.xc_id, 'png'),
    user: (rec, sourceId) => UserRec.spectroPath(sourceId.name),
  }),

  audioPath: (rec: Rec): string => matchRec(rec, {
    xc:   rec             => SearchRecs.assetPath('audio', rec.species, rec.xc_id, 'mp4'),
    user: (rec, sourceId) => UserRec.audioPath(sourceId.name),
  }),

  hasCoords: (rec: Rec): boolean => {
    return !_.isNil(rec.lat) && !_.isNil(rec.lng);
  },

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

  xcUrl: (rec: Rec): string => {
    return `https://www.xeno-canto.org/${rec.xc_id}`;
  },

  speciesUrl: (rec: Rec): string => {
    return `https://www.allaboutbirds.org/guide/${rec.species_com_name.replace(/ /g, '_')}`;
  },

  // TODO Open in user's preferred map app i/o google maps
  //  - Docs for zoom levels: https://developers.google.com/maps/documentation/maps-static/dev-guide#Zoomlevels
  mapUrl: (rec: Rec, opts: {zoom: number}): string | null => {
    if (!Rec.hasCoords(rec)) {
      // TODO How to zoom when we don't know (lat,lng)?
      const {place} = rec;
      return `https://maps.google.com/maps?oi=map&q=${place}`;
    } else {
      // TODO How to show '$place' as label instead of '$lat,$lng'?
      const {lat, lng, place} = rec;
      const {zoom} = opts;
      return `https://www.google.com/maps/place/${lat},${lng}/@${lat},${lng},${zoom}z`;
    }
  },

};

export const UserRec = {

  audioPath: (name: string): string => {
    return `${fs.dirs.DocumentDir}/user-recs-v0/${name}`;
  },

  spectroPath: (name: string): string => {
    return `${fs.dirs.DocumentDir}/user-recs-v0/${name}.spectros/denoise=true.png`;
  },

};

//
// SearchPathParams
//

import { matchPath } from 'react-router-native';

import { log, rich } from './log';

export type SearchPathParams =
  | SearchPathParamsNone
  | SearchPathParamsRandom
  | SearchPathParamsSpecies
  | SearchPathParamsRec;
export type SearchPathParamsNone    = { kind: 'none' };
export type SearchPathParamsRandom  = { kind: 'random', seed: number };
export type SearchPathParamsSpecies = { kind: 'species', species: string };
export type SearchPathParamsRec     = { kind: 'rec', sourceId: SourceId };

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
  match = matchPath<{sourceId: SourceId}>(path, {path: '/rec/:sourceId*'});
  if (match) return {kind: 'rec', sourceId: decodeURIComponent(match.params.sourceId)};
  log.warn(`searchPathParamsFromPath: Unexpected path[${path}], returning {kind: none}`);
  return {kind: 'none'};
}

//
// Misc.
//

export const Models = {
  search: {
    path: `search_recs/models/search.json`,
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

  serverConfigPath: 'search_recs/server-config.json',

  // TODO Test asset paths on android (see notes in README)
  dbPath: 'search_recs/search_recs.sqlite3',

  // TODO(asset_main_bundle): Why implicitly relative to fs.dirs.MainBundleDir?
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
