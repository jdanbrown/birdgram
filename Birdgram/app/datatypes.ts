import jsonStableStringify from 'json-stable-stringify';
import _ from 'lodash';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { Filters } from './components/SearchScreen';
import { BarchartProps } from './ebird';
import { debug_print } from './log';
import { Places } from './places';
import { match, matchNull, matchUndefined, Omit, parseUrl } from './utils';

//
// Species
//

export type Species     = string; // = SpeciesMetadata.shorthand
export type SpeciesCode = string; // = SpeciesMetadata.species_code

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
// Places (= lists of species)
//

export interface Place {
  name:    string;
  props:   BarchartProps;
  species: Array<Species>;
}

export const Place = {

  id: ({props}: Place): string => {
    return jsonStableStringify(props); // A bit verbose in our locations, but simple and sound
  },

  find: (id: string, places: Array<Place>): Place | undefined => {
    return _.find(places, place => id === Place.id(place))
  },

};

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

export const Quality = {
  values: ['A', 'B', 'C', 'D', 'E', 'no score'] as Array<Quality>,
};

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

import { Location } from 'history';
import { matchPath } from 'react-router-native';
import queryString from 'query-string';

import { log, rich } from './log';
import { pretty } from './utils';

// QUESTION Unify with Query? 1-1 so far
export type SearchPathParams =
  | SearchPathParamsNone
  | SearchPathParamsRandom
  | SearchPathParamsSpecies
  | SearchPathParamsRec;
export type SearchPathParamsNone    = { kind: 'none' };
export type SearchPathParamsRandom  = { kind: 'random',  filters: Filters, seed: number };
export type SearchPathParamsSpecies = { kind: 'species', filters: Filters, species: string };
export type SearchPathParamsRec     = { kind: 'rec',     filters: Filters, sourceId: SourceId };

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

// TODO(add_filters_to_location)
export function searchPathParamsFromLocation(location: Location): SearchPathParams {
  const tryParseInt = (_default: number, s: string): number => { try { return parseInt(s); } catch { return _default; } };
  const {pathname: path, search} = location;
  const queries = queryString.parse(search);
  debug_print('searchPathParamsFromLocation', {location, queries});
  let match;
  match = matchPath<{}>(path, {path: '/', exact: true});
  if (match) return {kind: 'none'};
  match = matchPath<{seed: string}>(path, {path: '/random/:seed'});
  if (match) return {kind: 'random',  filters: {}, seed: tryParseInt(0, decodeURIComponent(match.params.seed))};
  match = matchPath<{species: string}>(path, {path: '/species/:species'});
  if (match) return {kind: 'species', filters: {}, species: decodeURIComponent(match.params.species)};
  match = matchPath<{sourceId: SourceId}>(path, {path: '/rec/:sourceId*'});
  if (match) return {kind: 'rec',     filters: {}, sourceId: decodeURIComponent(match.params.sourceId)};
  log.warn(`searchPathParamsFromLocation: Unknown location[${location}], returning {kind: none}`);
  return {kind: 'none'};
}

//
// App globals
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

  serverConfigPath:    'search_recs/server-config.json',
  metadataSpeciesPath: 'search_recs/metadata/species.json',
  dbPath:              'search_recs/search_recs.sqlite3', // TODO Test asset paths on android (see notes in README)

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

export type MetadataSpecies = Array<SpeciesMetadata>;
