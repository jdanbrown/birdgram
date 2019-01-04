import jsonStableStringify from 'json-stable-stringify';
import _ from 'lodash';
import moment from 'moment';
import queryString from 'query-string';
import { matchPath } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
const fs = RNFB.fs;

import { Filters } from './components/SearchScreen';
import { GeoCoords } from './components/Geo';
import { BarchartProps } from './ebird';
import { debug_print, log, Log, rich } from './log';
import { Places } from './places';
import { Location } from './router';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifNull, ifUndefined, json, JsonSafeNumber, Interval,
  local, mapEmpty, mapUndefined, match, matchNull, matchUndefined, Omit, parseUrl, parseUrlNoQuery, parseUrlWithQuery,
  pretty, requireSafePath, safeParseInt, safePath, showDate, showSuffix, splitFirst, throw_, typed, unjson,
} from './utils';
import { XC } from './xc';

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
// Source / SourceId
//

export type SourceId = string;
export type Source = XCSource | UserSource | EditSource;
export interface XCSource   { kind: 'xc';   xc_id: number; }
export interface UserSource { kind: 'user'; name: string; }
export interface EditSource { kind: 'edit'; edit: Edit; }

export interface SourceShowOpts {
  species: XC | null; // Show species if xc sourceId (using XC dep)
  long?: boolean;
}

export const SourceId = {

  // Omitted b/c noops
  // stringify: (x: SourceId): string   => ...
  // parse:     (x: string):   SourceId => ...

  split: (sourceId: SourceId): {kind: string, ssp: string} => {
    const [kind, ssp] = splitFirst(sourceId, ':');
    return {kind, ssp};
  },

  show: (sourceId: SourceId, opts: SourceShowOpts): string => {
    return Source.show(Source.parse(sourceId), opts);
  },

};

export const Source = {

  // NOTE Must map {kind:'xc',xc_id} -> `xc:${xc_id}` for consistency with db payload (see util.py:rec_to_source_id)
  stringify: (source: Source): SourceId => {
    return matchSource(source, {
      xc:   ({xc_id}) => `xc:${xc_id}`,
      user: ({name})  => `user:${name}`,
      edit: ({edit})  => `edit:${Edit.stringify(edit)}`, // Something that nests, e.g. for edits of edits
    });
  },

  parse: (sourceId: SourceId): Source => {
    const {kind, ssp} = SourceId.split(sourceId);
    return match<string, Source>(kind,
      ['xc',          () => ({kind: 'xc',   xc_id: safeParseInt(ssp)})],
      ['user',        () => ({kind: 'user', name: ssp})],
      ['edit',        () => ({kind: 'edit', edit: Edit.parse(ssp)})],
      [match.default, () => { throw `Unknown sourceId type: ${sourceId}`; }],
    );
  },

  show: (source: Source, opts: SourceShowOpts): string => {
    return matchSource(source, {
      xc: ({xc_id}) => {
        const xc = opts.species;
        return [
          `XC${xc_id}`,
          !xc ? '' : ` (${xc.speciesFromXCID.get(xc_id) || '?'})`,
        ].join('');
      },
      user: ({name}) => {
        // Ignore hash (in name=`${date}-${time}-${hash}`), since seconds resolution should be unique enough for human
        //  - TODO Rethink after rec sharing (e.g. add usernames to avoid collisions)
        return [
          !opts.long ? '' : 'Recording: ',
          showDate(Source._parseUserName(name)),
        ].join('');
      },
      edit: ({edit}) => {
        return Edit.show(edit, opts);
      },
    });
  },

  // QUESTION Should we pass around Moment's i/o Date's?
  _parseUserName: (name: string): Date => {
    return moment(
      name.replace(/[^\d]/g, '').slice(0, 14), // Strip all non-numbers to be robust to changing formats
      'YYYYMMDDhhmmss',
    ).toDate();
  },

};

export function matchSourceId<X>(sourceId: SourceId, cases: {
  xc:   (source: XCSource)   => X,
  user: (source: UserSource) => X,
  edit: (source: EditSource) => X,
}): X {
  return matchSource(Source.parse(sourceId), cases);
}

export function matchSource<X>(source: Source, cases: {
  xc:   (source: XCSource)   => X,
  user: (source: UserSource) => X,
  edit: (source: EditSource) => X,
}): X {
  switch(source.kind) {
    case 'xc':   return cases.xc   (source);
    case 'user': return cases.user (source);
    case 'edit': return cases.edit (source);
  }
}

// HACK Refactor callers so we don't need this
export function matchUserSourceId<X>(sourceId: SourceId,
  f: (x: {name: string}) => X,
): X {
  return matchSourceId(sourceId, {
    xc:   () => { throw `Expected user sourceId, got: ${sourceId}`; },
    user: x  => f(x),
    edit: () => { throw `Expected user sourceId, got: ${sourceId}`; },
  });
}

//
// Rec
//

// TODO How to prevent callers from constructing a Rec i/o one of the subtypes? [Maybe classes with constructors?]
export type Rec = XCRec | UserRec | EditRec;

export interface XCRec extends _RecImpl {
  kind:  'xc';
  xc_id: number;
}

export interface UserRec extends _RecImpl {
  kind:    'user';
  f_preds: Array<number>;
}

export interface EditRec extends _RecImpl {
  kind:      'edit';
  edit:      Edit; // Redundant with .source_id, included as field for convenience (burden on producer)
  // parent: Rec;  // Require consumers to load parents as needed, else we risk O(forks) work to load an edit rec
  f_preds:   Array<number>;
}

export interface _RecImpl {

  // bubo
  source_id: SourceId; // More appropriate than id for mobile (see python util.rec_to_source_id for details)
  // id: string;       // Hide so that we don't accidentally use it (so that we'll get type errors if we try)
  duration_s: number;

  // xc
  species:               string; // (From ebird)
  species_taxon_order:   string; // (From ebird)
  species_species_group: string; // (From ebird)
  species_family:        string; // (From ebird)
  species_order:         string; // (From ebird)
  species_com_name:      string; // (From xc)
  species_sci_name:      string; // (From xc)
  recs_for_sp:           number;
  quality:               Quality;
  lat?:                  number;
  lng?:                  number;
  date:                  string; // sqlite datetime
  month_day:             string; // sqlite string
  year:                  number; // sqlite bigint
  place:                 string;
  place_only:            string;
  state:                 string;
  state_only:            string;
  recordist:             string;
  license_type:          string;
  remarks:               string;

  // HACK Provided only from SearchScreen.loadRecsFromQuery -> rec case
  //  - TODO Split out SearchRecResult for /rec, separate from both query_rec and search results for /species, /random, etc.
  slp?:  number;
  d_pc?: number;

}

export type Quality = 'A' | 'B' | 'C' | 'D' | 'E' | 'no score';
export type F_Preds = Array<number>;

export const Quality = {
  values: ['A', 'B', 'C', 'D', 'E', 'no score'] as Array<Quality>,
};

export function matchRec<X>(rec: Rec, cases: {
  xc:   (rec: XCRec,   source: XCSource)   => X,
  user: (rec: UserRec, source: UserSource) => X,
  edit: (rec: EditRec, source: EditSource) => X,
}): X {
  // HACK Switch on rec.source_id until we refactor all Rec constructors to include .kind
  return matchSourceId(rec.source_id, {
    xc:   source => cases.xc   (rec as XCRec,   source),
    user: source => cases.user (rec as UserRec, source),
    edit: source => cases.edit (rec as EditRec, source),
  });
}

export interface Rec_f_preds {
  [key: string]: number;
}

export interface SpectroPathOpts {
  f_bins:  number;
  denoise: boolean;
}

export const Rec = {

  // Primary storage (DocumentDir)
  userRecDir: `${fs.dirs.DocumentDir}/user-recs-v0`,  // User recordings
  editDir:    `${fs.dirs.DocumentDir}/edits-v0`,      // User edits
  // Files that can be regenerated by need (CacheDir)
  spectroCacheDir: `${fs.dirs.CacheDir}/spectros-v0`, // Spectros will recreate if missing

  audioPath: (rec: Rec): string => matchRec(rec, {
    xc:   (rec, source) => XCRec.audioPath(rec),
    user: (rec, source) => UserRec.audioPath(source),
    edit: (rec, source) => EditRec.audioPath(source),
  }),

  spectroPath: (
    rec:  Rec,
    opts: SpectroPathOpts, // Ignored for xc rec [TODO Clean up]
  ): string => matchRec(rec, {
    xc:   (rec, source) => XCRec.spectroPath(rec),
    user: (rec, source) => UserRec.spectroPath(source, opts),
    edit: (rec, source) => EditRec.spectroPath(source, opts),
  }),

  // HACK Shared across UserRec.spectroPath + RecordScreen:EditRecording
  //  - EditRecording can't use UserRec.spectroPath because it assumes a user sourceId
  //  - EditRecording can't use Rec.spectroPath because it maps xc sourceIds to their static asset spectroPath (40px)
  //  - TODO Add concept for "writable spectroPath for user/xc rec"
  spectroCachePath: (sourceId: SourceId, opts: SpectroPathOpts): string => {
    return `${Rec.spectroCacheDir}/${safePath(sourceId)}.spectros/f_bins=${opts.f_bins},denoise=${opts.denoise}.png`;
  },

  f_preds: (rec: Rec): Rec_f_preds => {
    return matchRec(rec, {
      xc:   rec => rec as unknown as Rec_f_preds,                               // Expose .f_preds_* from sqlite
      user: rec => _.fromPairs(rec.f_preds.map((p, i) => [`f_preds_${i}`, p])), // Materialize {f_preds_*:p} from .f_preds
      edit: rec => _.fromPairs(rec.f_preds.map((p, i) => [`f_preds_${i}`, p])), // Materialize {f_preds_*:p} from .f_preds
    });
  },

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

  recUrl: (rec: Rec): string | null => {
    return matchRec(rec, {
      xc:   rec => XCRec.recUrl(rec),
      user: rec => null,
      edit: rec => null, // TODO(user_metadata): Requires reading .metadata.json (see DB.loadRec)
    });
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

export const XCRec = {

  audioPath: (rec: XCRec): string => {
    return SearchRecs.assetPath('audio', rec.species, rec.xc_id, 'mp4');
  },

  // TODO (When needed)
  // sourceFromAudioFilename: (filename: string): XCSource => {
  //   ...
  // },

  spectroPath: (rec: XCRec): string => {
    // From assets, not spectroCacheDir
    return SearchRecs.assetPath('spectro', rec.species, rec.xc_id, 'png');
  },

  recUrl: (rec: XCRec): string => {
    return `https://www.xeno-canto.org/${rec.xc_id}`;
  },

};

export const UserRec = {

  log: new Log('UserRec'),

  audioPath: (source: UserSource): string => {
    return `${Rec.userRecDir}/${source.name}`;
  },

  sourceFromAudioFilename: (filename: string): UserSource => {
    return Source.parse(`user:${filename}`) as UserSource; // HACK Type
  },

  spectroPath: (source: UserSource, opts: SpectroPathOpts): string => {
    return Rec.spectroCachePath(Source.stringify(source), opts);
  },

  newAudioPath: async (ext: string, metadata: {
    coords: GeoCoords | null,
  }): Promise<string> => {

    // Create audioPath
    const timestamp = safePath( // Avoid ':' for ios paths
      new Date().toISOString()
      .slice(0, 19)                            // Cosmetic: drop millis/tz
      .replace(/[-:]/g, '').replace(/T/g, '-') // Cosmetic: 'YYYY-MM-DDThh:mm:ss' -> 'YYYYMMDD-hhmmss'
    );
    const hash = chance.hash({length: 8}); // Long enough to be unique across users
    const audioPath = `${Rec.userRecDir}/${timestamp}-${hash}.${ext}`;

    // Ensure parent dir for caller
    //  - Also for us, to write the metadata file (next step)
    ensureParentDir(audioPath);

    // Save metadata
    const metadataPath = `${audioPath}.metadata.json`;
    const metadataData = pretty(metadata);
    await fs.createFile(metadataPath, metadataData, 'utf8');

    // Done
    UserRec.log.info('newAudioPath', rich({audioPath, metadataPath, metadata}));
    return audioPath;

  },

  listAudioPaths: async (exts?: Array<string>): Promise<Array<string>> => {
    return (await UserRec.listAudioFilenames(exts)).map(x => `${Rec.userRecDir}/${x}`);
  },

  listAudioFilenames: async (_exts?: Array<string>): Promise<Array<string>> => {
    // HACK Filter exts to determine audio files vs. others (e.g. .metadata.json)
    const exts = _exts || ['wav', 'mp4', 'm4a', 'aac', 'mp3'];
    return (
      (await fs.ls(await ensureDir(Rec.userRecDir)))
      .sort()
      .filter(x => exts.includes(extname(x).replace(/^\./, '')))
    );
  },

};

export const EditRec = {

  log: new Log('EditRec'),

  audioPath: (source: EditSource): string => {
    const filename = `${SourceId.split(Source.stringify(source)).ssp}`; // Don't include 'edit:'
    return `${Rec.editDir}/${filename}`;
  },

  sourceFromAudioFilename: (filename: string): EditSource => {
    return Source.parse(`edit:${filename}`) as EditSource; // HACK Type
  },

  spectroPath: (source: EditSource, opts: SpectroPathOpts): string => {
    return Rec.spectroCachePath(Source.stringify(source), opts);
  },

  newAudioPath: async (edit: Edit, metadata: {
    // Include all parent rec fields in metadata
    //  - Includes all rec metadata (species), but no rec assets (audio, spectro)
    //  - [Unstable api] Also happens to include f_preds because we store them in the db i/o as an asset file
    //  - Namespace under .parent, so that non-inheritable stuff like species/quality/duration_s doesn't get confused
    parent: Rec,
  }): Promise<string> => {

    // Create audioPath
    const source: EditSource = {kind: 'edit', edit};
    const audioPath = EditRec.audioPath(source);

    // Ensure parent dir for caller
    //  - Also for us, to write the metadata file (next step)
    await ensureParentDir(audioPath);

    // Save metadata
    const metadataPath = `${audioPath}.metadata.json`;
    const metadataData = pretty(metadata);
    await fs.createFile(metadataPath, metadataData, 'utf8');

    // Done
    EditRec.log.info('newAudioPath', rich({audioPath, metadataPath, edit, metadata}));
    return audioPath;

  },

  listAudioPaths: async (): Promise<Array<string>> => {
    return (await EditRec.listAudioFilenames()).map(x => `${Rec.editDir}/${x}`);
  },

  listAudioFilenames: async (): Promise<Array<string>> => {
    return (
      (await fs.ls(await ensureDir(Rec.editDir)))
      .sort()
    );
  },

};

//
// Edit
//

// A (commited) edit, which has a well defined mapping to an edit rec file
export interface Edit extends DraftEdit {
  parent: SourceId;
  hash:   string; // Ensure a new filename for each EditRec (so we don't have to think about file collisions / purity)
}

// A draft edit, which isn't yet associated with any edit rec file
//  - For RecordScreen.state
export interface DraftEdit {
  clips?: Array<Interval>;
  // Room to grow: freq filter, gain adjust, airbrush, ...
}

export const Edit = {

  // Can't simply json/unjson because it's unsafe for Interval (which needs to JsonSafeNumber)
  stringify: (edit: Edit): string => {
    return queryString.stringify(typed<{[key in keyof Edit]: undefined | string | string[]}>({
      hash:   edit.hash,
      parent: edit.parent,
      clips:  mapUndefined(edit.clips, x => x.map(x => x.stringify())),
    }));
  },
  parse: (x: string | {[key: string]: undefined | string | string[]}): Edit => {
    const q = typeof x === 'string' ? queryString.parse(x) : x;
    return {
      parent: Edit._asSingle(ifUndefined(q.parent, () => throw_(`Field 'parent' required: ${json(q)}`))),
      hash:   Edit._asSingle(ifUndefined(q.hash,   () => throw_(`Field 'hash' required: ${json(q)}`))),
      clips:  mapUndefined(q.clips, x => Edit._asArray(x).map(Interval.parse)),
    };
  },

  show: (edit: Edit, opts: SourceShowOpts): string => {
    const parts = [
      // Ignore hash for human
      SourceId.show(edit.parent, opts),
      ...(edit.clips || []).map(x => x.show()),
    ];
    return (parts
      .filter(x => !_.isEmpty(x)) // Exclude null, undefined, '' (and [], {})
      .join(' ')
    );
  },

  // Handle the `undefined | string | string[]` fields from queryString.parse
  _asSingle: (x: string | string[]): string   => typeof x === 'string' ? x   : throw_(`Expected string: ${json(x)}`),
  _asArray:  (x: string | string[]): string[] => typeof x === 'string' ? [x] : x,

};

export const DraftEdit = {

  hasEdits: (edit: DraftEdit): boolean => {
    return !_(edit).values().every(_.isEmpty);
  },

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
  // debug_print('searchPathParamsFromLocation', {location, queries}); // XXX Debug
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

  // TODO After verifying that asset dirs are preserved on android, simplify the basenames back to `${xc_id}.${format}`
  assetPath: (kind: string, species: string, xc_id: number, format: string): string => (
    `${fs.dirs.MainBundleDir}/search_recs/${kind}/${species}/${kind}-${species}-${xc_id}.${format}`
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
