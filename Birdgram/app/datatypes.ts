import { EventEmitter } from 'fbemitter';
import jsonStableStringify from 'json-stable-stringify';
import _ from 'lodash';
import moment from 'moment';
import queryString from 'query-string';
import { AsyncStorage } from 'react-native';
import { matchPath } from 'react-router-native';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
import traverse from 'traverse';
const {base64, fs} = RNFB;

import { Filters } from './components/SearchScreen';
import { GeoCoords } from './components/Geo';
import { BarchartProps } from './ebird';
import { debug_print, log, Log, rich } from './log';
import { NativeSpectro } from './native/Spectro';
import { Places } from './places';
import { Location } from './router';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchNull, matchUndefined, NoKind,
  Omit, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane, requireSafePath, safeParseInt, safeParseIntOrNull,
  safePath, showDate, showSuffix, splitFirst, stripExt, throw_, tryElse, typed, unjson,
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
export interface UserSource { kind: 'user';
  created:  Date;
  uniq:     string;
  ext:      string;
  filename: string; // Preserve so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
}
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

  stripType: (sourceId: SourceId): string => {
    return SourceId.split(sourceId).ssp;
  },

  show: (sourceId: SourceId, opts: SourceShowOpts): string => {
    return matchNull(Source.parse(sourceId), {
      x:    source => Source.show(source, opts),
      null: ()     => `[Malformed: ${sourceId}]`,
    });
  },

};

export const Source = {

  // NOTE Preserve {kind:'xc',xc_id} -> `xc:${xc_id}` to stay consistent with db payload (see util.py:rec_to_source_id)
  stringify: (source: Source): SourceId => {
    return matchSource(source, {
      xc:   ({xc_id}) => `xc:${xc_id}`,
      user: source    => `user:${UserSource.stringify(source)}`,
      edit: ({edit})  => `edit:${Edit.stringify(edit)}`, // Ensure can nest safely (e.g. edits of edits)
    });
  },

  parse: (sourceId: SourceId): Source | null => {
    const {kind, ssp} = SourceId.split(sourceId);
    return match<string, Source | null>(kind,
      ['xc',          () => mapNull(safeParseIntOrNull(ssp), xc_id => typed<XCSource>   ({kind: 'xc',   xc_id}))],
      ['user',        () => mapNull(UserSource.parse(ssp),   x     => typed<UserSource> ({kind: 'user', ...x}))],
      ['edit',        () => mapNull(Edit.parse(ssp),         edit  => typed<EditSource> ({kind: 'edit', edit}))],
      [match.default, () => { throw `Unknown sourceId type: ${sourceId}`; }],
    );
  },

  parseOrFail: (sourceId: SourceId): Source => {
    return matchNull(Source.parse(sourceId), {
      null: ()     => { throw `Failed to parse sourceId[${sourceId}]`; },
      x:    source => source,
    });
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
      user: ({created, uniq, ext}) => {
        // Ignore uniq (in name=`${date}-${time}-${uniq}`), since seconds resolution should be unique enough for human
        //  - TODO Rethink after rec sharing (e.g. add usernames to avoid collisions)
        return [
          !opts.long ? '' : 'Recording: ',
          showDate(created),
        ].join('');
      },
      edit: ({edit}) => {
        return Edit.show(edit, opts);
      },
    });
  },

  stringifyDate: (d: Date): string => {
    return (d
      .toISOString()         // Render from local to utc
      .replace(/[^\d]/g, '') // Strip 'YYYY-MM-DDThh:mm:ss.SSS' -> 'YYYYMMDDThhmmssSSS'
    );
  },

  // [Maybe simpler to pass around Moment i/o Date?]
  parseDate: (s: string): Date => {
    return moment.utc(         // Parse from utc into local
      s.replace(/[^\d]/g, ''), // Strip all non-numbers to be robust to changing formats
      'YYYYMMDDhhmmssSSS',     // This is robust to missing values at the end (e.g. no SSS -> millis=0)
    ).toDate();
  },

  // Examples
  //  - xc:   'xc-1234'
  //  - user: 'user-20190109205640977-336b2bb7'
  //  - clip: 'clip-20190108011526401-d892e4be'
  pathBasename: (source: Source): string => {
    return requireSafePath(
      matchSource(source, {
        xc:   source => XCRec.pathBasename(source),
        user: source => UserRec.pathBasename(source),
        edit: source => EditRec.pathBasename(source),
      }),
    );
  },

};

export const UserSource = {

  stringify: (source: NoKind<UserSource>): string => {
    // Return preserved filename so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
    return source.filename;
  },

  parse: (ssp: string): NoKind<UserSource> | null => {
    try {
      // Format: 'user-${created}-${uniq}.${ext}'
      //  - Assume uniq has no special chars, and let created + ext match everything outside of '-' and '.'
      //  - And make 'user-' optional for back compat [TODO(edit_rec): Test if this is actually required for Saved/Recents to load]
      const match = ssp.match(/^(?:user-)?(.+)-([^-.]+)\.(.+)$/);
      if (!match)   throw `UserSource.parse: Invalid ssp[${ssp}]`;
      const [_, created, uniq, ext] = match;
      if (!created) throw `UserSource.parse: Invalid created[${created}]`;
      if (!uniq)    throw `UserSource.parse: Invalid uniq[${uniq}]`;
      if (!ext)     throw `UserSource.parse: Invalid ext[${created}]`;
      return {
        created: Source.parseDate(created),
        uniq,
        ext,
        filename: ssp, // Preserve so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
      };
    } catch (e) {
      log.warn('UserSource.parse: Failed', rich({ssp, e}));
      return null;
    }
  },

  new: (source: Omit<UserSource, 'kind' | 'filename'>): UserSource => {
    return typed<UserSource>({
      kind:     'user',
      filename: UserSource._newFilename(source), // Nothing to preserve for fresh user rec (e.g. not from saved file)
      ...source,
    });
  },

  _newFilename: (source: Omit<UserSource, 'kind' | 'filename'>): string => {
    // Generate filename from UserSource metadata
    return `user-${Source.stringifyDate(source.created)}-${source.uniq}.${source.ext}`;
  },

};

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

// Prefer matchSource(source) to avoid having to handle the null case when sourceId fails to parse
export function matchSourceId<X>(sourceId: SourceId, cases: {
  null: (sourceId: SourceId) => X,
  xc:   (source: XCSource)   => X,
  user: (source: UserSource) => X,
  edit: (source: EditSource) => X,
}): X {
  return matchNull(Source.parse(sourceId), {
    null: ()     => cases.null(sourceId),
    x:    source => matchSource(source, cases),
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
  edit:      Edit;
  // parent: Rec;  // Require consumers to load the parent Rec, else we risk O(parents) work to load an edit rec
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
    null: sourceId => { throw `matchRec: No Rec should have an invalid sourceId: ${sourceId}`; },
    xc:   source   => cases.xc   (rec as XCRec,   source),
    user: source   => cases.user (rec as UserRec, source),
    edit: source   => cases.edit (rec as EditRec, source),
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
  userRecDir:      `${fs.dirs.DocumentDir}/user-recs-v0`,  // User recordings
  editDir:         `${fs.dirs.DocumentDir}/edits-v0`,      // User edits
  // Files that can be regenerated by need (CacheDir)
  spectroCacheDir: `${fs.dirs.CacheDir}/spectros-v0`, // Spectros will recreate if missing

  // Events
  emitter: new EventEmitter(),

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

  // A writable spectroCachePath for nonstandard f_bins/denoise
  //  - HACK Shared across UserRec.spectroPath + RecordScreen:EditRecording
  //    - EditRecording can't use UserRec.spectroPath because it assumes a user sourceId
  //    - EditRecording can't use Rec.spectroPath because it maps xc sourceIds to their static asset spectroPath (40px)
  //    - TODO Add concept for "writable spectroPath for user/xc rec"
  spectroCachePath: (source: Source, opts: SpectroPathOpts): string => {
    return [
      `${Rec.spectroCacheDir}`,
      `${Source.pathBasename(source)}.spectros`,
      `f_bins=${opts.f_bins},denoise=${opts.denoise}.png`,
    ].join('/');
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

  listAudioSources: async <FileSource>(FileRec: {
    audioDir:                string,
    sourceFromAudioFilename: (filename: string) => Promise<FileSource | null>,
  }): Promise<Array<FileSource>> => {
    return _.flatten(await Promise.all(
      (await Rec.listAudioFilenames(FileRec.audioDir)).map(async filename => {
        const source = await FileRec.sourceFromAudioFilename(filename);
        if (!source) {
          log.warn("listAudioSources: Dropping: Failed to parse sourceId", rich({
            audioDir: FileRec.audioDir,
            filename,
          }));
          return [];
        } else {
          return [source];
        }
      })
    ));
  },

  listAudioPaths: async (dir: string): Promise<Array<string>> => {
    return (await Rec.listAudioFilenames(dir)).map(x => `${dir}/${x}`);
  },

  listAudioFilenames: async (dir: string): Promise<Array<string>> => {
    const excludes = [/\.metadata\.json$/];
    return (
      (await fs.ls(await ensureDir(dir)))
      .filter(x => !_.some(excludes, exclude => exclude.test(x)))
      .sort()
    );
  },

};

export const XCRec = {

  pathBasename: (source: XCSource): string => {
    return safePath(Source.stringify(source)); // e.g. 'xc-1234'
  },

  audioPath: (rec: XCRec): string => {
    return SearchRecs.assetPath('audio', rec.species, rec.xc_id, 'mp4');
  },

  // TODO (When needed)
  // sourceFromAudioFilename: async (filename: string): Promise<XCSource | null> => {
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

  audioDir: Rec.userRecDir,

  pathBasename: (source: UserSource): string => {
    return stripExt(source.filename); // e.g. 'user-20190109205640977-336b2bb7'
  },

  audioPath: (source: UserSource): string => {
    // Use preserved filename so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
    return `${UserRec.audioDir}/${source.filename}`;
  },

  sourceFromAudioFilename: async (filename: string): Promise<UserSource | null> => {
    return mapNull(Source.parse(`user:${filename}`), x => x as UserSource); // HACK Type
  },

  spectroPath: (source: UserSource, opts: SpectroPathOpts): string => {
    return Rec.spectroCachePath(source, opts);
  },

  new: async (audioPath: string): Promise<UserSource> => {

    // Make UserSource
    const userSource = await UserRec.sourceFromAudioFilename(basename(audioPath));
    if (!userSource) throw `stopRecording: audioPath from Nativespectro.stop() should parse to source: ${audioPath}`;

    // Notify listeners that a new UserRec was created (e.g. SavedScreen)
    Rec.emitter.emit('user', userSource);

    return userSource;
  },

  newAudioPath: async (ext: string, metadata: {
    coords: GeoCoords | null,
  }): Promise<string> => {

    // Create audioPath
    const source = UserSource.new({
      created: new Date(),
      uniq:    chance.hash({length: 8}), // Long enough to be unique across users
      ext,
    });
    const audioPath = UserRec.audioPath(source);

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

  // Wrap Rec.listAudio*
  listAudioSources:   async (): Promise<Array<UserSource>> => Rec.listAudioSources   (UserRec),
  listAudioPaths:     async (): Promise<Array<string>>     => Rec.listAudioPaths     (UserRec.audioDir),
  listAudioFilenames: async (): Promise<Array<string>>     => Rec.listAudioFilenames (UserRec.audioDir),

};

export const EditRec = {

  log: new Log('EditRec'),

  audioDir: Rec.editDir,

  // Give edit rec files a proper audio file ext, else e.g. ios AKAudioFile(forReading:) fails to infer the filetype
  //  - Use .wav to match kAudioFormatLinearPCM in NativeSpectro.editAudioPathToAudioPath
  audioExt: 'wav',

  // Problem: filename/path limits are very small, so we can't represent all Edit fields in an EditRec filename
  //  - Solution: omit .parent from filenames and store in AsyncStorage, keyed by Edit.uniq
  //  - e.g. ios: NAME_MAX=255 (max bytes in filename), PATH_MAX=1024 (max bytes in path)
  //    - https://github.com/theos/sdks/blob/2236ceb/iPhoneOS11.2.sdk/usr/include/sys/syslimits.h
  pathBasename: (source: EditSource): string => {
    const {edit} = source;
    return `edit-${Source.stringifyDate(edit.created)}-${edit.uniq}`; // e.g. 'edit-20190108011526401-d892e4be'
  },

  audioPath: (source: EditSource): string => {
    const pathBasename = Source.pathBasename(source);
    return `${EditRec.audioDir}/${pathBasename}.${EditRec.audioExt}`;
  },

  sourceFromAudioFilename: async (filename: string): Promise<EditSource | null> => {
    const ext = extname(filename).replace(/^\./, '');
    if (ext != EditRec.audioExt) throw `Expected ext[${EditRec.audioExt}], got ext[${ext}] in filename[${filename}]`;

    // Load Edit (AsyncStorage)
    //  - e.g. .parent can't safely store in the filename (too long)
    //  - TODO Perf: refactor callers so we can use AsyncStorage.multiGet
    const pathBasename = basename(filename, `.${ext}`);
    return mapNull(await Edit.load(pathBasename), edit => typed<EditSource>({
      kind: 'edit',
      edit,
    }));

  },

  spectroPath: (source: EditSource, opts: SpectroPathOpts): string => {
    return Rec.spectroCachePath(source, opts);
  },

  new: async (props: {parent: Rec, draftEdit: DraftEdit}): Promise<EditSource> => {
    const {parent, draftEdit} = props;

    // Make editSource <- edit <- (parent, draftEdit)
    const edit = {
      ...draftEdit,
      parent:  parent.source_id,
      created: new Date(),
      uniq:    chance.hash({length: 8}),
    };
    const editSource: EditSource = {
      kind: 'edit',
      edit,
    };

    // Store Edit (AsyncStorage)
    //  - e.g. .parent can't safely store in the filename (too long)
    await Edit.store(EditRec.pathBasename(editSource), edit);

    // Edit parent audio file -> edit audio file
    const parentAudioPath = Rec.audioPath(parent);
    const editAudioPath   = await EditRec.newAudioPath(edit, {parent});
    await NativeSpectro.editAudioPathToAudioPath({
      parentAudioPath,
      editAudioPath,
      draftEdit: typed<DraftEdit>(edit),
    });

    // Notify listeners that a new EditRec was created (e.g. SavedScreen)
    Rec.emitter.emit('edit', editSource);

    return editSource;
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
    const audioPath = EditRec.audioPath({kind: 'edit', edit});

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

  // Wrap Rec.listAudio*
  listAudioSources:   async (): Promise<Array<EditSource>> => Rec.listAudioSources   (EditRec),
  listAudioPaths:     async (): Promise<Array<string>>     => Rec.listAudioPaths     (EditRec.audioDir),
  listAudioFilenames: async (): Promise<Array<string>>     => Rec.listAudioFilenames (EditRec.audioDir),

};

//
// Edit
//

// A (commited) edit, which has a well defined mapping to an edit rec file
export interface Edit extends DraftEdit {
  parent:  SourceId;
  created: Date;
  uniq:    string; // Ensure a new filename for each EditRec (so we don't have to think about file collisions / purity)
}

// A draft edit, which isn't yet associated with any edit rec file
//  - Produced by RecordScreen (UX for creating edits from existing recs)
//  - Consumed by NativeSpectro (Spectro.swift:Spectro.editAudioPathToAudioPath)
export interface DraftEdit {
  clips?: Array<Clip>;
}

export interface Clip {
  time:  Interval;
  gain?: number; // TODO Unused
  // Room to grow: freq filter, airbrush, ...
}

// TODO Avoid unbounded O(parents) length for edit sourceId's [already solved for edit rec filenames]
//  - Add a layer of indirection for async (sync parse -> async load)
//  - Make it read Edit metadata to populate Edit.parent (from AsyncStorage like EditRec.sourceFromAudioFilename)
export const Edit = {

  // Can't simply json/unjson because it's unsafe for Interval (which needs to JsonSafeNumber)
  stringify: (edit: Edit): string => {
    return qsSane.stringify(typed<{[key in keyof Edit]: any}>({
      // HACK parent: Avoid chars that need uri encoding, since something keeps garbling them [router / location?]
      parent:  base64.encode(edit.parent).replace(/=+$/, ''),
      created: Source.stringifyDate(edit.created),
      uniq:    edit.uniq,
      clips:   mapUndefined(edit.clips, clips => clips.map(clip => Clip.jsonSafe(clip))),
    }));
  },
  parse: (x: string): Edit | null => {
    var q: any; // To include in logging (as any | null)
    try {
      q = qsSane.parse(x);
      return {
        // HACK parent: Avoid chars that need uri encoding, since something keeps garbling them [router / location?]
        parent:  Edit._required(q, 'parent',  (x: string) => /:/.test(x) ? x : base64.decode(x)),
        created: Edit._required(q, 'created', (x: string) => Source.parseDate(x)),
        uniq:    Edit._required(q, 'uniq',    (x: string) => x),
        clips:   Edit._optional(q, 'clips',   (xs: any[]) => xs.map(x => Clip.unjsonSafe(x))),
      };
    } catch (e) {
      log.warn('Edit.parse: Failed', rich({q, x, e}));
      return null;
    }
  },

  show: (edit: Edit, opts: SourceShowOpts): string => {
    const parts = [
      // Ignore uniq for human
      SourceId.show(edit.parent, opts),
      // Strip outer '[...]' per clip so we can wrap all clips together in one '[...]'
      sprintf('[%s]', (edit.clips || []).map(x => Clip.show(x).replace(/^\[|\]$/g, '')).join(',')),
    ];
    return (parts
      .filter(x => !_.isEmpty(x)) // Exclude null, undefined, '' (and [], {})
      .join(' ')
    );
  },

  // Parse results of qsSane.parse
  //  - TODO Add runtime type checks for X [how?] so we fail when q[k] isn't an X
  _optional: <X, Y>(q: any, k: keyof Edit, f: (x: X) => Y): Y | undefined => mapUndefined(q[k], x => f(x)),
  _required: <X, Y>(q: any, k: keyof Edit, f: (x: X) => Y): Y             => f(Edit._requireKey(q, k)),
  _requireKey: (q: any, k: keyof Edit): any => ifUndefined(q[k], () => throw_(`Edit: Field '${k}' required: ${json(q)}`)),

  store: async (pathBasename: string, edit: Edit): Promise<void> => {
    const k = `Edit.${pathBasename}`;
    await AsyncStorage.setItem(k, Edit.stringify(edit));
  },
  load: async (pathBasename: string): Promise<Edit | null> => {
    const k = `Edit.${pathBasename}`;
    const s = await AsyncStorage.getItem(k);
    if (s === null) {
      log.warn('Edit.load: Key not found', rich({k}));
      return null;
    } else {
      return Edit.parse(s);
    }
  },

};

export const DraftEdit = {

  // For NativeSpectro.editAudioPathToAudioPath [see HACK there]
  jsonSafe: (draftEdit: DraftEdit): any => {
    return {
      clips: mapUndefined(draftEdit.clips, xs => xs.map(x => Clip.jsonSafe(x))),
    };
  },
  // TODO When needed
  // unjsonSafe: (x: any): DraftEdit => {
  // },

  hasEdits: (edit: DraftEdit): boolean => {
    return !_(edit).values().every(_.isEmpty);
  },

};

export const Clip = {

  stringify: (clip: Clip): string => {
    return json(Clip.jsonSafe(clip));
  },
  parse: (x: string): Clip => {
    return Clip.unjsonSafe(unjson(x));
  },

  jsonSafe: (clip: Clip): any => {
    return {
      time: clip.time.jsonSafe(),
      gain: clip.gain,
    };
  },
  unjsonSafe: (x: any): Clip => {
    return {
      time: Interval.unjsonSafe(x.time),
      gain: x.gain,
    };
  },

  show: (clip: Clip): string => {
    return [
      (clip.time.intersect(Interval.nonNegative) || clip.time).show(), // Replace [–1.23] with [0.00–1.23], but keep [1.23–] as is
      showSuffix('×', clip.gain, x => sprintf('%.2f', x)),
    ].join('');
  },

}

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
