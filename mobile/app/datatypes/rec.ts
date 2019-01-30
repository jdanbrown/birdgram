import { EventEmitter } from 'fbemitter';
import _ from 'lodash';
import { AsyncStorage } from 'react-native';
import DeviceInfo from 'react-native-device-info';
import RNFB from 'rn-fetch-blob';
const {fs} = RNFB;

import { GeoCoords } from 'app/components/Geo';
import { config } from 'app/config';
import {
  DraftEdit, Edit_v2, Edit, EditSource, matchSourceId, SearchRecs, Source, SourceId, SourceParseOpts, Species,
  UserSource, XCSource,
} from 'app/datatypes';
import { debug_print, Log, rich } from 'app/log';
import { NativeSpectro } from 'app/native/Spectro';
import { NativeTagLib } from 'app/native/TagLib';
import { Places } from 'app/places';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifError, ifNil, ifNull, ifUndefined, json,
  jsonSafeError, JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchError, matchNull,
  matchUndefined, NoKind, Omit, parseDate, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane,
  requireSafePath, safeParseInt, safeParseIntOrNull, safePath, showDate, showSuffix, splitFirst, stringifyDate,
  stripExt, throw_, tryElse, typed, unjson,
} from 'app/utils';

// TODO How to prevent callers from constructing a Rec i/o one of the subtypes? [Maybe classes with constructors?]
export type Rec = XCRec | UserRec | EditRec;

export interface XCRec extends _RecImpl {
  kind:  'xc';
  xc_id: number;
}

export interface UserRec extends _RecImpl {
  kind:     'user';
  f_preds:  Array<number>;
  // TODO(cache_user_metadata): Move slow UserRec.metadata -> fast UserSource.metadata
  //  - UserRec.metadata: slow load from audio file tags (primary storage)
  //  - UserSource.metadata: fast (-er, still async) load from AsyncStorage cache (derived storage, disposable) of audio file tags
  metadata: UserMetadata;
}

// TODO(unify_edit_user_recs): Kill EditRec/EditSource everywhere! Will require updating lots of matchRec/matchSource callers
export interface EditRec extends _RecImpl {
  kind:     'edit';
  edit:     Edit;
  // parent:   Rec;  // Require consumers to load the parent Rec, else we risk O(parents) work to load an edit rec
  f_preds:  Array<number>;
  // TODO(unify_edit_user_recs): Unify edit + user recs so we don't have to add another kind of metadata.species labeling for edit recs
}

export interface UserMetadata {
  // Immutable facts (captured at record time)
  //  - TODO(user_metadata): Move {created,uniq} filename->metadata so that user can safely rename files (e.g. share, export/import)
  //  - NOTE(user_metadata): File metadata already contains {created,uniq} (guaranteed by UserRec.writeMetadata from the start)
  // created: Date;
  // uniq:    string;
  edit:    null | Edit_v2,   // null if a new recording, non-null if an edit of another recording (edit.parent)
  creator: null | Creator,   // null if unknown creator
  coords:  null | GeoCoords; // null if unknown gps
  // Mutable user data
  species: UserSpecies;
}

export interface Creator {
  // username:        string, // TODO(share_recs): Let users enter a short username to display i/o their deviceName
  deviceName:      string,
  appBundleId:     string,
  appVersion:      string,
  appVersionBuild: string,
}

export type UserSpecies = // TODO Make proper ADT with matchUserSpecies
  | {kind: 'unknown'}
  | {kind: 'maybe', species: Array<Species>} // Not yet used
  | {kind: 'known', species: Species};

// XXX(unify_edit_user_recs): Kill this after unifying edit + user recs (and add .edit to UserMetadata)
export interface EditMetadata {
  edit: Edit;
}

export const Creator = {

  get: (): Creator => {
    return {
      deviceName:      DeviceInfo.getDeviceName(),
      appBundleId:     config.env.APP_BUNDLE_ID,
      appVersion:      config.env.APP_VERSION,
      appVersionBuild: config.env.APP_VERSION_BUILD,
    };
  },

  jsonSafe: (creator: Creator): any => {
    return typed<{[key in keyof Creator]: any}>({
      deviceName:      creator.deviceName,
      appBundleId:     creator.appBundleId,
      appVersion:      creator.appVersion,
      appVersionBuild: creator.appVersionBuild,
    });
  },
  unjsonSafe: (x: any): Creator => {
    return {
      deviceName:      x.deviceName,
      appBundleId:     x.appBundleId,
      appVersion:      x.appVersion,
      appVersionBuild: x.appVersionBuild,
    };
  },

};

export const UserMetadata = {

  new: (props: {
    edit:     UserMetadata['edit'],
    creator?: UserMetadata['creator'],
    coords:   UserMetadata['coords'],
    species?: UserMetadata['species'],
  }): UserMetadata => {
    return {
      // Immutable facts
      edit:    props.edit,
      creator: ifUndefined(props.creator, () => Creator.get()),
      coords:  props.coords,
      // Mutable user data (initial values)
      species: ifUndefined(props.species, () => typed<UserSpecies>({kind: 'unknown'})),
    };
  },

  jsonSafe: (metadata: UserMetadata): any => {
    return typed<{[key in keyof UserMetadata]: any}>({
      edit:    mapNull(metadata.edit,    Edit_v2.jsonSafe),
      creator: mapNull(metadata.creator, Creator.jsonSafe),
      coords:  metadata.coords,
      species: UserSpecies.jsonSafe(metadata.species),
    });
  },
  unjsonSafe: (x: any): UserMetadata => {
    return {
      edit:    mapNull(x.edit,    Edit_v2.unjsonSafe),
      creator: mapNull(x.creator, Creator.unjsonSafe),
      coords:  x.coords,
      species: UserSpecies.unjsonSafe(x.species),
    };
  },

};

export const UserSpecies = {

  // TODO How to do typesafe downcast? This unjsonSafe leaks arbitrary errors
  jsonSafe:   (x: UserSpecies): any         => x,
  unjsonSafe: (x: any):         UserSpecies => x as UserSpecies, // HACK Type

  show: (userSpecies: UserSpecies): string => {
    switch (userSpecies.kind) {
      case 'unknown': return '?';
      case 'maybe':   return `${userSpecies.species.join('/')}?`;
      case 'known':   return userSpecies.species;
    }
  },

};

export const EditMetadata = {
  jsonSafe: (metadata: EditMetadata): any => {
    return typed<{[key in keyof EditMetadata]: any}>({
      edit: Edit.jsonSafe(metadata.edit),
    });
  },
  unjsonSafe: (x: any): EditMetadata => {
    return {
      edit: Edit.unjsonSafe(x.edit),
    };
  },
};

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
  opts: SourceParseOpts, // XXX(cache_user_metadata)
  xc:   (rec: XCRec,   source: XCSource)   => X,
  user: (rec: UserRec, source: UserSource) => X,
  edit: (rec: EditRec, source: EditSource) => X,
}): X {
  // HACK Switch on rec.source_id until we refactor all Rec constructors to include .kind
  return matchSourceId(rec.source_id, {
    opts: cases.opts,
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

  log: new Log('Rec'),

  // Primary storage (DocumentDir)
  userRecDir:      `${fs.dirs.DocumentDir}/Recordings`, // User recordings
  // Files that can be regenerated by need (CacheDir)
  spectroCacheDir: `${fs.dirs.CacheDir}/spectros-v0`, // Spectros will recreate if missing

  // Back compat: migrate old-style {user-recs-v0,edits-v0}/ dirs to new-style Recordings/ dir
  //  - XXX(unify_edit_user_recs): After all (three) active users have migrated
  //  - (Used by App.componentDidMount)
  old_userRecDir:  `${fs.dirs.DocumentDir}/user-recs-v0`,
  old_editDir:     `${fs.dirs.DocumentDir}/edits-v0`,
  trash_editDir:   `${fs.dirs.DocumentDir}/_trash_edits-v0`,

  // Events
  emitter: new EventEmitter(),

  audioPath: (rec: Rec): string => matchRec(rec, {
    opts: {userMetadata: null}, // XXX(cache_user_metadata): Not used for audioPath
    xc:   (rec, source) => XCRec.audioPath(rec),
    user: (rec, source) => UserRec.audioPath(source),
    edit: (rec, source) => EditRec.audioPath(source),
  }),

  spectroPath: (
    rec:  Rec,
    opts: SpectroPathOpts, // Ignored for xc rec [TODO Clean up]
  ): string => matchRec(rec, {
    opts: {userMetadata: null}, // XXX(cache_user_metadata): Not used for spectroPath
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

  // Load metadata from audio file comment tag (as json)
  //  - Version so we can maintain back compat with data from previous code versions
  readMetadata: async (audioPath: string): Promise<null | {
    version:  number,
    metadata: {},
  }> => {
    const commentTag = await NativeTagLib.readComment(audioPath);
    return matchError(() => unjson(commentTag || 'null'), {
      error: e => {
        Rec.log.warn('readMetadata: Ignoring malformed json', pretty({audioPath, commentTag}));
        return null;
      },
      x: versionedMetadata => {
        Rec.log.debug('readMetadata', rich({audioPath, versionedMetadata}));
        return versionedMetadata;
      },
    });
  },

  // Write metadata to audio file comment tag (as json)
  //  - Version so we can maintain back compat with data from previous code versions
  writeMetadata: async (audioPath: string, versionedMetadata: {
    version:  number,
    metadata: {},
  }): Promise<void> => {
    Rec.log.debug('writeMetadata', rich({audioPath, versionedMetadata}));
    await NativeTagLib.writeComment(audioPath, json(versionedMetadata));
  },

  f_preds: (rec: Rec): Rec_f_preds => {
    return matchRec(rec, {
      opts: {userMetadata: null}, // XXX(cache_user_metadata): Not used for f_preds
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
      opts: {userMetadata: null}, // XXX(cache_user_metadata): Not used for recUrl
      xc:   rec => XCRec.recUrl(rec),
      user: rec => null,
      edit: rec => null, // TODO(unify_edit_user_recs): Revisit where to get parent url after unifying user + edit recs
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

  listAudioSources: async <FileSource>(_FileRec: {
    audioDir:            string,
    sourceFromAudioPath: (path: string) => Promise<FileSource | null>,
  }): Promise<Array<FileSource>> => {
    return _.flatten(await Promise.all(
      (await Rec.listAudioPaths(_FileRec.audioDir)).map(async audioPath => {
        const source = await _FileRec.sourceFromAudioPath(audioPath);
        if (!source) {
          Rec.log.warn("listAudioSources: Dropping: Failed to parse sourceId", pretty({audioPath}));
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
    const excludes = [/\.metadata\.json$/]; // Back compat
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

  spectroPath: (rec: XCRec): string => {
    // From assets, not spectroCacheDir
    return SearchRecs.assetPath('spectro', rec.species, rec.xc_id, 'png');
  },

  // TODO (When needed)
  // sourceFromAudioPath: async (audioPath: string): Promise<XCSource | null> => {
  //   ...
  // },

  recUrl: (rec: XCRec): string => {
    return `https://www.xeno-canto.org/${rec.xc_id}`;
  },

};

export const UserRec = {

  log: new Log('UserRec'),

  audioDir: Rec.userRecDir,

  // Give user rec files a proper audio file ext, else e.g. ios AKAudioFile(forReading:) fails to infer the filetype
  //  - Use .wav to match kAudioFormatLinearPCM in NativeSpectro.create
  audioExt: 'wav',

  pathBasename: (source: UserSource): string => {
    return stripExt(source.filename); // e.g. 'user-20190109205640977-336b2bb7'
  },

  audioPath: (source: UserSource): string => {
    // Use preserved filename so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
    return `${UserRec.audioDir}/${source.filename}`;
  },

  spectroPath: (source: UserSource, opts: SpectroPathOpts): string => {
    return Rec.spectroCachePath(source, opts);
  },

  // Load user metadata
  //  - TODO(cache_user_metadata): Add cache read/write (currently just a passthru to readMetadata)
  //    1. Try reading from cache (AsyncStorage)
  //    2. Else readMetadata (from audio file tags) and write to cache (AsyncStorage)
  loadMetadata: async (audioPath: string): Promise<UserMetadata> => {
    return UserRec.readMetadata(audioPath);
  },

  // Read user metadata from audio file tags
  readMetadata: async (audioPath: string): Promise<UserMetadata> => {
    const versionedMetadata = await Rec.readMetadata(audioPath);
    const defaults: UserMetadata = {
      edit:    null,              // Not an edit of another rec
      creator: null,              // Unknown creator
      coords:  null,              // Unknown gps
      species: {kind: 'unknown'}, // Unknown species
    };
    if (!versionedMetadata) {
      // Noisy (when lots of pre-metadata user recs, e.g. mine)
      // UserRec.log.warn('readMetadata: No metadata (in tags), returning defaults', pretty({
      //   audioPath, versionedMetadata, defaults,
      // }));
      return defaults;
    } else {
      const {version, metadata} = versionedMetadata;
      if (version === 1) {
        return ifError(() => UserMetadata.unjsonSafe(metadata), e => {
          UserRec.log.warn('readMetadata: Invalid metadata (in tags), returning defaults', pretty({
            e: jsonSafeError(e), audioPath, versionedMetadata, defaults,
          }));
          return defaults;
        });
      } else {
        throw `UserRec.readMetadata: Unknown version[${version}] for metadata[${metadata}]`;
      }
    }
  },

  // Write user metadata to audio file tags
  //  - TODO(cache_user_metadata): Invalidate/update cache (AsyncStorage)
  writeMetadata: async (audioPath: string, metadata: UserMetadata): Promise<void> => {
    // HACK(user_metadata): Parse {created,uniq} from filename so we can store it with the file metadata
    //  - This lets us avoid having to deal with back compat later, since all versions of stored file metadata include {created,uniq}
    //  - Kill this after we move {created,uniq} filename->metadata
    await matchNull(await UserRec._sourceFromAudioPath(audioPath, {
      userMetadata: null, // ...
    }), {
      null: async () => {
        UserRec.log.error('writeMetadata: Failed to _sourceFromAudioPath, not writing tags', pretty({audioPath, metadata}));
      },
      x: async ({created, uniq}) => {
        await Rec.writeMetadata(audioPath, {
          version: 1,
          metadata: {
            created: stringifyDate(created),    // jsonSafe: Date -> string
            uniq,                               // jsonSafe: string
            ...UserMetadata.jsonSafe(metadata), // Do last so that metadata.{created,uniq} are preserved if they exist
          },
        });
      },
    });
  },

  // (Callers: RecordScreen.startRecording)
  //  - Generate audioPath but don't create the file (will be created by RecordScreen.startRecording -> NativeSpectro)
  newAudioPath: async (ext: string): Promise<string> => {

    // Create audioPath
    const source = UserSource.new({
      created: new Date(),
      uniq:    chance.hash({length: 8}), // Long enough to be unique across users
      ext,
      metadata: null, // TODO(cache_user_metadata): Populate after moving slow UserRec.metadata -> fast UserSource.metadata
    });
    const audioPath = UserRec.audioPath(source);

    // Ensure parent dir for caller
    await ensureParentDir(audioPath);

    // Done
    UserRec.log.info('newAudioPath', rich({audioPath}));
    return audioPath;

  },

  // (Callers: RecordScreen.stopRecording)
  //  - Assumes file at audioPath exists (created by RecordScreen.startRecording -> NativeSpectro)
  new: async (audioPath: string, metadata: UserMetadata): Promise<UserSource> => {

    // Write user metadata
    //  - Assumes file at audioPath exists
    await UserRec.writeMetadata(audioPath, metadata);

    // Make UserSource
    const userSource = await UserRec.sourceFromAudioPath(audioPath);
    if (!userSource) throw `stopRecording: audioPath from Nativespectro.stop() should parse to source: ${audioPath}`;

    // Log (before notify)
    UserRec.log.info('new', rich({audioPath, metadata, userSource}));

    // Notify listeners that a new UserRec was created (e.g. SavedScreen)
    Rec.emitter.emit('user', userSource);

    return userSource;
  },

  // (Callers: RecordScreen "Done editing" button)
  //  - Creates audioPath (via newAudioPath) and file (via editAudioPathToAudioPath)
  newFromEdit: async (props: {parent: Rec, draftEdit: DraftEdit}): Promise<UserSource> => {
    const parent  = props.parent;
    var draftEdit = props.draftEdit;

    // Attach to grandparent (flat) i/o parent (recursive), else we'd have to deal with O(n) parent chains
    //  - Load parent's UserMetadata, if a user rec
    //  - If an edit rec, attach to parent's parent with a merged edit [XXX(unify_edit_user_recs)]
    //  - Else, attach to parent with our edit
    const parentEdit: null | Edit_v2 = await matchRec(props.parent, {
      opts: {userMetadata: null}, // HACK(cache_user_metadata): Don't need source.metadata b/c already have parentRec.metadata
      xc:   async (parentRec, parentSource) => null,
      user: async (parentRec, parentSource) => parentRec.metadata.edit, // null | Edit_v2
      edit: async (parentRec, parentSource) => Edit.to_v2(parentRec.edit), // XXX(unify_edit_user_recs)
    });
    const edit = matchNull(parentEdit, {
      null: () => ({
        // Make a new edit rec rooted at parent (which isn't already an edit rec, by construction)
        parent: parent.source_id,
        edits:  [draftEdit],
      }),
      x: parentEdit => ({
        // Preserve parent's parent, adding draftEdit to its edits
        parent: parentEdit.parent,
        edits:  [...parentEdit.edits, draftEdit],
      }),
    });

    // Make userSource <- metadata <- edit
    const metadata = UserMetadata.new({
      edit,
      coords: null, // TODO Copy coords from XCRec (.lat,.lng) / UserRec (.metadata.coords)
      species: {kind: 'unknown'}, // 'unknown' species even if parent is known, e.g. clipping down to an unknown bg species
    });
    const userSource = UserSource.new({
      created: new Date(),
      uniq:    chance.hash({length: 8}),
      ext:     UserRec.audioExt,
      metadata,
    });
    UserRec.log.info('newFromEdit', rich({userSource}));

    // Edit parent audio file -> edit audio file
    const audioPath = UserRec.audioPath(userSource);
    await NativeSpectro.editAudioPathToAudioPath({
      parentAudioPath: Rec.audioPath(parent),
      editAudioPath:   await ensureParentDir(audioPath),
      draftEdit,
    });

    // Write user metadata
    //  - Requires audioPath to exist (created above by editAudioPathToAudioPath)
    await UserRec.writeMetadata(audioPath, metadata);

    // Notify listeners that a new UserRec was created (e.g. SavedScreen)
    Rec.emitter.emit('user', userSource);

    return userSource;
  },

  // (Callers: Rec.listAudioSources, UserRec.new)
  sourceFromAudioPath: async (audioPath: string): Promise<UserSource | null> => {
    // Load user metadata (from AsyncStorage cache, else from audio file tags)
    const metadata = await UserRec.loadMetadata(audioPath);
    // Construct UserSource from (filename, UserMetadata)
    return UserRec._sourceFromAudioPath(audioPath, {userMetadata: metadata});
  },

  // Split out for writeMetadata
  _sourceFromAudioPath: async (audioPath: string, opts: SourceParseOpts): Promise<UserSource | null> => {
    const filename = basename(audioPath);
    return mapNull(
      Source.parse(`user:${filename}`, opts),
      x => x as UserSource, // HACK Type
    );
  },

  // Wrap Rec.listAudio*
  listAudioSources:   async (): Promise<Array<UserSource>> => Rec.listAudioSources   (UserRec),
  listAudioPaths:     async (): Promise<Array<string>>     => Rec.listAudioPaths     (UserRec.audioDir),
  listAudioFilenames: async (): Promise<Array<string>>     => Rec.listAudioFilenames (UserRec.audioDir),

};

// TODO(unify_edit_user_recs): Kill EditRec/EditSource everywhere! Will require updating lots of matchRec/matchSource callers
export const EditRec = {

  log: new Log('EditRec'),

  audioDir: Rec.old_editDir,

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

  spectroPath: (source: EditSource, opts: SpectroPathOpts): string => {
    return Rec.spectroCachePath(source, opts);
  },

  // Load edit metadata
  //  - TODO(cache_user_metadata): Add cache read/write (currently just a passthru to readMetadata)
  //    1. Try reading from cache (AsyncStorage)
  //    2. Else readMetadata (from audio file tags) and write to cache (AsyncStorage)
  loadMetadata: async (audioPath: string): Promise<EditMetadata | null> => {
    return EditRec.readMetadata(audioPath);
  },

  // Read edit metadata, falling back to (and upgrading) old versions for back compat
  //  - v1: Read from audio file tags
  //  - v0: Read from AsyncStorage
  readMetadata: async (audioPath: string): Promise<EditMetadata | null> => {

    // HACK All edit files named 'edit-' i/o 'editv1-' were accidentally .caf i/o .wav. Don't try to Rec.readMetadata on
    // them, else you'll get errors (b/c taglib doesn't like .caf).
    const isCaf = 'caf' === await NativeTagLib.audioFiletype(audioPath)
    if (isCaf) {
      EditRec.log.info('readMetadata: XXX Old .caf i/o .wav edit rec: Ignoring v1 tags, skipping v0->v1 upgrade', pretty({audioPath}));
    }

    var metadata: EditMetadata | null = null;
    if (!isCaf && metadata === null) {
      metadata = await EditRec._readMetadata_v1(audioPath);
    }
    if (metadata === null) {
      metadata = await EditRec._readMetadata_v0(audioPath);
      if (!isCaf && metadata !== null) {
        EditRec.log.info('readMetadata: Upgrading old stored metadata v0->v1 (AsyncStorage->tags)', {audioPath, metadata});
        EditRec.writeMetadata(audioPath, metadata);
      }
    }
    if (metadata === null) {
      // Shouldn't normally happen, but be robust
      EditRec.log.warn('readMetadata: No metadata found for any of versions[v1,v0], returning null', {audioPath});
    }

    return metadata;
  },

  // Read edit metadata from audio file tags
  _readMetadata_v1: async (audioPath: string): Promise<EditMetadata | null> => {
    const versionedMetadata = await Rec.readMetadata(audioPath);
    if (!versionedMetadata) {
      // Noisy (when lots of pre-metadata edit recs, e.g. mine)
      // EditRec.log.warn('_readMetadata_v1: No metadata (in tags), returning null', pretty({audioPath, versionedMetadata}));
      return null;
    } else {
      const {version, metadata} = versionedMetadata;
      if (version === 1) {
        return ifError(() => EditMetadata.unjsonSafe(metadata), e => {
          EditRec.log.warn('_readMetadata_v1: Invalid metadata (in tags), returning null', pretty({
            e: jsonSafeError(e), audioPath, versionedMetadata,
          }));
          return null;
        });
      } else {
        throw `EditRec.readMetadata: Unknown version[${version}] for metadata[${metadata}]`;
      }
    }
  },

  // Read edit metadata from AsyncStorage
  _readMetadata_v0: async (audioPath: string): Promise<EditMetadata | null> => {
    const filename = basename(audioPath);
    const ext = extname(filename).replace(/^\./, '');
    if (ext != EditRec.audioExt) throw `Expected ext[${EditRec.audioExt}], got ext[${ext}] in filename[${filename}]`;
    const pathBasename = basename(filename, `.${ext}`);
    return mapNull(
      await Edit.load(pathBasename),
      edit => ({edit}), // EditMetadata = {edit: Edit}
    );
  },

  // Write edit metadata to audio file tags
  //  - TODO(cache_user_metadata): Invalidate/update cache (AsyncStorage)
  writeMetadata: async (audioPath: string, metadata: EditMetadata): Promise<void> => {
    await Rec.writeMetadata(audioPath, {
      version: 1,
      metadata: EditMetadata.jsonSafe(metadata),
    });
  },

  // (Callers: EditRec.new)
  //  - Generate audioPath but don't create the file (will be created by EditRec.new)
  newAudioPath: async (edit: Edit): Promise<string> => {

    // Create audioPath
    const audioPath = EditRec.audioPath({kind: 'edit', edit});

    // Ensure parent dir for caller
    await ensureParentDir(audioPath);

    // Done
    EditRec.log.info('newAudioPath', rich({audioPath, edit}));
    return audioPath;

  },

  // (Callers: RecordScreen "Done editing" button)
  //  - Creates audioPath (via newAudioPath) and file (via editAudioPathToAudioPath)
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

    // Edit parent audio file -> edit audio file
    const parentAudioPath = Rec.audioPath(parent);
    const editAudioPath   = await EditRec.newAudioPath(edit);
    await NativeSpectro.editAudioPathToAudioPath({
      parentAudioPath,
      editAudioPath,
      draftEdit: typed<DraftEdit>(edit),
    });

    // Write edit metadata
    //  - Requires editAudioPath to exist (created above by editAudioPathToAudioPath)
    const metadata = typed<EditMetadata>({
      edit,
    });
    await EditRec.writeMetadata(editAudioPath, metadata);

    // Notify listeners that a new EditRec was created (e.g. SavedScreen)
    Rec.emitter.emit('edit', editSource);

    return editSource;
  },

  // (Callers: Rec.listAudioSources)
  sourceFromAudioPath: async (audioPath: string): Promise<EditSource | null> => {
    // Load edit metadata (from AsyncStorage cache, else from audio file tags)
    const metadata: EditMetadata | null = await EditRec.loadMetadata(audioPath);
    // Construct EditSource from EditMetadata
    //  - XXX(unify_edit_user_recs): Simplify
    return mapNull(metadata, ({edit}) => typed<EditSource>({
      kind: 'edit',
      edit,
    }));
  },

  // Wrap Rec.listAudio*
  listAudioSources:   async (): Promise<Array<EditSource>> => Rec.listAudioSources   (EditRec),
  listAudioPaths:     async (): Promise<Array<string>>     => Rec.listAudioPaths     (EditRec.audioDir),
  listAudioFilenames: async (): Promise<Array<string>>     => Rec.listAudioFilenames (EditRec.audioDir),

};
