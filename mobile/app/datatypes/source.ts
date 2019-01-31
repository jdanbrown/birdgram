import _ from 'lodash';
import moment from 'moment';
import { AsyncStorage } from 'react-native';
import DeviceInfo from 'react-native-device-info';
import { sprintf } from 'sprintf-js';

import { GeoCoords } from 'app/components/Geo';
import { config } from 'app/config';
import { DraftEdit, Edit, Species, UserRec, XCRec } from 'app/datatypes';
import { debug_print, log, Log, rich } from 'app/log';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchError, matchErrorAsync,
  matchNull, matchUndefined, Omit, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane, requireSafePath,
  safeParseInt, safeParseIntOrNull, safePath, showDate, showSuffix, splitFirst, stripExt, throw_, tryElse, typed,
  unjson,
} from 'app/utils';
import { XC } from 'app/xc';

export type SourceId = string;
export type Source = XCSource | UserSource;
export interface XCSource   { kind: 'xc';   xc_id: number; }
export interface UserSource { kind: 'user';
  // TODO(cache_user_metadata): Move {created,uniq} filename->metadata so that user can safely rename files (e.g. share, export/import)
  created:  Date;
  uniq:     string;
  ext:      string;
  filename: string; // Preserve so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
  metadata: UserMetadata;
}

export interface UserMetadata {
  // Immutable facts (captured at record time)
  //  - TODO(cache_user_metadata): Move {created,uniq} filename->metadata so that user can safely rename files (e.g. share, export/import)
  //  - NOTE(cache_user_metadata): File metadata already contains {created,uniq} (guaranteed by UserRec.writeMetadata from the start)
  // created: Date;
  // uniq:    string;
  edit:    null | Edit,      // null if a new recording, non-null if an edit of another recording (edit.parent)
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

  stringify: (metadata: UserMetadata): string => json(UserMetadata.jsonSafe(metadata)),
  parse:     (x: string): UserMetadata        => UserMetadata.unjsonSafe(unjson(x)),

  jsonSafe: (metadata: UserMetadata): any => {
    return typed<{[key in keyof UserMetadata]: any}>({
      edit:    mapNull(metadata.edit,    Edit.jsonSafe),
      creator: mapNull(metadata.creator, Creator.jsonSafe),
      coords:  metadata.coords,
      species: UserSpecies.jsonSafe(metadata.species),
    });
  },
  unjsonSafe: (x: any): UserMetadata => {
    return {
      edit:    mapNull(x.edit,    Edit.unjsonSafe),
      creator: mapNull(x.creator, Creator.unjsonSafe),
      coords:  x.coords,
      species: UserSpecies.unjsonSafe(x.species),
    };
  },

  // TODO(cache_user_metadata): Groundwork for caching readMetadata calls (which we might not ever need to do)
  // store: async (name: string, metadata: UserMetadata): Promise<void> => {
  //   const k = `${UserMetadata._storePrefix}.${name}`;
  //   await AsyncStorage.setItem(k, UserMetadata.stringify(metadata));
  // },
  // load: async (name: string): Promise<UserMetadata | null> => {
  //   const k = `${UserMetadata._storePrefix}.${name}`;
  //   return mapNull(
  //     await AsyncStorage.getItem(k), // null if key not found
  //     s => UserMetadata.parse(s),
  //   );
  // },
  // _storePrefix: 'UserMetadata.cache',

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

export interface SourceShowOpts {
  species: XC | null; // Show species if xc sourceId (using XC dep)
  long?: boolean;
}

// TODO(cache_user_metadata): Can we kill this? UserSource.metadata is no longer nullable
export interface HasUserMetadata {
  userMetadata: UserMetadata;
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

  // TODO Can we get rid of this? Last remaining caller is Edit.show...
  show: (sourceId: SourceId, opts: SourceShowOpts & HasUserMetadata): string => {
    if (SourceId.isOldStyleEdit(sourceId)) {
      return `[Deprecated old-style edit rec]`;
    } else {
      return matchNull(Source.parse(sourceId, _.pick(opts, 'userMetadata')), {
        x:    source => Source.show(source, opts),
        null: ()     => `[Malformed: ${sourceId}]`,
      });
    }
  },

  isOldStyleEdit: (sourceId: SourceId): boolean => {
    return sourceId.startsWith('edit:');
  },

};

export const Source = {

  // NOTE Preserve {kind:'xc',xc_id} -> `xc:${xc_id}` to stay consistent with db payload (see util.py:rec_to_source_id)
  stringify: (source: Source): SourceId => {
    return matchSource(source, {
      xc:   ({xc_id}) => `xc:${xc_id}`,
      user: source    => `user:${UserSource.stringify(source)}`,
    });
  },

  // TODO De-dupe parse/load
  // Sync variant of load that takes userMetadata from caller
  parse: (sourceId: SourceId, opts: HasUserMetadata): Source | null => {
    const {kind, ssp} = SourceId.split(sourceId);
    return match<string, Source | null>(kind,
      ['xc',          () => XCSource.parse(ssp)],
      ['user',        () => UserSource.parse(ssp, opts)],
      ['edit',        () => null], // Back compat with old-style edit recs (EditSource, EditRec)
      [match.default, () => { throw `Source.parse: Unknown sourceId type: ${sourceId}`; }],
    );
  },

  // TODO De-dupe parse/load
  // Async variant of parse that loads userMetadata
  load: async (sourceId: SourceId): Promise<Source | null> => {
    const {kind, ssp} = SourceId.split(sourceId);
    return await match<string, Promise<Source | null>>(kind,
      ['xc',          async () => XCSource.parse(ssp)],
      ['user',        async () => await UserSource.load(ssp)],
      ['edit',        async () => null], // Back compat with old-style edit recs (EditSource, EditRec)
      [match.default, async () => { throw `Source.load: Unknown sourceId type: ${sourceId}`; }],
    );
  },

  parseOrFail: (sourceId: SourceId, opts: HasUserMetadata): Source => {
    return matchNull(Source.parse(sourceId, opts), {
      null: ()     => { throw `Source.parseOrFail: Failed to parse sourceId[${sourceId}]`; },
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
      user: ({created, uniq, ext, metadata}) => {
        // Ignore uniq (in name=`${date}-${time}-${uniq}`), since seconds resolution should be unique enough for human
        //  - TODO Rethink after rec sharing (e.g. add usernames to avoid collisions)
        const parts = (
          metadata && metadata.edit ? [
            !metadata  ? '' : `[${UserSpecies.show(metadata.species)}]`,
            !opts.long ? '' : 'Edit:',
            Edit.show(metadata.edit, opts),
          ] : [
            !metadata  ? '' : `[${UserSpecies.show(metadata.species)}]`,
            !opts.long ? '' : 'Recording:',
            showDate(created),
          ]
        );
        return (parts
          .filter(x => !_.isEmpty(x)) // Exclude null, undefined, '' (and [], {})
          .join(' ')
        );
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
      }),
    );
  },

};

export const XCSource = {

  parse: (ssp: string): XCSource | null => {
    return mapNull(safeParseIntOrNull(ssp), xc_id => typed<XCSource>({
      kind: 'xc',
      xc_id,
    }));
  },

};

export const UserSource = {

  stringify: (source: UserSource): string => {
    // Return preserved filename so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
    return source.filename;
  },

  // Sync variant of load that takes userMetadata from caller
  parse: (ssp: string, opts: HasUserMetadata): UserSource | null => {
    return mapNull(UserSource._parse(ssp), x => ({...x,
      metadata: opts.userMetadata,
    }));
  },

  // (Callers: parse, UserRec.writeMetadata)
  _parse: (ssp: string): Omit<UserSource, 'metadata'> | null => {
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
        kind:     'user',
        created:  Source.parseDate(created),
        uniq,
        ext,
        filename: ssp, // Preserve so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
      };
    } catch (e) {
      log.warn('UserSource._parse: Failed', rich({ssp, e}));
      return null;
    }
  },

  // Async variant of parse that loads userMetadata
  load: async (ssp: string): Promise<UserSource | null> => {
    return await mapNull(
      UserSource.parse(ssp, {
        userMetadata: null as unknown as UserMetadata, // HACK Safe b/c unused in UserRec.audioPath
      }),
      async sourceNoMetadata => {
        const audioPath = UserRec.audioPath(sourceNoMetadata);
        return await matchErrorAsync(async () => await UserRec.loadMetadata(audioPath), {
          x: async metadata => ({
            ...sourceNoMetadata,
            metadata,
          }),
          error: async e => {
            // Disabled logging: very noisy for source not found (e.g. user deleted a user rec, or xc dataset changed)
            //  - Rely on the caller to warn/error as appropriate
            // log.debug('UserSource.load: Failed to loadMetadata, returning null', rich({ssp, audioPath, e})); // XXX Debug
            return null;
          },
        });
      },
    );
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
}): X {
  switch(source.kind) {
    case 'xc':   return cases.xc   (source);
    case 'user': return cases.user (source);
  }
}
