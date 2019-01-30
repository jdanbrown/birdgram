import _ from 'lodash';
import moment from 'moment';
import { sprintf } from 'sprintf-js';

import { DraftEdit, Edit, UserMetadata, UserRec, UserSpecies, XCRec } from 'app/datatypes';
import { debug_print, log, Log, rich } from 'app/log';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchNull, matchUndefined, NoKind,
  Omit, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane, requireSafePath, safeParseInt, safeParseIntOrNull,
  safePath, showDate, showSuffix, splitFirst, stripExt, throw_, tryElse, typed, unjson,
} from 'app/utils';
import { XC } from 'app/xc';

export type SourceId = string;
export type Source = XCSource | UserSource;
export interface XCSource   { kind: 'xc';   xc_id: number; }
export interface UserSource { kind: 'user';
  // TODO(user_metadata): Move {created,uniq} filename->metadata so that user can safely rename files (e.g. share, export/import)
  created:  Date;
  uniq:     string;
  ext:      string;
  filename: string; // Preserve so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
  // TODO(cache_user_metadata): Kill null after moving slow UserRec.metadata -> fast UserSource.metadata
  //  - See example load/store code in UserSource (from old EditRec)
  metadata: UserMetadata | null;
}

// XXX(cache_user_metadata): Manually thread UserMetadata down from Source.parse callers, until we can kill null UserSource.metadata
export interface SourceParseOpts {
  userMetadata: UserMetadata | null;
}

export interface SourceShowOpts {
  species: XC | null; // Show species if xc sourceId (using XC dep)
  long?: boolean;
  userMetadata: UserMetadata | null; // XXX(cache_user_metadata)
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

  parse: (sourceId: SourceId, opts: SourceParseOpts): Source | null => {
    const {kind, ssp} = SourceId.split(sourceId);
    return match<string, Source | null>(kind,
      ['xc',          () => mapNull(safeParseIntOrNull(ssp),     xc_id => typed<XCSource>   ({kind: 'xc',   xc_id}))],
      ['user',        () => mapNull(UserSource.parse(ssp, opts), x     => typed<UserSource> ({kind: 'user', ...x}))],
      ['edit',        () => null], // Back compat with old-style edit recs (EditSource, EditRec)
      [match.default, () => { throw `Unknown sourceId type: ${sourceId}`; }],
    );
  },

  parseOrFail: (sourceId: SourceId, opts: SourceParseOpts): Source => {
    return matchNull(Source.parse(sourceId, opts), {
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

export const UserSource = {

  stringify: (source: NoKind<UserSource>): string => {
    // Return preserved filename so we can roundtrip outdated filename formats (e.g. saved user recs from old code versions)
    return source.filename;
  },

  parse: (ssp: string, opts: SourceParseOpts): NoKind<UserSource> | null => {
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
        metadata: ifNull(opts.userMetadata, () => null), // TODO(cache_user_metadata): Kill null case
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

  // TODO(cache_user_metadata): Reference code for AsyncStorage, preserved from old EditRec (before unifying user/edit recs)
  // store: async (pathBasename: string, edit: Edit): Promise<void> => {
  //   const k = `Edit.${pathBasename}`;
  //   await AsyncStorage.setItem(k, Edit.stringify(edit));
  // },
  // load: async (pathBasename: string): Promise<Edit | null> => {
  //   const k = `Edit.${pathBasename}`;
  //   const s = await AsyncStorage.getItem(k);
  //   if (s === null) {
  //     log.warn('Edit.load: Key not found', rich({k}));
  //     return null;
  //   } else {
  //     return Edit.parse(s);
  //   }
  // },

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

// Prefer matchSource(source) to avoid having to handle the null case when sourceId fails to parse
export function matchSourceId<X>(sourceId: SourceId, cases: {
  opts: SourceParseOpts, // XXX(cache_user_metadata)
  null: (sourceId: SourceId) => X,
  xc:   (source: XCSource)   => X,
  user: (source: UserSource) => X,
}): X {
  return matchNull(Source.parse(sourceId, cases.opts), {
    null: ()     => cases.null(sourceId),
    x:    source => matchSource(source, cases),
  });
}
