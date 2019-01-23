import _ from 'lodash';
import moment from 'moment';
import { sprintf } from 'sprintf-js';

import { Edit, EditRec, UserRec, XCRec } from 'app/datatypes';
import { debug_print, log, Log, rich } from 'app/log';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchNull, matchUndefined, NoKind,
  Omit, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane, requireSafePath, safeParseInt, safeParseIntOrNull,
  safePath, showDate, showSuffix, splitFirst, stripExt, throw_, tryElse, typed, unjson,
} from 'app/utils';
import { XC } from 'app/xc';

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
