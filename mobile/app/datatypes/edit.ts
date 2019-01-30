import _ from 'lodash';
import { AsyncStorage } from 'react-native';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
const {base64} = RNFB;

import { Source, SourceId, SourceShowOpts } from 'app/datatypes';
import { debug_print, log, Log, rich } from 'app/log';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  jsonSafeError, JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchNull,
  matchUndefined, NoKind, Omit, parseDate, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane,
  requireSafePath, safeParseInt, safeParseIntOrNull, safePath, showDate, showSuffix, splitFirst, stringifyDate,
  stripExt, throw_, tryElse, typed, unjson,
} from 'app/utils';

// XXX(unify_edit_user_recs)
// A (commited) edit, which maps to an edit rec file
export interface Edit extends DraftEdit {
  parent:  SourceId;
  // TODO(unify_edit_user_recs): Kill Edit.{created,uniq} since they're redundant with UserSource.{created,uniq}
  created: Date;
  uniq:    string; // Ensure a new filename for each EditRec (so we don't have to think about file collisions / purity)
}

// A (commited) edit, which maps to a user rec file
export interface Edit_v2 {
  parent: SourceId;         // A non-edit rec (don't allow O(n) parent chains)
  edits:  Array<DraftEdit>; // The full sequence of edits from .parent (editing an edit rec preserved .parent and extends .edits)
}

// A draft edit, which isn't yet associated with any user rec file
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

export const Edit_v2 = {

  // Can't simply json/unjson because it's unsafe for Interval (which needs to JsonSafeNumber)
  jsonSafe: (edit: Edit_v2): any => {
    return typed<{[key in keyof Edit_v2]: any}>({
      parent: edit.parent,
      edits:  edit.edits.map(draftEdit => DraftEdit.jsonSafe(draftEdit)),
    });
  },
  unjsonSafe: (x: any): Edit_v2 => {
    return {
      parent: Edit_v2._required(x, 'parent', (x: string) => x),
      edits:  Edit_v2._required(x, 'edits',  (xs: any[]) => xs.map(x => DraftEdit.unjsonSafe(x))),
    };
  },

  show: (edit: Edit_v2, opts: SourceShowOpts): string => {
    const parts = [
      SourceId.show(edit.parent, opts),
      ...edit.edits.map(x => DraftEdit.show(x)),
    ];
    return (parts
      .filter(x => !_.isEmpty(x)) // Exclude null, undefined, '' (and [], {})
      .join(' ')
    );
  },

  // Parse results of qsSane.parse
  //  - TODO Add runtime type checks for X [how?] so we fail when q[k] isn't an X
  _optional: <X, Y>(q: any, k: keyof Edit_v2, f: (x: X) => Y): Y | undefined => mapUndefined(q[k], x => f(x)),
  _required: <X, Y>(q: any, k: keyof Edit_v2, f: (x: X) => Y): Y             => f(Edit_v2._requireKey(q, k)),
  _requireKey: (q: any, k: keyof Edit_v2): any => ifUndefined(q[k], () => throw_(`Edit_v2: Field '${k}' required: ${json(q)}`)),

};

// TODO Avoid unbounded O(parents) length for edit sourceId's [already solved for edit rec filenames]
//  - Add a layer of indirection for async (sync parse -> async load)
//  - Make it read Edit metadata to populate Edit.parent (from AsyncStorage like EditRec.sourceFromAudioPath)
export const Edit = {

  // XXX(unify_edit_user_recs)
  to_v2: (edit: Edit): Edit_v2 => {
    return {
      parent: edit.parent,
      edits:  [{clips: edit.clips}],
    };
  },

  // Various crap to render as url query string i/o json [is this actually worthwhile?]
  stringify: (edit: Edit): string => {
    const x = Edit.jsonSafe(edit);
    return qsSane.stringify(typed<{[key in keyof Edit]: any}>({
      // HACK parent: Avoid chars that need uri encoding, since something keeps garbling them [router / location?]
      parent:  base64.encode(x.parent).replace(/=+$/, ''),
      created: Source.stringifyDate(parseDate(x.created)), // iso8601 string -> 'YYYYMMDDThhmmssSSS'
      uniq:    x.uniq,
      clips:   x.clips,
    }));
  },
  parse: (s: string): Edit | null => {
    var q: any; // To include in logging (as any | null)
    try {
      q = qsSane.parse(s);
      return Edit.unjsonSafe({
        // HACK parent: Avoid chars that need uri encoding, since something keeps garbling them [router / location?]
        parent:  /:/.test(q.parent) ? q.parent : base64.decode(q.parent),
        created: stringifyDate(Source.parseDate(q.created)), // 'YYYYMMDDThhmmssSSS' -> iso8601 string
        uniq:    q.uniq,
        clips:   q.clips,
      });
    } catch (e) {
      log.warn('Edit.parse: Failed', rich({e: jsonSafeError(e), s, q}));
      return null;
    }
  },

  // Can't simply json/unjson because it's unsafe for Interval (which needs to JsonSafeNumber)
  jsonSafe: (edit: Edit): any => {
    return typed<{[key in keyof Edit]: any}>({
      parent:  edit.parent,
      created: stringifyDate(edit.created),
      uniq:    edit.uniq,
      clips:   mapUndefined(edit.clips, clips => clips.map(clip => Clip.jsonSafe(clip))),
    });
  },
  unjsonSafe: (x: any): Edit => {
    return {
      parent:  Edit._required(x, 'parent',  (x: string) => x),
      created: Edit._required(x, 'created', (x: string) => parseDate(x)),
      uniq:    Edit._required(x, 'uniq',    (x: string) => x),
      clips:   Edit._optional(x, 'clips',   (xs: any[]) => xs.map(x => Clip.unjsonSafe(x))),
    };
  },

  show: (edit: Edit, opts: SourceShowOpts): string => {
    const parts = [
      // Ignore uniq for human
      SourceId.show(edit.parent, opts),
      DraftEdit.show(edit),
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

  // TODO(user_metadata): Migrate old edit recs so we can kill this v0 code, or keep it (forever) for back compat?
  // Store Edit in AsyncStorage b/c we can't shove it all into the audio filename
  //  - TODO(?): Unwind this once we move Edit to audio metadata (tags stored in the file)
  //    - Else users can't share edit recs (without transmitting AsyncStorage items, which would be anti-simple)
  //    - Back compat: if no audio metadata but AsyncStorage item exists, write AsyncStorage item to audio metadata
  //      - This will trigger on each edit rec load
  //      - And we can ensure it triggers in the share path (once we get there)
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

  hasEdits: (edit: DraftEdit): boolean => {
    return !_(edit).values().every(_.isEmpty);
  },

  // For NativeSpectro.editAudioPathToAudioPath [see HACK there]
  jsonSafe: (draftEdit: DraftEdit): any => {
    return {
      clips: mapUndefined(draftEdit.clips, xs => xs.map(x => Clip.jsonSafe(x))),
    };
  },
  unjsonSafe: (x: any): DraftEdit => {
    return {
      clips: mapUndefined(x.clips, xs => xs.map((x: any) => Clip.unjsonSafe(x))),
    };
  },

  show: (draftEdit: DraftEdit): string => {
    const parts = [
      // Strip outer '[...]' per clip so we can wrap all clips together in one '[...]'
      sprintf('[%s]', (draftEdit.clips || []).map(x => Clip.show(x).replace(/^\[|\]$/g, '')).join(',')),
    ];
    return (parts
      .filter(x => !_.isEmpty(x)) // Exclude null, undefined, '' (and [], {})
      .join(' ')
    );
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
