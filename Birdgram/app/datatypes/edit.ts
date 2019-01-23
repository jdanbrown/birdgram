import _ from 'lodash';
import { AsyncStorage } from 'react-native';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
const {base64} = RNFB;

import { Source, SourceId, SourceShowOpts } from 'app/datatypes';
import { debug_print, log, Log, rich } from 'app/log';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchNull, matchUndefined, NoKind,
  Omit, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane, requireSafePath, safeParseInt, safeParseIntOrNull,
  safePath, showDate, showSuffix, splitFirst, stripExt, throw_, tryElse, typed, unjson,
} from 'app/utils';

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
