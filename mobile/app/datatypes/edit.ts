import _ from 'lodash';
import { AsyncStorage } from 'react-native';
import RNFB from 'rn-fetch-blob';
import { sprintf } from 'sprintf-js';
const {base64} = RNFB;

import { Source, SourceId, SourceShowOpts, UserMetadata } from 'app/datatypes';
import { debug_print, log, Log, rich } from 'app/log';
import {
  assert, basename, chance, ensureDir, ensureParentDir, extname, ifEmpty, ifNil, ifNull, ifUndefined, json,
  jsonSafeError, JsonSafeNumber, Interval, local, mapEmpty, mapNil, mapNull, mapUndefined, match, matchNull,
  matchUndefined, NoKind, Omit, parseDate, parseUrl, parseUrlNoQuery, parseUrlWithQuery, pretty, qsSane,
  requireSafePath, safeParseInt, safeParseIntElseNull, safePath, showDate, showSuffix, splitFirst, stringifyDate,
  stripExt, throw_, tryElse, typed, unjson,
} from 'app/utils';

// A (commited) edit, which maps to a user rec file (contained within UserRec.metadata.edit)
export interface Edit {
  parent: Source;           // A non-edit rec (don't allow O(n) parent chains)
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

export const Edit = {

  // Can't simply json/unjson because it's unsafe for Interval (which needs to JsonSafeNumber)
  jsonSafe: (edit: Edit): any => {
    return typed<{[key in keyof Edit]: any}>({
      parent: Source.jsonSafe(edit.parent),
      edits:  edit.edits.map(draftEdit => DraftEdit.jsonSafe(draftEdit)),
    });
  },
  unjsonSafe: (x: any): Edit => {
    return {
      parent: Edit._required(x, 'parent', (x: any)    => Source.unjsonSafe(x)),
      edits:  Edit._required(x, 'edits',  (xs: any[]) => xs.map(x => DraftEdit.unjsonSafe(x))),
    };
  },

  show: (edit: Edit, opts: SourceShowOpts): string => {
    return (
      [
        Source.show(edit.parent, opts),
        ...edit.edits.map(x => DraftEdit.show(x)),
      ]
      .filter(x => !_.isEmpty(x)) // Exclude null, undefined, '' (and [], {})
      .join(' ')
    );
  },

  // Parse results of qsSane.parse
  //  - TODO Add runtime type checks for X [how?] so we fail when q[k] isn't an X
  _optional: <X, Y>(q: any, k: keyof Edit, f: (x: X) => Y): Y | undefined => mapUndefined(q[k], x => f(x)),
  _required: <X, Y>(q: any, k: keyof Edit, f: (x: X) => Y): Y             => f(Edit._requireKey(q, k)),
  _requireKey: (q: any, k: keyof Edit): any => ifUndefined(q[k], () => throw_(`Edit: Field '${k}' required: ${json(q)}`)),

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

  stringify: (clip: Clip): string => json(Clip.jsonSafe(clip)),
  parse:     (x: string): Clip    => Clip.unjsonSafe(unjson(x)),

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
