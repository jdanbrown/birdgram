import _ from 'lodash';
import sqlitePlugin, { SQLitePlugin } from 'react-native-sqlite-plugin-legacy-support';

import { Log, puts, rich, tap } from 'app/log';
import { json, local, matchNull, pretty, typed } from 'app/utils';

const log = new Log('sqlite-async');

// HACK Adapt react-native-sqlite-plugin-legacy-support like react-native-sqlite-storage
//  - TODO Clean up after we de-risk that react-native-sqlite-plugin-legacy-support works

export type OpenArgs     = SQLitePlugin.OpenArgs;
export type DeleteArgs   = SQLitePlugin.DeleteArgs;
export type _Database    = SQLitePlugin.Database;
export type _Results     = SQLitePlugin.Results;
export type _Transaction = SQLitePlugin.Transaction;

export async function openDatabase(args: OpenArgs): Promise<Database> {
  const _db = await _openDatabase(args);
  return new Database(_db);
}

export async function _openDatabase(args: OpenArgs): Promise<_Database> {
  return new Promise<_Database>((resolve, reject) => {
    sqlitePlugin.openDatabase(
      args,
      (_db: _Database) => resolve(_db),
      (e: Error)       => reject(e),
    );
  });
}

export class Database {

  constructor(
    public _db: _Database,
  ) {}

  transaction = async (fn: (tx: Transaction) => void): Promise<void> => {
    return new Promise<void>((resolve, reject) => {
      this._db.transaction(
        _tx => fn(new Transaction(_tx)),
        // NOTE Backwards args: (fn, error, success)
        (e: Error) => reject(e),
        ()         => resolve(),
      );
    });
  };

}

export class Transaction {

  constructor(
    public _tx: _Transaction,
  ) {}

  executeSql = (
    statement: string,
    params?:   any[],
    success?:  (tx: Transaction, results: Results) => void,
    error?:    (tx: Transaction, err: Error)       => boolean | void,
  ): void => {
    return this._tx.executeSql(
      statement,
      params,
      !success ? success : (_tx, _results) => success(new Transaction(_tx), new Results(_results)),
      !error   ? error   : (_tx, err)      => error(new Transaction(_tx), err),
    );
  };

}

export class Results {

  constructor(
    public _results: _Results,
  ) {}

  get rowsAffected(): number             { return this._results.rowsAffected; }
  get insertId():     number | undefined { return this._results.insertId; }

  get rows(): ResultSetRowList {
    const _rows = this._results.rows;
    return {
      length: _rows.length,
      item: i => _rows.item(i),
      raw: () => _.range(_rows.length).map(i => _rows.item(i)),
    };
  }

}

export interface ResultSetRowList {
  length: number;
  item(index: number): object;
  raw(): object[];
}
