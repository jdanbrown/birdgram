import _ from 'lodash';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
import SqlString from 'sqlstring-sqlite';

import { log } from './log';
import { Timer } from './utils';

export function querySql<Row>(
  db: SQLiteDatabase,
  _sql: string | BindSql,
  opts: {logTruncate: number} = {logTruncate: 200},
): <X>(onResults: (results: ResultSet<Row>) => Promise<X>) => Promise<X> {

  // Unpack args
  let sql: string;
  let params: any[] | undefined;
  if (typeof _sql !== 'string') {
    ({sql, params} = _sql);
  } else {
    sql = _sql;
    params = undefined;
  }

  // sql + params
  //  - Format using sqlstring-sqlite (e.g. array support) instead of react-native-sqlite-storage (e.g. no array support)
  sql = formatSql(sql, params);
  params = undefined;

  const timer = new Timer();
  const sqlTrunc = _.truncate(sql, {length: opts.logTruncate});
  log.debug('[querySql] Running...', sqlTrunc);
  return onResults => new Promise((resolve, reject) => {
    // TODO How to also `await db.transaction`? (Do we even want to?)
    db.transaction(tx => {
      tx.executeSql( // [How do you use the Promise version of tx.executeSql without jumping out of the tx?]
        sql,
        [],
        (tx, {rows, rowsAffected, insertId}) => {
          log.info('[querySql]', `time[${timer.time()}s]`, `rows[${rows.length}]`, sqlTrunc);
          resolve(onResults({
            rows: {
              length: rows.length,
              item: i => (rows.item(i) as unknown as Row),
              raw: () => (rows.raw() as unknown[] as Row[]),
            },
            insertId,
            rowsAffected,
          }))
        },
        e => reject(e),
      );
    });
  });

}

// TODO Is this worthwhile given that we immediately formatSql in the impl anyway? Unlikely that impl will change soon.
export interface BindSql {
  sql: string;
  params?: any[];
}
export function bindSql(sql: string, params?: any[]): BindSql {
  return {sql, params}
}

export function formatSql(sql: string, params?: any[]): string {
  return SqlString.format(sql, params || []);
}

// Mimic @types/react-native-sqlite-storage, but add <Row>
export interface ResultSet<Row> {
  rows: ResultSetRowList<Row>;
  rowsAffected: number;
  insertId: number;
}
export interface ResultSetRowList<Row> {
  length: number;
  item(index: number): Row;
  raw(): Row[];
}
