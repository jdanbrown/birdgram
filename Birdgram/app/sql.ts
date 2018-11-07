import _ from 'lodash';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
import _SQL from 'sqlstring-sqlite';

import { log } from './log';
import { Timer } from './utils';

// Re-export sqlstring as SQL + add extra methods
export const SQL = {..._SQL,

  // Shorthand: SQL.id(x) ~ SQL.raw(SQL.escapeId(x))
  id: (id: string): _SQL.ToSqlString => ({
    toSqlString: () => _SQL.escapeId(id),
  }),

};

// Template literal
//  - Docs: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals
//  - Example usage:
//      querySql(db, sqlf`
//        select *
//        from ${SQL.id('table_name')}
//        where x > ${y * 3}
//        ${SQL.raw(!limit ? '' : 'limit 10')}
//      `)
export function sqlf(strs: TemplateStringsArray, ...args: any[]): string {
  return (
    _(strs)
    .zip(args)
    .flatMap(([str, arg]) => [str, arg === undefined ? '' : SQL.escape(arg)])
    .value()
    .join('')
  );
}

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
  sql = SQL.format(sql, params);
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

// XXX Is this worthwhile given that we immediately SQL.format in the impl anyway? Unlikely that impl will change soon.
export interface BindSql {
  sql: string;
  params?: any[];
}
export function bindSql(sql: string, params?: any[]): BindSql {
  return {sql, params}
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
