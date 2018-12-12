import _ from 'lodash';
import { SQLiteDatabase } from 'react-native-sqlite-storage';
import _SQL from 'sqlstring-sqlite';

import { Log, rich } from './log';
import { noawait, Timer, yamlPretty } from './utils';

const log = new Log('sql');

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
  sql: string, // Assumes caller handles formatting (e.g. via sqlf`...`)
  opts: {
    logTruncate?: number | null, // null to not truncate
    logQueryPlan?: boolean,
  } = {},
): <X>(onResults: (results: ResultSet<Row>) => Promise<X>) => Promise<X> {

  // Default opts
  opts = {...opts,
    logTruncate: opts.logTruncate !== undefined ? opts.logTruncate : 500, // Preserve null
    logQueryPlan: _.defaultTo(opts.logQueryPlan, false),
  };

  // Log query plan (if requested)
  if (opts.logQueryPlan) {
    const timer = new Timer();
    noawait(new Promise((resolve, reject) => {
      db.transaction(tx => {
        tx.executeSql(
          sqlf`explain query plan ${SQL.raw(sql)}`,
          [],
          (tx, {rows}) => {
            const planRows = rows.raw() as QueryPlanRow[];
            const plan = queryPlanFromRows(planRows);
            log.debug('querySql: EXPLAIN QUERY PLAN', `time[${timer.time()}s]`,
              '\n' + queryPlanPretty(plan),
            );
            resolve();
          },
          e => reject(e),
        );
      });
    }));
  }

  // Run query
  const timer = new Timer();
  const sqlTrunc = !opts.logTruncate ? sql : _.truncate(sql, {length: opts.logTruncate});
  log.debug('querySql: Running...', sqlTrunc);
  return onResults => new Promise((resolve, reject) => {
    // TODO How to also `await db.transaction`? (Do we even want to?)
    db.transaction(tx => {
      // [How do you use the Promise version of tx.executeSql without jumping out of the tx?]
      tx.executeSql(
        sql,
        [],
        (tx, {rows, rowsAffected, insertId}) => {
          log.info(`querySql: time[${timer.time()}s]`, `rows[${rows.length}]`, sqlTrunc);
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

//
// `EXPLAIN QUERY PLAN`
//

// Convert flat rows from `explain query plan` to a tree-shaped object that's readable via e.g. pretty()
export interface QueryPlanRow { id: number; parent: number; detail: string; }
export interface QueryPlan    { detail: string; children?: Array<QueryPlan>; }
export function queryPlanFromRows(planRows: Array<QueryPlanRow>): QueryPlan {
  // Make the root node (id=0) ourselves b/c planRows doesn't include it
  const nodeById: {[key: number]: QueryPlan} = {
    0: {detail: 'QUERY PLAN'}, // 'QUERY PLAN' like cli output of `explain query plan`
  };
  // Mutate each planRow into the plan tree
  planRows.forEach(({id, parent, detail}) => {
    const node = {detail};
    nodeById[id] = node;
    const parentNode = nodeById[parent];
    if (parentNode.children === undefined) parentNode.children = [];
    parentNode.children.push(node);
  });
  return nodeById[0];
}

// More compact pretty representation ({detail, children} -> {[detail]: children})
export function queryPlanPretty(plan: QueryPlan): string {
  return yamlPretty(_queryPlanPretty(plan));
}
export interface _QueryPlanPretty { [key: string]: null | Array<_QueryPlanPretty>; }
export function _queryPlanPretty(plan: QueryPlan): _QueryPlanPretty {
  return {
    [plan.detail]: plan.children === undefined ? [] : plan.children.map(_queryPlanPretty),
  };
}
