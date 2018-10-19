import { SQLiteDatabase } from 'react-native-sqlite-storage';
import SqlString from 'sqlstring-sqlite';

import { log } from './log';

export function querySql<Row>(
  db: SQLiteDatabase,
  sql: string,
  params?: any[],
): <X>(onResults: (results: ResultSet<Row>) => X) => Promise<X> {
  console.debug('[querySql]', 'sql:', sql, 'params:', params);
  return onResults => new Promise((resolve, reject) => {
    // TODO How to also `await db.transaction`? (Do we even want to?)
    db.transaction(tx => {
      tx.executeSql( // [How do you use the Promise version of tx.executeSql without jumping out of the tx?]
        // Format using sqlstring-sqlite (e.g. array support) instead of react-native-sqlite-storage (e.g. no array support)
        SqlString.format(sql, params || []),
        [],
        (tx, {rows, rowsAffected, insertId}) => {
          console.info('[querySql]', 'results:', rows.length, 'sql:', sql, 'params:', params);
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
