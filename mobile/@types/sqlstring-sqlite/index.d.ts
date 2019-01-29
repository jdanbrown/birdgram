declare module 'sqlstring-sqlite' {

// Type definitions for sqlstring 2.2
// Project: https://github.com/mysqljs/sqlstring
// Definitions by: Marvin Hagemeister <https://github.com/marvinhagemeister>
// Definitions: https://github.com/DefinitelyTyped/DefinitelyTyped
// TypeScript Version: 2.3

export function format(sql: string, values: undefined | null | any[]): string;
export function escape(value: any): string;
export function escapeId(value: any, dotQualifier?: boolean): string;
export function raw(sql: string): Raw;

export type Raw = ToSqlString;
export interface ToSqlString {
  toSqlString: () => string;
}

}
