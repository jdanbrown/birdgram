// Manual type defs for react-native-sqlite-plugin-legacy-support
//  - Version: https://github.com/brodybits/react-native-sqlite-plugin-legacy-support#042f681
//  - Based on:
//    - https://github.com/DefinitelyTyped/DefinitelyTyped/blob/1b29bd0/types/cordova-sqlite-storage/index.d.ts
//    - https://github.com/DefinitelyTyped/DefinitelyTyped/blob/1b29bd0/types/react-native-sqlite-storage/index.d.ts
declare module 'react-native-sqlite-plugin-legacy-support' {

// Type definitions for cordova-sqlite-storage 1.5
// Project: https://github.com/xpbrew/cordova-sqlite-storage
// Definitions by: rafw87 <https://github.com/rafw87>
// Definitions: https://github.com/DefinitelyTyped/DefinitelyTyped

interface Window {
    sqlitePlugin: SQLitePlugin.SQLite;
}

export const sqlitePlugin: SQLitePlugin.SQLite;
export default sqlitePlugin;

export namespace SQLitePlugin {
    type TransactionFunction = (tx: Transaction) => void;

    type SuccessCallback = () => void;
    type DatabaseSuccessCallback = (db: Database) => void;
    type StatementSuccessCallback = (results: Results) => void;
    type TransactionStatementSuccessCallback = (tx: Transaction, results: Results) => void;

    type ErrorCallback = (err: Error) => void;
    type TransactionStatementErrorCallback = (tx: Transaction, err: Error) => boolean | void;

    interface OpenArgs {
        name: string;
        location?: string;
        iosDatabaseLocation?: string;
        androidDatabaseImplementation?: number;
        androidLockWorkaround?: number;
        createFromLocation?: number;
        [key: string]: any;
    }
    interface DeleteArgs {
        name: string;
        location?: string;
        iosDatabaseLocation?: string;
    }

    interface Results {
        rowsAffected: number;
        insertId?: number;
        rows: {
            length: number;
            item(i: number): any;
        };
    }

    interface Transaction {
        executeSql(statement: string, params?: any[], success?: TransactionStatementSuccessCallback, error?: TransactionStatementErrorCallback): void;
    }

    interface Database {
        transaction(fn: TransactionFunction, error?: ErrorCallback, success?: SuccessCallback): void;
        readTransaction(fn: TransactionFunction, error?: ErrorCallback, success?: SuccessCallback): void;

        executeSql(statement: string, params?: any[], success?: StatementSuccessCallback, error?: ErrorCallback): void;
        sqlBatch(sqlStatements: Array<string|[string, any[]]>, success?: SuccessCallback, error?: ErrorCallback): void;

        close(success?: SuccessCallback, error?: ErrorCallback): void;
    }

    interface SQLite {
        openDatabase(args: OpenArgs, success?: DatabaseSuccessCallback, error?: ErrorCallback): Database;
        deleteDatabase(args: DeleteArgs, success?: SuccessCallback, error?: ErrorCallback): void;
        selfTest(success?: SuccessCallback, error?: ErrorCallback): void;
        echoTest(ok?: (value: string) => void, error?: (msg: string) => void): void;
    }
}

}
