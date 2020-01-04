/*
 * Copyright (c) 2012-present Christopher J. Brody (aka Chris Brody)
 * Copyright (C) 2011 Davide Bertola
 *
 * This library is available under the terms of the MIT License (2008).
 * See http://opensource.org/licenses/alphabetical for full text.
 */

// #import <Cordova/CDVPlugin.h>

#import <Foundation/Foundation.h>

// Used to remove dependency on sqlite3.h in this header:
struct sqlite3;

enum WebSQLError {
    UNKNOWN_ERR = 0,
    DATABASE_ERR = 1,
    VERSION_ERR = 2,
    TOO_LARGE_ERR = 3,
    QUOTA_ERR = 4,
    SYNTAX_ERR = 5,
    CONSTRAINT_ERR = 6,
    TIMEOUT_ERR = 7
};
typedef int WebSQLError;

@interface SQLitePlugin : NSObject {
    NSMutableDictionary *openDBs;
}

@property (nonatomic, copy) NSMutableDictionary *openDBs;
@property (nonatomic, copy) NSMutableDictionary *appDBPaths;

-(void) pluginInitialize;

// Self-test
-(void) echoIt: (NSArray *)arguments
         error: (void (^)(NSObject *))error
       success: (void (^)(NSObject *))success;

// Open / Close / Delete

-(void)openNow: (NSArray *)arguments
         error: (void (^)(NSObject *))error
       success: (void (^)())success;

-(void)closeNow: (NSArray *)arguments
          error: (void (^)(NSObject *))error
        success: (void (^)())success;

-(void)deleteNow: (NSArray *)arguments
           error: (void (^)(NSObject *))error
         success: (void (^)())success;

// Batch processing interface

-(void) executeSqlBatchNow: (NSArray *)arguments
                     error: (void (^)(NSObject *))error
                   success: (void (^)(NSObject *))success;

#if 0
-(void) executeSqlBatchNow: (CDVInvokedUrlCommand*)command;
#endif

@end /* vim: set expandtab : */
