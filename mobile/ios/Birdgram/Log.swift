//
//  Based on:
//  RCTLog.swift
//  Created by Jimmy Dee on 4/5/17.
//  Copyright © 2017 Branch Metrics. All rights reserved.
//

// Under at least some conditions, output from NSLog has been unavailable in the RNBranch module.
// Hence that module uses the RCTLog macros from <React/RCTLog.h>. The React logger is nicer than
// NSLog anyway, since it provides log levels with runtime filtering, file and line context and
// an identifier for the thread that logged the message.
//
// This wrapper lets you use functions with the same name in Swift. For example:
//
// RCTLogInfo("application launched")
//
// generates
//
// 2017-04-06 12:31:09.611 [info][tid:main][AppDelegate.swift:18] application launched
//
// This is currently part of this sample app. There may be some issues integrating it into an
// Objective-C library, either react-native-branch or react-native itself, but it may find its
// way into one or the other eventually. Feel free to reuse it as desired.

import Bubo // (For __DEV__)

public enum Log {

  public static func error(_ msg: @autoclosure () throws -> String, _ file: String = #file, _ line: UInt = #line) rethrows {
    if __DEV__ { RCTSwiftLog.error ("ERROR \(try msg())", file: file, line: line) }
  }

  public static func warn(_ msg: @autoclosure () throws -> String, _ file: String = #file, _ line: UInt = #line) rethrows {
    if __DEV__ { RCTSwiftLog.warn  ("WARN  \(try msg())", file: file, line: line) }
  }

  public static func info(_ msg: @autoclosure () throws -> String, _ file: String = #file, _ line: UInt = #line) rethrows {
    if __DEV__ { RCTSwiftLog.info  ("INFO  \(try msg())", file: file, line: line) }
  }

  public static func debug(_ msg: @autoclosure () throws -> String, _ file: String = #file, _ line: UInt = #line) rethrows {
    if __DEV__ { RCTSwiftLog.trace ("DEBUG \(try msg())", file: file, line: line) }
  }

  // Aliases
  public static func log   (_ msg: @autoclosure () throws -> String, _ file: String = #file, _ line: UInt = #line) rethrows {
    try info(msg(), file, line)
  }
  public static func trace (_ msg: @autoclosure () throws -> String, _ file: String = #file, _ line: UInt = #line) rethrows {
    try debug(msg(), file, line)
  }

  // HACK A variant of debug_print that shows up in the rndebugger console (unlike swift print())
  public static func debug_print<X>(_ x: X, _ file: String = #file, _ line: UInt = #line) -> X {
    if __DEV__ { RCTSwiftLog.trace ("PRINT \(x)", file: file, line: line) }
    return x
  }

}
