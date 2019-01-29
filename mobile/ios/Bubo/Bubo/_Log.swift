// Like Birdgram/Log, but not connected to react-native
//  - TODO How to RCTLog from Bubo i/o Birdgram?
public enum _Log {

  public static func error (_ msg: @autoclosure () throws -> String) rethrows { if __DEV__ { print("ERROR", try msg()) } }
  public static func warn  (_ msg: @autoclosure () throws -> String) rethrows { if __DEV__ { print("WARN ", try msg()) } }
  public static func info  (_ msg: @autoclosure () throws -> String) rethrows { if __DEV__ { print("INFO ", try msg()) } }
  public static func debug (_ msg: @autoclosure () throws -> String) rethrows { if __DEV__ { print("DEBUG", try msg()) } }

  // Aliases
  public static func log   (_ msg: @autoclosure () throws -> String) rethrows { try info  (try msg()) }
  public static func trace (_ msg: @autoclosure () throws -> String) rethrows { try debug (try msg()) }

}
