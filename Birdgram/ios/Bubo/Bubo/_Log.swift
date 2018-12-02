// Like Birdgram/Log, but not connected to react-native
//  - TODO How to RCTLog from Bubo i/o Birdgram?
public enum _Log {

  public static func error (_ msg: @autoclosure () -> String) { if __DEV__ { print("ERROR", msg()) } }
  public static func warn  (_ msg: @autoclosure () -> String) { if __DEV__ { print("WARN ", msg()) } }
  public static func info  (_ msg: @autoclosure () -> String) { if __DEV__ { print("INFO ", msg()) } }
  public static func debug (_ msg: @autoclosure () -> String) { if __DEV__ { print("DEBUG", msg()) } }

  // Aliases
  public static func log   (_ msg: @autoclosure () -> String) { info(msg) }
  public static func trace (_ msg: @autoclosure () -> String) { debug(msg) }

}
