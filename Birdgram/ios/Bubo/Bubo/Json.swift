import SwiftyJSON
import Yams

// HACK b/c json/yaml are too complicated to make do simple things
public func show(_ xs: Dictionary<String, Any?>) -> String {
  return String(describing: xs)
}
public func pretty(_ xs: Dictionary<String, Any?>) -> String {
  return String(format: "%@", xs)
}

// TODO Poop: Both SwiftyJSON and Foundation's JSONSerialization only allow proper dict/list at top level
//  - e.g. SwiftyJSON's JSON("foo") tries to load instead of dump :/
public enum Json {

  // public static func dump<X>(_ x: X) throws -> String {
  //   guard let s = JSON(x).rawString() else {
  //     throw AppError("Failed to dump to json: \(x)")
  //   }
  //   return s
  // }

  // public static func load(_ x: String) -> JSON {
  //   return JSON(parseJSON: x)
  // }

  // // XXX -> SwiftyJSON
  // public static func _json<X>(_ x: X) throws -> String {
  //   return try String(data: JSONSerialization.data(withJSONObject: x), encoding: .utf8)!
  // }

}

public enum Yaml {

  public static func dump<X>(_ x: X) throws -> String {
    return try Yams.dump(
      object: x,
      width: -1,
      allowUnicode: true
    )
  }

}
