// TODO Use more standard stuff (print/debugPrint/dump):
//  - https://developer.apple.com/documentation/swift/swift_standard_library/debugging_and_reflection

import Surge
import SwiftyJSON
import Yams

//
// HACK b/c json/yaml are too complicated to make do simple things
//  - TODO Figure out how to make generic show/pp (how to do typeclass-style generic overloading in swift?)
//

// Dictionary<String, Any?>
public func p      (_ x: Dictionary<String, Any?>) { print(show(x)) }
public func pp     (_ x: Dictionary<String, Any?>) { print(pretty(x)) }
public func show   (_ x: Dictionary<String, Any?>) -> String { return String(describing: x) }
public func pretty (_ x: Dictionary<String, Any?>) -> String { return String(format: "%@", x) }

// [Double]
public func p      (_ xs: [Double]) { print(show(xs)) }
public func pp     (_ xs: [Double]) { print(pretty(xs)) }
public func show   (_ xs: [Double], prec: Int = 3) -> String { return "[\(  xs.map { x in String(format: "%.\(prec)f", x).padLeft(prec + 3) }.joined(separator: " "))]" }
public func pretty (_ xs: [Double], prec: Int = 3) -> String { return "[\n\(xs.map { x in String(format: "%.\(prec)f", x).padLeft(prec + 3) }.joined(separator: "\n  "))\n]" }

// [Float]
public func p      (_ xs: [Float]) { print(show(xs)) }
public func pp     (_ xs: [Float]) { print(pretty(xs)) }
public func show   (_ xs: [Float], prec: Int = 3) -> String { return "[\(  xs.map { x in String(format: "%.\(prec)f", x).padLeft(prec + 3) }.joined(separator: " "))]" }
public func pretty (_ xs: [Float], prec: Int = 3) -> String { return "[\n\(xs.map { x in String(format: "%.\(prec)f", x).padLeft(prec + 3) }.joined(separator: "\n  "))\n]" }

// [Float]
public func p      (_ xs: ArraySlice<Float>) { print(show(xs)) }
public func pp     (_ xs: ArraySlice<Float>) { print(pretty(xs)) }
public func show   (_ xs: ArraySlice<Float>, prec: Int = 3) -> String { return "[\(  xs.map { x in String(format: "%.\(prec)f", x).padLeft(prec + 3) }.joined(separator: " "))]" }
public func pretty (_ xs: ArraySlice<Float>, prec: Int = 3) -> String { return "[\n\(xs.map { x in String(format: "%.\(prec)f", x).padLeft(prec + 3) }.joined(separator: "\n  "))\n]" }

// [Int]
public func p      (_ xs: [Int]) { print(show(xs)) }
public func pp     (_ xs: [Int]) { print(pretty(xs)) }
public func show   (_ xs: [Int]) -> String { return "[\(xs.map { x in String(format: "%d", x) }.joined(separator: " "))]" }
public func pretty (_ xs: [Int]) -> String { return "[\n\(xs.map { x in String(format: "%d", x) }.joined(separator: "\n  "))\n]" }

// [UInt8]
public func p      (_ xs: [UInt8]) { print(show(xs)) }
public func pp     (_ xs: [UInt8]) { print(pretty(xs)) }
public func show   (_ xs: [UInt8]) -> String { return "[\(xs.map { x in String(format: "%d", x) }.joined(separator: " "))]" }
public func pretty (_ xs: [UInt8]) -> String { return "[\n\(xs.map { x in String(format: "%d", x) }.joined(separator: "\n  "))\n]" }

// [Int]
public func p      (_ xs: ArraySlice<Int>) { print(show(xs)) }
public func pp     (_ xs: ArraySlice<Int>) { print(pretty(xs)) }
public func show   (_ xs: ArraySlice<Int>) -> String { return "[\(xs.map { x in String(format: "%d", x) }.joined(separator: " "))]" }
public func pretty (_ xs: ArraySlice<Int>) -> String { return "[\n\(xs.map { x in String(format: "%d", x) }.joined(separator: "\n  "))\n]" }

// [UInt8]
public func p      (_ xs: ArraySlice<UInt8>) { print(show(xs)) }
public func pp     (_ xs: ArraySlice<UInt8>) { print(pretty(xs)) }
public func show   (_ xs: ArraySlice<UInt8>) -> String { return "[\(xs.map { x in String(format: "%d", x) }.joined(separator: " "))]" }
public func pretty (_ xs: ArraySlice<UInt8>) -> String { return "[\n\(xs.map { x in String(format: "%d", x) }.joined(separator: "\n  "))\n]" }

// Matrix<Float>
public func p      (_ X: Matrix<Float>) { print(show(X)) }
public func pp     (_ X: Matrix<Float>) { print(pretty(X)) }
public func show   (_ X: Matrix<Float>, prec: Int = 3) -> String { return pretty(X, prec: prec) }
public func pretty (_ X: Matrix<Float>, prec: Int = 3) -> String {
  return X.map { row in show(row, prec: prec) }.joined(separator: "\n")
}

// [[Float]]
public func p      (_ X: [[Float]]) { print(show(X)) }
public func pp     (_ X: [[Float]]) { print(pretty(X)) }
public func show   (_ X: [[Float]], prec: Int = 3) -> String { return pretty(X, prec: prec) }
public func pretty (_ X: [[Float]], prec: Int = 3) -> String {
  return X.map { row in show(row, prec: prec) }.joined(separator: "\n")
}

// TODO Re-home
public func rounded(_ x: Double, _ prec: Int) -> Double {
  var pow: Double = 1.0
  for _ in 0..<prec { pow *= 10 } // TODO How the f do you pow(a,b)?
  return (x * pow).rounded() / pow
}
public func rounded(_ x: Float, _ prec: Int) -> Float {
  var pow: Float = 1.0
  for _ in 0..<prec { pow *= 10 } // TODO How the f do you pow(a,b)?
  return (x * pow).rounded() / pow
}

//
// json/yaml
//

// TODO Poop: Both SwiftyJSON and Foundation's JSONSerialization only allow proper dict/list at top level
//  - e.g. SwiftyJSON's JSON("foo") tries to load instead of dump :/
public enum Json {

  // XXX -> SwiftyJSON
  public static func dumps<X>(_ x: X) throws -> String {
    return try String(data: JSONSerialization.data(withJSONObject: x), encoding: .utf8)!
  }

  // TODO How to normal dumps with SwiftyJSON? Currently relying on normal JSONSerialization for non-pretty dumps
  public static func pretty<X>(_ x: X) throws -> String {
    guard let s = JSON(x).rawString() else {
      throw AppError("Failed to dumps to json: \(x)")
    }
    return s
  }

  public static func loads(_ x: String) -> JSON {
    return JSON(parseJSON: x)
  }

  public static func loadFromPath(_ path: String, encoding: String.Encoding = .utf8) throws -> JSON {
    return loads(try String(contentsOfFile: path, encoding: encoding))
  }

}

public enum Yaml {

  public static func dumps<X>(_ x: X) throws -> String {
    return try Yams.dump(
      object: x,
      width: -1,
      allowUnicode: true
    )
  }

}
