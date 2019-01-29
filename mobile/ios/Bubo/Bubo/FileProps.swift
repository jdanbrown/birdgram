import Foundation
import SwiftNpy
import SwiftyJSON
import Surge

// TODO Re-home
public protocol Loadable {
  init(load props: FileProps) throws
}

// Hacky simple approximation to Decoder/Decodable [https://developer.apple.com/documentation/swift/codable]
public struct FileProps {

  public let props: Props
  public let path:  String // "@file:..." values are loaded relative to dirname(path)

  // From js FileProps (e.g. app/datatypes.ts)
  public init(fromJs _props: Props) throws {
    var props = _props // Copy so we can mutate
    guard let _path = props.removeValue(forKey: "_path") else { throw AppError("_path is required: \(_props)") }
    guard let path  = _path as? String                   else { throw AppError("_path must be string: \(_path)") }
    self.init(props: props, path: path)
  }

  // From file (e.g. Tests/test_model.swift)
  //  - NOTE If Yaml.loadFromPath is ever a bottleneck, Json.loadFromPath appears to be way faster (~25x)
  public init(path: String) throws {
    let timer = Timer()
    let props = try Yaml.loadFromPath(path) as Props // (Yaml i/o Json because too much unhelpful complexity to convert JSON->Props)
    _Log.info(String(format: "FileProps.init: time[%.3f], path[%@]", timer.time(), path))
    self.init(props: props, path: path)
  }

  public init(
    props: Props,
    path:  String
  ) {
    self.props = props
    self.path  = path
  }

  public func copy(
    props: Props?  = nil,
    path:  String? = nil
  ) -> FileProps {
    return FileProps(
      props: props ?? self.props,
      path:  path  ?? self.path
    )
  }

  // Separate at() from _at() for overloads that don't change type (e.g. [Double])
  public func at<X>(_ k: String) throws -> X { return try _at(k) }

  // FileProps: at():Props -> then re-wrap preserving .path
  public func at(_ k: String) throws -> FileProps   { return copy(props: try at(k)) }
  public func at(_ k: String) throws -> [FileProps] { return (try at(k) as [Props]).map { copy(props: $0) } }

  // Loadable: at():Props -> init(load:)
  public func at<X: Loadable>(_ k: String) throws -> X   { return try X(load: at(k)) }
  public func at<X: Loadable>(_ k: String) throws -> [X] { return try (at(k) as [FileProps]).map { try X(load: $0) } }

  // Types that can resolve via "@file:<path>" (e.g. .npy)
  //  - HACK Assume all inputs are Double and convert from there (true for json inputs, false otherwise)
  public func at(_ k: String) throws -> [Int]          { return toInts   (try at(k) as [Double]) }
  public func at(_ k: String) throws -> [[Int]]        { return toInts   (try at(k) as [[Double]]) }
  public func at(_ k: String) throws -> [Float]        { return toFloats (try at(k) as [Double]) }
  public func at(_ k: String) throws -> [[Float]]      { return toFloats (try at(k) as [[Double]]) }
  public func at(_ k: String) throws -> Matrix<Float>  { return toFloats (try at(k) as Matrix<Double>) }
  public func at(_ k: String) throws -> [Double]       { return try ifFile(k) { $0.array1() } ?? _at(k) as [Double] }
  public func at(_ k: String) throws -> [[Double]]     { return try ifFile(k) { $0.array2() } ?? _at(k) as [[Double]] }
  public func at(_ k: String) throws -> Matrix<Double> { return try ifFile(k) { $0.matrix() } ?? Matrix(at(k) as [[Double]]) }

  // at() i/o subscript because subscript can't `throws` (ugh)
  public func _at<X>(_ k: String) throws -> X {
    // _Log.debug("FileProps.at: k[\(k)]") // XXX Debug
    guard let v = props[k] else { throw AppError("Key not found: \(k) (from path: \(path))") }
    guard let x = v as? X else {
      throw AppError("Expected type \(X.self), got type \(type(of: v)): key[\(k)]=\("\(v)".slice(to: 1000)) (from path: \(path))")
    }
    return x
  }

  // Resolve from file if the value is a string matching "@file:<path>", else use the value as is
  public func ifFile<X>(_ k: String, _ f: (Npy) -> X) throws -> X? {
    guard let v = props[k] else { throw AppError("Key not found: \(k) (from path: \(path))") }
    guard let s = v as? String   else { return nil } // Load normally (e.g. we stored the array/matrix inline instead of in @file)
    guard s.hasPrefix("@file:") else { return nil } // Load normally (unless X=String, this will trigger a type-cast error downstream)
    let (_, ext) = pathSplitExt(s)
    switch ext {
      case "npy": return f(try Npy(path: pathDirname(path) / s.dropPrefix("@file:"))) // Resolve .npy via Npy()
      default:    throw AppError("Unknown file extension[\(ext)] for key[\(k)]=value[\(s)] (from path: \(path))")
    }
  }

}
