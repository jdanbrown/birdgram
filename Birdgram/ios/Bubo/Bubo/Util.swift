import Foundation

// Mimic react-native js __DEV__ to distinguish Xcode Debug vs. Release build
//  - Keep same name across js + swift for grep-ability
#if DEBUG
public let __DEV__ = true
#else
public let __DEV__ = false
#endif

// Based on https://gist.github.com/nicklockwood/c5f075dd9e714ad8524859c6bd46069f
public enum AppError: Error, CustomStringConvertible {

  case message(String)
  case generic(Error)

  public init(_ message: String) {
    self = .message(message)
  }

  public init(_ error: Error) {
    if let error = error as? AppError {
      self = error
    } else {
      self = .generic(error)
    }
  }

  public var description: String {
    switch self {
      case let AppError.message(message): return message
      case let AppError.generic(error):   return (error as CustomStringConvertible).description
    }
  }

}

public func local<X>(f: () throws -> X) rethrows -> X {
  return try f()
}

// Generic <X> i/o Never because Never isn't bottom [https://forums.swift.org/t/pitch-never-as-a-bottom-type/5920]
public func throw_<X>(_ e: Error) throws -> X {
  throw e
}

public func TODO<X>(file: StaticString = #file, line: UInt = #line) -> X {
  preconditionFailure("TODO", file: file, line: line)
}

// For debugging
public func puts        <X>(_ x: X) -> X { return tap(x) { x in print("puts", x) } }
public func debug_print <X>(_ x: X) -> X { return tap(x) { x in print("PRINT", x) } }
public func tap         <X>(_ x: X, _ f: (X) -> Void) -> X { f(x); return x }

public func checkStatus(_ status: OSStatus) throws -> Void {
  if (status != 0) {
    throw NSError(domain: NSOSStatusErrorDomain, code: Int(status))
  }
}

public func pathJoin(_ paths: String...) -> String {
  return paths.joined(separator: "/")
}

public func / (x: String, y: String) -> String {
  return pathJoin(x, y)
}

public func pathDirname(_ path: String) -> String {
  return (path as NSString).deletingLastPathComponent
}

public func pathBasename(_ path: String) -> String {
  return (path as NSString).lastPathComponent
}

public func pathSplitExt(_ path: String) -> (name: String, ext: String) {
  return (
    name: (path as NSString).deletingPathExtension,
    ext:  (path as NSString).pathExtension
  )
}

public func ensureDir(_ path: String) throws -> String {
  try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
  return path
}

public func ensureParentDir(_ path: String) throws -> String {
  let _ = try ensureDir(pathDirname(path))
  return path
}

public func documentsDirectory() -> String {
  // https://stackoverflow.com/questions/24055146/how-to-find-nsdocumentdirectory-in-swift
  return NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
}

// Common type for react-native js/ios serdes
public typealias Props = Dictionary<String, Any>

public func propsGetOptional<X>(_ props: Props, _ key: String) throws -> X? {
  guard let any: Any = props[key] else { return nil }
  guard let x:   X   = any as? X  else { throw AppError("Failed to convert \(key)[\(any)] to type \(X.self)") }
  return x
}

// FIXME Doesn't support nil values (different from missing keys)
public func propsGetRequired<X>(_ props: Props, _ key: String) throws -> X {
  guard let x: X = try propsGetOptional(props, key) else { throw AppError("\(key) is required, in props: \(props)") }
  return x
}

extension Float {

  public func ifNan(_ x: Float) -> Float {
    return !self.isNaN ? self : x;
  }

  public func isFiniteOr(_ x: Float) -> Float {
    return self.isFinite ? self : x;
  }

}

public struct Interval<X: Comparable>: CustomStringConvertible {

  public let lo: X
  public let hi: X

  public init(_ lo: X, _ hi: X) {
    self.lo = lo
    self.hi = hi
  }

  public func clamp(_ x: X) -> X {
    return x.clamped(lo, hi)
  }

  // Union (expand ranges)
  public static func | (_ a: Interval<X>, _ b: Interval<X>) -> Interval<X> {
    return Interval(min(a.lo, b.lo), max(a.hi, b.hi))
  }

  // Intersect (contract ranges)
  public static func & (_ a: Interval<X>, _ b: Interval<X>) -> Interval<X>? {
    let (lo, hi) = (max(a.lo, b.lo), min(a.hi, b.hi))
    return lo > hi ? nil : Interval(lo, hi)
  }

  public var description: String {
    get { return "Interval(\(lo), \(hi))" }
  }

}

extension Interval where X == Float {

  // Normalize [lo,hi] -> [0,1] (clamp values out of [lo,hi])
  public func norm(_ x: Float) -> Float {
    return (x.clamped(lo, hi) - lo) / (hi - lo)
  }

  public static let bottom = Interval(Float.infinity,  -Float.infinity) // Unit for union, zero for intersect
  public static let top    = Interval(-Float.infinity, Float.infinity)  // Unit for intersect, zero for union

}

extension Comparable {

  public func clamped(_ lo: Self, _ hi: Self) -> Self {
    return min(max(self, lo), hi)
  }

}

extension Collection {

  // Already exists! https://developer.apple.com/documentation/swift/sequence/2905332-flatmap
  //  - NOTE flatMap seems to be _way_ faster than `Array(....joined())` [just optimized multiple occurrences for big gains]
  //  - NOTE To avoid deprecation errors, make sure f returns [C.Element] instead of C
  // public func flatMap(_ xs: Self, _ f: (Element) -> Self) -> FlattenCollection<[Self]> {
  //   return xs.map(f).joined()
  // }

  public func only() -> Element {
    if count != 1 { preconditionFailure("only: Expected 1 element, have \(count) elements") }
    return first!
  }

}

extension String {

  // https://stackoverflow.com/a/46133083/397334
  subscript(_ range: CountableRange<Int>) -> String {
    let a = index(startIndex, offsetBy: max(0, range.lowerBound))
    let b = index(startIndex, offsetBy: min(self.count, range.upperBound))
    return String(self[a..<b])
  }

  // Less fussy alternative to subscript[RangeExpression]
  //  - Ignores out-of-range indices (like py) i/o fatal-ing
  //  - Returns eager Collection i/o potentially lazy SubSequence (e.g. xs.slice(...) i/o Array(xs.slice(...)))
  //  - TODO Move up to Collection: how to generically init(...) at the end?
  public func slice(from: Int? = nil, to: Int? = nil) -> String {
    // precondition(to == nil || through == nil, "Can't specify both to[\(to)] and through[\(through)]") // TODO through:
    let (startIndex, endIndex) = (0, count)
    var a = from ?? startIndex
    var b = to   ?? endIndex
    if a < 0 { a = count + a }
    if b < 0 { b = count + b }
    a = a.clamped(startIndex, endIndex)
    b = b.clamped(a,          endIndex)
    return String(self[a..<b])
  }

  public func dropPrefix(_ s: String) -> String {
    precondition(hasPrefix(s), "dropPrefix: string[\(self)] doesn't have prefix[\(s)]")
    return String(self.dropFirst(s.count))
  }

  public func dropSuffix(_ s: String) -> String {
    precondition(hasSuffix(s), "dropSuffix: string[\(self)] doesn't have suffix[\(s)]")
    return String(self.dropLast(s.count))
  }

}

extension Array {

  public func chunked(_ chunkSize: Int) -> [[Element]] {
    return stride(from: 0, to: count, by: chunkSize).map {
      Array(self[$0 ..< Swift.min($0 + chunkSize, count)])
    }
  }

  public func repeated(_ n: Int) -> [Element] {
    return [Element]([Array](repeating: self, count: n).joined())
  }

  // Less fussy alternative to subscript[RangeExpression]
  //  - Ignores out-of-range indices (like py) i/o fatal-ing
  //  - Returns eager Collection i/o potentially lazy SubSequence (e.g. xs.slice(...) i/o Array(xs.slice(...)))
  //  - Impose `Index == Int` constraint so we don't have to figure out how to do non-crashing arithmetic with .formIndex
  //  - TODO Move up to Collection: how to generically init(...) at the end?
  public func slice(from: Index? = nil, to: Index? = nil) -> Array {
    // precondition(to == nil || through == nil, "Can't specify both to[\(to)] and through[\(through)]") // TODO through:
    var a = from ?? startIndex
    var b = to   ?? endIndex
    if a < 0 { a = count + a }
    if b < 0 { b = count + b }
    a = a.clamped(startIndex, endIndex)
    b = b.clamped(a,          endIndex)
    return Array(self[a..<b])
  }

}

extension ArraySlice {

  public func chunked(_ chunkSize: Int) -> [[Element]] {
    return stride(from: 0, to: count, by: chunkSize).map {
      Array(self[$0 ..< Swift.min($0 + chunkSize, count)])
    }
  }

}

extension StringProtocol {

  public func padLeft(_ n: Int, _ element: Element = " ") -> String {
    return String(repeatElement(element, count: Swift.max(0, n - count))) + suffix(Swift.max(count, count - n))
  }

  public func padRight(_ n: Int, _ element: Element = " ") -> String {
    return suffix(Swift.max(count, count - n)) + String(repeatElement(element, count: Swift.max(0, n - count)))
  }

}

extension Dictionary {

  public func has(_ k: Key) -> Bool {
    return index(forKey: k) != nil
  }

  public mutating func getOrSet(_ k: Key, _ f: () -> Value) -> Value {
    // Test for key using index() i/o self[k], since the latter conflates nil for missing key with nil values
    if (!has(k)) { self[k] = f() }
    return self[k]!
  }

}

public func nowSeconds() -> Double {
  return Double(DispatchTime.now().uptimeNanoseconds) / 1e9
}

public class Timer {

  var startTime: Double = nowSeconds()

  public init() {}

  public func time() -> Double {
    return nowSeconds() - startTime
  }

  public func reset() -> Void {
    startTime = nowSeconds()
  }

  public func lap() -> Double {
    let _time = time()
    reset()
    return _time
  }

}

public func timed<X>(_ f: () throws -> X) rethrows -> (x: X, time: Double) {
  let timer = Timer()
  let x = try f()
  return (x: x, time: timer.time())
}

@discardableResult
public func printTime<X>(_ label: String? = nil, format: String = "[%.3fs]", _ f: () throws -> X) rethrows -> X {
  let (x: x, time: time) = try timed(f)
  print(String(format: format + (label != nil ? " \(label as Any)" : ""), time))
  return x
}

public func timeit<X>(_ n: Int = 3, _ label: String? = nil, format: String = "[%.3fs]", _ f: () throws -> X) rethrows {
  for _ in 0..<n {
    try printTime(label, format: format, f)
  }
}
