import Foundation
import Surge
import SwiftNpy

// TODO Write tests
/*  - Scratch:

# py
x = np.random.rand(5) * 100         # This works with both Npy and Npz
x = np.random.rand(5_000_000) * 100 # This works with Npy but breaks Npz (see below)
X = np.random.rand(3,5)
np.save('/tmp/array.npy',  x)
np.save('/tmp/matrix.npy', X)
np.savez('/tmp/z.npz', x=x, X=X)
np.savez_compressed('/tmp/z_compressed.npz', x=x, X=X)
display(
    np.load('/tmp/array.npy'),
    np.load('/tmp/matrix.npy'),
    dict(np.load('/tmp/z.npz')),
    dict(np.load('/tmp/z_compressed.npz')),
)

// swift
print((try np.load("/tmp/array.npy") as [Double]).slice(to: 10))
print(try np.load("/tmp/matrix.npy") as Matrix<Double>)
let npz = try np.load("/tmp/z.npz") as Npz
print(npz["x"]?.array() as [Double]? as Any)
print(npz["X"]?.matrix() as Matrix<Double>? as Any)
print("z.npz\n  ",           (try np.load("/tmp/z.npz")            ["x"]!.array() as [Double]).slice(to: 10))
print("z_compresed.npz\n  ", (try np.load("/tmp/z_compressed.npz") ["x"]!.array() as [Double]).slice(to: 10))

*/

// XXX Avoid: I encountered weird failures when loading an .npz with a 5M elem array, but it works fine with Npy
//  - "Fatal error: Failed `unzGoToFirstFile`: file .../SwiftZip/SwiftZip.swift, line 25"
//  - Workaround: store separate .npy files, avoid .npz for now
extension Npz {

  public init(path: String) throws {
    try self.init(contentsOf: URL(fileURLWithPath: path))
  }

}

extension Npy {

  public init(path: String) throws {
    try self.init(contentsOf: URL(fileURLWithPath: path))
  }

  // TODO How to deal with numeric types more generically? Lots of boilerplate here

  public func array1() -> [Int]    { let _ = shape1(); return elementsCast() }
  public func array1() -> [Float]  { let _ = shape1(); return elementsCast() }
  public func array1() -> [Double] { let _ = shape1(); return elementsCast() }

  // public func array2() -> [[Int]] // TODO Can't reuse matrix()
  public func array2() -> [[Float]]  { let _ = shape2(); return (matrix() as Matrix<Float>).map  { Array($0) } }
  public func array2() -> [[Double]] { let _ = shape2(); return (matrix() as Matrix<Double>).map { Array($0) } }

  // public func matrix() -> Matrix<Int> // No Matrix<Int>
  public func matrix() -> Matrix<Float>  { let (r, c) = shape2(); return Matrix(rows: r, columns: c, grid: elementsCast()) }
  public func matrix() -> Matrix<Double> { let (r, c) = shape2(); return Matrix(rows: r, columns: c, grid: elementsCast()) }

  public func shape1() -> Int {
    precondition(shape.count == 1, "Expected 1-dim shape, got: \(shape)")
    return shape[0]
  }

  public func shape2() -> (Int, Int) {
    precondition(shape.count == 2, "Expected 2-dim shape, got: \(shape)")
    return (shape[0], shape[1])
  }

  // Like elements(), but cast i/o crashing if the requested type doesn't match the stored type
  public func elementsCast() -> [Int] {
    switch dataType {
      case .bool:    return elements(Bool.self)   .map { Int($0 ? 1 : 0) }
      case .uint8:   return elements(UInt8.self)  .map { Int($0) }
      case .uint16:  return elements(UInt16.self) .map { Int($0) }
      case .uint32:  return elements(UInt32.self) .map { Int($0) }
      case .uint64:  return elements(UInt64.self) .map { Int($0) }
      case .int8:    return elements(Int8.self)   .map { Int($0) }
      case .int16:   return elements(Int16.self)  .map { Int($0) }
      case .int32:   return elements(Int32.self)  .map { Int($0) }
      case .int64:   return elements(Int64.self)  .map { Int($0) }
      case .float32: return elements(Float.self)  .map { Int($0) }
      case .float64: return elements(Double.self) .map { Int($0) }
    }
  }

  // Like elements(), but cast i/o crashing if the requested type doesn't match the stored type
  public func elementsCast() -> [Float] {
    switch dataType {
      case .bool:    return elements(Bool.self)   .map { Float($0 ? 1 : 0) }
      case .uint8:   return elements(UInt8.self)  .map { Float($0) }
      case .uint16:  return elements(UInt16.self) .map { Float($0) }
      case .uint32:  return elements(UInt32.self) .map { Float($0) }
      case .uint64:  return elements(UInt64.self) .map { Float($0) }
      case .int8:    return elements(Int8.self)   .map { Float($0) }
      case .int16:   return elements(Int16.self)  .map { Float($0) }
      case .int32:   return elements(Int32.self)  .map { Float($0) }
      case .int64:   return elements(Int64.self)  .map { Float($0) }
      case .float32: return elements(Float.self)
      case .float64: return elements(Double.self) .map { Float($0) }
    }
  }

  // Like elements(), but cast i/o crashing if the requested type doesn't match the stored type
  public func elementsCast() -> [Double] {
    switch dataType {
      case .bool:    return elements(Bool.self)   .map { Double($0 ? 1 : 0) }
      case .uint8:   return elements(UInt8.self)  .map { Double($0) }
      case .uint16:  return elements(UInt16.self) .map { Double($0) }
      case .uint32:  return elements(UInt32.self) .map { Double($0) }
      case .uint64:  return elements(UInt64.self) .map { Double($0) }
      case .int8:    return elements(Int8.self)   .map { Double($0) }
      case .int16:   return elements(Int16.self)  .map { Double($0) }
      case .int32:   return elements(Int32.self)  .map { Double($0) }
      case .int64:   return elements(Int64.self)  .map { Double($0) }
      case .float32: return elements(Float.self)  .map { Double($0) }
      case .float64: return elements(Double.self)
    }
  }

}
