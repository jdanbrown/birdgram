import Foundation
import Surge

extension Matrix where Scalar == Float {

  // Like Array.slice(from:to:)
  //  - Ignores out-of-range indexes i/o fatal-ing
  public func slice(rows: (from: Int?, to: Int?)?, columns: (from: Int?, to: Int?)?) -> Matrix {
    var X = self
    if let (from, to) = rows    { X = Matrix(Array(X)   .slice(from: from, to: to)) }
    if let (from, to) = columns { X = Matrix(Array(X.T) .slice(from: from, to: to)).T }
    return X
  }

}

extension Matrix where Scalar == Double {

  // Like Array.slice(from:to:)
  //  - Ignores out-of-range indexes i/o fatal-ing
  public func slice(rows: (from: Int?, to: Int?)?, columns: (from: Int?, to: Int?)?) -> Matrix {
    var X = self
    if let (from, to) = rows    { X = Matrix(Array(X)   .slice(from: from, to: to)) }
    if let (from, to) = columns { X = Matrix(Array(X.T) .slice(from: from, to: to)).T }
    return X
  }

}

// HACK How to do these more idiomatically / generically?
public func toInts    (_ xs: [Double])       -> [Int]          { return xs.map  { Int($0) } }
public func toFloats  (_ xs: [Double])       -> [Float]        { return xs.map  { Float($0) } }
public func toInts    (_ xs: [Float])        -> [Int]          { return xs.map  { Int($0) } }
public func toDoubles (_ xs: [Float])        -> [Double]       { return xs.map  { Double($0) } }
public func toInts    (_ xs: [[Double]])     -> [[Int]]        { return xs.map  { toInts($0) } }
public func toFloats  (_ xs: [[Double]])     -> [[Float]]      { return xs.map  { toFloats($0) } }
public func toInts    (_ xs: [[Float]])      -> [[Int]]        { return xs.map  { toInts($0) } }
public func toDoubles (_ xs: [[Float]])      -> [[Double]]     { return xs.map  { toDoubles($0) } }
public func toFloats  (_ xs: Matrix<Double>) -> Matrix<Float>  { return xs.vect { toFloats($0) } }
public func toDoubles (_ xs: Matrix<Float>)  -> Matrix<Double> { return xs.vect { toDoubles($0) } }

// TODO Maybe faster with cblas_sgemv? But probably fast enough as is (with cblas_sgemm), since Matrix() doesn't copy.
public func * (X: Matrix<Float>, ys: [Float]) -> [Float] {
  let Z  = X * Matrix(rows: ys.count, columns: 1, grid: ys)
  let zs = Z.grid
  precondition(Z.shape == (X.rows, 1))
  precondition(zs.count == X.rows)
  return zs
}
public func * (xs: [Float], Y: Matrix<Float>) -> [Float] {
  let Z  = Matrix(rows: 1, columns: xs.count, grid: xs) * Y
  let zs = Z.grid
  precondition(Z.shape == (1, Y.columns))
  precondition(zs.count == Y.columns)
  return zs
}

public prefix func - (xs: [Float]) -> [Float] { return neg(xs) }

// Mimic the O(n) malloc in `[Float] op Float` [TODO How to vectorize without malloc?]
public func + (x: Float, ys: [Float]) -> [Float] { return Array<Float>(repeating: x, count: numericCast(ys.count)) .+ ys }
public func - (x: Float, ys: [Float]) -> [Float] { return Array<Float>(repeating: x, count: numericCast(ys.count)) .- ys }
public func * (x: Float, ys: [Float]) -> [Float] { return Array<Float>(repeating: x, count: numericCast(ys.count)) .* ys }
public func / (x: Float, ys: [Float]) -> [Float] { return Array<Float>(repeating: x, count: numericCast(ys.count)) ./ ys }

// Elem-wise unary operations for Matrix, like for Array
public func abs(_ X: Matrix<Float>) -> Matrix<Float> { return X.vect { abs($0) } }

// Elem-wise binary operations for Matrix, like for Array
public func .+ (X: Matrix<Float>, Y: Matrix<Float>) -> Matrix<Float> { return elem_op_vect(X, Y, .+) }
public func .- (X: Matrix<Float>, Y: Matrix<Float>) -> Matrix<Float> { return elem_op_vect(X, Y, .-) }
public func .* (X: Matrix<Float>, Y: Matrix<Float>) -> Matrix<Float> { return elem_op_vect(X, Y, .*) }
public func ./ (X: Matrix<Float>, Y: Matrix<Float>) -> Matrix<Float> { return elem_op_vect(X, Y, ./) }

public func elem_op_vect(_ X: Matrix<Float>, _ Y: Matrix<Float>, _ f: ([Float], [Float]) -> [Float]) -> Matrix<Float> {
  precondition(X.shape == Y.shape, "Shapes must match: X[\(X.shape)] != Y[\(Y.shape)]")
  return Matrix(
    rows:    X.rows,
    columns: X.columns,
    grid:    f(X.grid, Y.grid)
  )
}

public func elem_op(_ X: Matrix<Float>, _ Y: Matrix<Float>, _ f: (Float, Float) -> Float) -> Matrix<Float> {
  return elem_op_vect(X, Y) { xs, ys in zip(xs, ys).map { x, y in f(x, y) } }
}

//
// In-place operations
//

// XXX Started and ended up not needing. Keeping for reference.

// public func *= (X: inout Matrix<Float>, Y: Matrix<Float>) {
//   lhs.withUnsafeMutableMemory { lm in
//     var scalar = rhs
//     vDSP_vsadd(lm.pointer, numericCast(lm.stride), &scalar, lm.pointer, numericCast(lm.stride), numericCast(lm.count))
//   }
// }

// public func *= (X: inout Matrix<Float>, Y: Matrix<Float>) {
//   precondition(X.columns == Y.rows, "Matrix dimensions not compatible with multiplication")
//   var Z = Matrix<Float>(rows: X.rows, columns: Y.columns, repeatedValue: 0.0)
//   if Z.rows > 0 && Z.columns > 0 { // Avoid https://github.com/mattt/Surge/issues/92
//     if X.columns > 0 { // HACK Avoid crash, mimic numpy (nonempty zero matrix) [TODO github issue]
//       Z.grid.withUnsafeMutableBufferPointer { pointer in
//         cblas_sgemm(
//           CblasRowMajor, CblasNoTrans, CblasNoTrans,
//           Int32(X.rows), Int32(Y.columns), Int32(X.columns),
//           1.0,
//           X.grid, Int32(X.columns),
//           Y.grid, Int32(Y.columns),
//           0.0,
//           pointer.baseAddress!, Int32(Y.columns)
//         )
//       }
//     }
//   }
//   return Z
// }

//
// abs
//

// abs([Complex]) -> [Real]
public func abs(_ reals: inout [Float], _ imags: inout [Float]) -> [Float] {
  precondition(reals.count == imags.count) // DSPSplitComplex() doesn't enforce this
  let n  = reals.count
  var zs = DSPSplitComplex(realp: &reals, imagp: &imags)
  var xs = [Float](repeating: .nan, count: numericCast(n))
  xs.withUnsafeMutableBufferPointer { xsp in
    vDSP_zvabs(&zs, 1, xsp.baseAddress!, 1, numericCast(n))
  }
  return xs
}

//
// pow operator **: nonstandard operator to match numpy usage
//

precedencegroup ExponentiationPrecedence {
  associativity: right
  higherThan:    MultiplicationPrecedence
}
infix operator **: ExponentiationPrecedence

public func ** (x: Float, y: Float) -> Float {
  return powf(x, y)
}

public func ** (xs: [Float], y: Float) -> [Float] {
  return pow(xs, y)
}

public func ** (xs: Matrix<Float>, y: Float) -> Matrix<Float> {
  return pow(xs, y)
}
