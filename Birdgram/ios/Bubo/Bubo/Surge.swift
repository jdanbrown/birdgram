import Foundation
import Surge

// HACK What's the idiomatic way to do these more generally?
public func toFloats  (_ xs: [Double]) -> [Float]  { return xs.map { Float($0) } }
public func toDoubles (_ xs: [Float])  -> [Double] { return xs.map { Double($0) } }

// TODO Maybe faster with cblas_sgemv? But probably fast enough as is (with cblas_sgemm), since Matrix() doesn't copy.
public func * (xs: Matrix<Float>, ys: [Float]) -> Matrix<Float> {
  return xs * Matrix(rows: ys.count, columns: 1, grid: ys)
}

// TODO Maybe faster with cblas_sgemv? But probably fast enough as is (with cblas_sgemm), since Matrix() doesn't copy.
public func * (xs: [Float], ys: Matrix<Float>) -> Matrix<Float> {
  return Matrix(rows: 1, columns: xs.count, grid: xs) * ys
}

public func / (x: Float, ys: [Float]) -> [Float] {
  // Mimic the O(n) malloc in [Float]/Float [TODO How to vectorize without malloc?]
  return Array<Float>(repeating: x, count: numericCast(ys.count)) ./ ys
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
