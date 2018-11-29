import Foundation
import Surge

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
