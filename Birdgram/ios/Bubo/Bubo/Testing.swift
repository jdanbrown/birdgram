// Utils for the "testing framework"
//  - Add tests in Tests/
//  - Run tests with bin/test

import Foundation
import SwiftyJSON
import Surge

// TODO Turn these into Xcode resources
public let iosDir            = pathDirname(#file) / "../.." // TODO How robust is #file? Which types of builds will break it?
public let iosTestsDataDir   = iosDir / "Tests/data"
public let iosTestsAssetsDir = iosDir / "Tests/assets"

// Test with data
public func test(
  _ name: String,
  _ body: (_ name: String, _ data: JSON) throws -> Void
) { // Don't `rethrows` so callers don't have to `try`
  var data: JSON?
  do {
    data = try Json.loadFromPath(iosTestsDataDir / name + ".json")
  } catch {
    print(red("Data file not found: \(name)"))
    return
  }
  do {
    try body(name, data!)
  } catch {
    print(red("Failed: \(name)"))
    print(red("  error: \(error)"))
    // TODO Stack trace?
  }
}

// Test without data
public func test(
  _ name: String,
  _ body: (_ name: String) throws -> Void
) { // Don't `rethrows` so callers don't have to `try`
  do {
    try body(name)
  } catch {
    print(red("Failed: \(name)"))
    print(red("  error: \(error)"))
    // TODO Stack trace?
  }
}

public func testTrue(_ name: String, _ test: Bool) {
  if test {
    print(green("Passed: \(name)"))
  } else {
    print(red("Failed: \(name)"))
  }
}

public func testEqual<X>(
  _ name: String,
  _ x: X,
  _ y: X,
  with: (X, X) -> Bool,
  show: (X) -> Any = { x in x }
) {
  var countEqual = true
  if let xs = x as? Array<Any>, let ys = y as? Array<Any> { // TODO How to downcast to more general Collection i/o Array?
    countEqual = xs.count == ys.count
  }
  if (
    countEqual && // Else crash when Collection.count's aren't equal
    with(x, y)
  ) {
    print(green("Passed: \(name)"))
  } else {
    print(red("Failed: \(name)"))
    print(red("  x: \(show(x))"))
    print(red("  y: \(show(y))"))
  }
}

public func testEqual<X: Equatable>(
  _ name: String,
  _ x: X,
  _ y: X
) {
  testEqual(name, x, y, with: { x, y in x == y })
}

// Boilerplate for tuples
public func testEqual2<A: Equatable, B: Equatable>(_ name: String,
  _ x: (A,B), _ y: (A,B), with: ((A,B), (A,B)) -> Bool = { x, y in x == y }
) {
  return testEqual(name, x, y, with: with)
}
public func testEqual3<A: Equatable, B: Equatable, C: Equatable>(_ name: String,
  _ x: (A,B,C), _ y: (A,B,C), with: ((A,B,C), (A,B,C)) -> Bool = { x, y in x == y }
) {
  return testEqual(name, x, y, with: with)
}

public func testAlmostEqual(
  _ name: String,
  _ x: Float,
  _ y: Float,
  tol: Float = 1e-7
) {
  testEqual(name, x, y,
    with: { x, y in np.almost_equal([x], [y], tol: tol) }
  )
}

public func testAlmostEqual(
  _ name: String,
  _ xs: [Float],
  _ ys: [Float],
  tol: Float = 1e-7,
  // For show(), not for comparison
  prec: Int = 3,
  showLimit: Int? = nil
) {
  testEqual(name, xs, ys,
    with: { xs, ys in np.almost_equal(xs, ys, tol: tol) },
    show: { xs in
      var _xs = xs
      if let showLimit = showLimit { _xs = _xs.slice(to: showLimit) }
      return "\(xs.count): \(show(_xs, prec: prec))"
    }
  )
}

public func testAlmostEqual(
  _ name: String,
  _ X: Matrix<Float>,
  _ Y: Matrix<Float>,
  tol: Float = 1e-7,
  // For show(), not for comparison
  prec: Int = 3,
  showLimit: (Int?, Int?) = (nil, nil)
) {
  testEqual(name, X, Y,
    with: { X, Y in X.shape == Y.shape && np.almost_equal(X.grid, Y.grid, tol: tol) },
    show: { X in
      var _X = X
      let (rows, columns) = showLimit
      if let rows    = rows    { _X = _X[rows:    0..<rows] }
      if let columns = columns { _X = _X[columns: 0..<columns] }
      return "\(X.shape)\n\(show(_X, prec: prec))"
    }
  )
}

//
// Colors (via ansi)
//

public func green (_ s: String) -> String { return "\u{001b}[32m\(s)\u{001b}[m" }
public func red   (_ s: String) -> String { return "\u{001b}[31m\(s)\u{001b}[m" }
