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
  let timer = Timer()
  var data: JSON?
  do {
    data = try Json.loadFromPath(iosTestsDataDir / name + ".json")
    let time = timer.time()
    if time > 0.5 { print(String(format: "SLOW  Reading data file took %.3fs: %@", time, name)) }
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
  _ _name: String,
  _ x: X,
  _ y: X,
  tol: Tol? = nil, // For print() only (used by callers for comparison, only reported here)
  with: (X, X) -> Bool,
  show: (X) -> Any = { x in x },
  showAfter: (() -> Any)? = nil
) {
  var name = _name
  if let tol = tol { name = "[\(tol)] \(name)" }
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
    if let showAfter = showAfter {
      print(red("\(showAfter())"))
    }
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
  tol: Tol = Tol(),
  equal_nan: Bool = true
) {
  testEqual(name, x, y,
    tol:  tol,
    with: { x, y in np.almost_equal([x], [y], tol: tol, equal_nan: equal_nan) }
  )
}

public func testAlmostEqual(
  _ name: String,
  _ xs: [Float],
  _ ys: [Float],
  tol: Tol = Tol(),
  equal_nan: Bool = true,
  // For show(), not for comparison
  prec: Int = 3,
  limit: Int? = 7
) {
  testEqual(name, xs, ys,
    tol:  tol,
    with: { xs, ys in np.almost_equal(xs, ys, tol: tol, equal_nan: equal_nan) },
    show: { xs in
      var _xs = xs
      if let limit = limit { _xs = _xs.slice(to: limit) }
      return "\(xs.count): \(show(_xs, prec: prec))"
    }
    // TODO showAfter (like Matrix)
  )
}

public func testAlmostEqual(
  _ name: String,
  _ X: Matrix<Float>,
  _ Y: Matrix<Float>,
  tol: Tol = Tol(),
  equal_nan: Bool = true,
  // For show(), not for comparison
  prec: Int = 3,
  limit: (Int?, Int?)? = (10, 7)
) {
  testEqual(name, X, Y,
    tol:  tol,
    with: { X, Y in X.shape == Y.shape && np.almost_equal(X.grid, Y.grid, tol: tol, equal_nan: equal_nan) },
    show: { X in
      return [
        "\(X.shape)",
        "\(show(X, prec: prec, limit: limit))",
      ].joined(separator: "\n")
    },
    showAfter: {
      let A  = np.abs(X - Y)
      let Aq = Stats.quantiles(A.grid, bins: 4)
      let R  = np.abs(X .- Y) ./ np.maximum(np.abs(X), np.abs(Y))
      let Rq = Stats.quantiles(R.grid, bins: 4)
      return [
        "  abs(x-y):",
        "  quantiles: \(show(Aq, prec: 10))",
        "\(show(A, prec: prec, limit: limit))",
        "  abs(x-y)/max(abs(x),abs(y)):",
        "  quantiles: \(show(Rq, prec: 10))",
        "\(show(R, prec: prec, limit: limit))",
      ].joined(separator: "\n")
    }
  )
}

//
// Colors (via ansi)
//

public func green (_ s: String) -> String { return "\u{001b}[32m\(s)\u{001b}[m" }
public func red   (_ s: String) -> String { return "\u{001b}[31m\(s)\u{001b}[m" }
