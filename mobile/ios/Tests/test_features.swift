// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
import Surge
import SigmaSwiftStatistics
import SwiftNpy
import SwiftyJSON
import Yams
import Bubo

func sig(_ name: String, _ xs: [Float], limit: Int? = 7) {
  print(String(format: "%@ %3d %@", name, xs.count, show(xs.slice(to: limit), prec: 3)))
}

func mat(_ name: String, _ X: Matrix<Float>) {
  print(String(format: "%@ %@\n%@", name, String(describing: X.shape), show(X, prec: 3)))
}

//
// SpectroLike
//

test("SpectroLike.norm_rms") { name, data in
  let X = Matrix(data["X"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let Y = Matrix(data["Y"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  testAlmostEqual(name, SpectroLike.norm_rms(X), Y)
}

test("SpectroLike.clip_below_median_per_freq") { name, data in
  let X = Matrix(data["X"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let Y = Matrix(data["Y"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  testAlmostEqual("SpectroLike.clip_below_median_per_freq", SpectroLike.clip_below_median_per_freq(X), Y)
}
