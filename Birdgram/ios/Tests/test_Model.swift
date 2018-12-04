// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
import Surge
import SigmaSwiftStatistics
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
  testAlmostEqual(name, SpectroLike.norm_rms(X), Y, tol: 1e-6)
}

test("SpectroLike.clip_below_median_per_freq") { name, data in
  let X = Matrix(data["X"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let Y = Matrix(data["Y"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  testAlmostEqual("SpectroLike.clip_below_median_per_freq", SpectroLike.clip_below_median_per_freq(X), Y, tol: 1e-7)
}

//
// Features config
//

testEqual("Features.f_bins must be 40 for tests to pass", Features.f_bins, 40)

//
// Features._spectro_denoise
//

test("Features._spectro_denoise") { name, data in
  let X = Matrix(data["X"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let Y = Matrix(data["Y"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  testAlmostEqual(name, Features._spectro_denoise(X), Y, tol: 1e-6)
}

//
// Features.spectro
//

// WARNING tol
test("Features.spectro: generate audio from hann window (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let denoise     = data["denoise"].boolValue
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let audio       = floor(scipy.signal.hann(512 * 3).slice(to: -100) * 1000) // floor to match .astype(np.int16)
  let S           = Features.spectro(audio, sample_rate: sample_rate, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features.spectro: generate audio from hann window (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let denoise     = data["denoise"].boolValue
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let audio       = floor(scipy.signal.hann(512 * 3).slice(to: -100) * 1000) // floor to match .astype(np.int16)
  let S           = Features.spectro(audio, sample_rate: sample_rate, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-4)
}

// WARNING tol
test("Features.spectro: XC415272 start=2.05 end=2.25 (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = Features.spectro(audio, sample_rate: sample_rate, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features.spectro: XC415272 start=2.05 end=2.25 (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = Features.spectro(audio, sample_rate: sample_rate, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features.spectro: XC415272 start=0 end=10 (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = Features.spectro(audio, sample_rate: sample_rate, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features.spectro: XC415272 start=0 end=10 (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = Features.spectro(audio, sample_rate: sample_rate, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-4)
}
