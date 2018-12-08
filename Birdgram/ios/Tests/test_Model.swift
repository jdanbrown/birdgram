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
// Instances (global)
//

let search     = try Search(load: FileProps(path: iosTestsAssetsDir / "search_recs/models/search.json"))
let projection = search.projection
let features   = projection.features

//
// Search.preds
//

// TODO(model_predict)

//
// Search.species_proba
//

// TODO(model_predict)

//
// Projection._feat/_agg
//

test("Projection._feat/_agg: XC416346 start=0 end=10") { name, data in
  let proj  = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _feat = data["feat"].arrayValue.map { $0.floatValue }
  let agg   = projection._agg(proj)
  let feat  = projection._feat(agg)
  testAlmostEqual(name, feat, _feat, tol: 1e-6, showLimit: 15)
}

test("Projection._feat/_agg: small example") { name, data in
  let proj  = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _feat = data["feat"].arrayValue.map { $0.floatValue }
  let agg   = projection._agg(proj)
  let feat  = projection._feat(agg)
  testAlmostEqual(name, feat, _feat, tol: 1e-6, showLimit: 15)
}

//
// Projection._proj
//

// WARNING tol
test("Projection._proj: XC416346 start=0 end=10") { name, data in
  let patches = Matrix(data["patches"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _proj   = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let proj    = projection._proj(patches)
  testAlmostEqual(name, proj, _proj, tol: 1e-5, showLimit: (10, 10))
}

test("Projection._proj: small example") { name, data in
  let patches = Matrix(data["patches"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _proj   = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let proj    = projection._proj(patches)
  testAlmostEqual(name, proj, _proj, tol: 1e-6, showLimit: (10, 10))
}

//
// Features._patches
//

test("Features._patches: XC416346 start=0 end=10") { name, data in
  let f           = data["f"].arrayValue.map { $0.floatValue }
  let t           = data["t"].arrayValue.map { $0.floatValue }
  let S           = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _patches    = Matrix(data["patches"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let spectro     = (f: f, t: t, S: S)
  let patches     = features._patches(spectro)
  testAlmostEqual(name, patches, _patches, tol: 1e-7)
}

test("Features._patches: small example") { name, data in
  let f           = data["f"].arrayValue.map { $0.floatValue }
  let t           = data["t"].arrayValue.map { $0.floatValue }
  let S           = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _patches    = Matrix(data["patches"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let spectro     = (f: f, t: t, S: S)
  let patches     = features._patches(spectro)
  testAlmostEqual(name, patches, _patches, tol: 1e-7)
}

//
// Features._spectro
//

// WARNING tol
test("Features._spectro: generate audio from hann window (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let audio       = floor(scipy.signal.hann(512 * 3).slice(to: -100) * 1000) // floor to match .astype(np.int16)
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features._spectro: generate audio from hann window (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let audio       = floor(scipy.signal.hann(512 * 3).slice(to: -100) * 1000) // floor to match .astype(np.int16)
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-4)
}

// WARNING tol
test("Features._spectro: XC416346 start=1.15 end=1.35 (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features._spectro: XC416346 start=1.15 end=1.35 (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features._spectro: XC416346 start=0 end=10 (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-3)
}

// WARNING tol
test("Features._spectro: XC416346 start=0 end=10 (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: 1e-4)
}

//
// Features._spectro_denoise
//

test("Features._spectro_denoise") { name, data in
  let X = Matrix(data["X"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let Y = Matrix(data["Y"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  testAlmostEqual(name, features._spectro_denoise(X), Y, tol: 1e-6)
}
