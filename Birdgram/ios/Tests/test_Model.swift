// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
import Surge
import SigmaSwiftStatistics
import SwiftNpy
import SwiftyJSON
import Yams
import Bubo

func sig(_ name: String, _ xs: [Float], prec: Int = 3, limit: Int? = 7) {
  print(String(format: "%@ %3d\n%@", name, xs.count, show(xs.slice(to: limit), prec: prec)))
}

func mat(_ name: String, _ X: Matrix<Float>, prec: Int = 3, limit: (Int?, Int?)? = (10, 7)) {
  print(String(format: "%@ %@\n%@", name, String(describing: X.shape), show(X, prec: prec, limit: limit)))
}

//
// Instances (global)
//

let search     = try Search(load: FileProps(path: iosTestsAssetsDir / "search_recs/models/search.json"))
let projection = search.projection
let features   = projection.features

//
// Search.f_preds
//

// WARNING tol
test("Search.f_preds: search_recs.sample(10) start=0 end=10: Compute audio->f_preds") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let audios      = data["audios"].arrayValue.map { $0.arrayValue.map { Float($0.intValue) } }
  let _f_predss   = data["f_predss"].arrayValue.map { $0.arrayValue.map { $0.floatValue } }
  let f_predss    = audios.map { audio in search.f_preds(audio, sample_rate: sample_rate) }
  let limit       = (5, nil) as (Int?, Int?)
  testAlmostEqual(name, Matrix(f_predss).T, Matrix(_f_predss).T, tol: Tol(rel: 1e-3), prec: 4, limit: limit)
  testAlmostEqual(name, Matrix(f_predss).T, Matrix(_f_predss).T, tol: Tol(abs: 1e-5), prec: 4, limit: limit)
  // testAlmostEqual(name, Matrix(f_predss).T, Matrix(_f_predss).T, tol: Tol(),          prec: 4, limit: limit) // XXX Shows diffs
}

// FIXME tol
//  - FIXME(wav_mp4_unstable_preds) wav vs. mp4 causes unstable f_preds
//    - Defer: doesn't look like a showstopper, revisit later to improve app quality
//    - (See .ipynb for details)
// test("Search.f_preds: search_recs.sample(10) start=0 end=10: Read search_recs.f_preds") { name, data in
//   let sample_rate = data["sample_rate"].intValue
//   let audios      = data["audios"].arrayValue.map { $0.arrayValue.map { Float($0.intValue) } }
//   let _f_predss   = data["f_predss"].arrayValue.map { $0.arrayValue.map { $0.floatValue } }
//   let f_predss    = audios.map { audio in search.f_preds(audio, sample_rate: sample_rate) }
//   let limit       = (5, nil) as (Int?, Int?)
//   // FIXME(model_predict): These don't always pass! Current random_state requires rel:0.999, abs:0.05
//   testAlmostEqual(name, Matrix(f_predss).T, Matrix(_f_predss).T, tol: Tol(rel: 0.80), prec: 4, limit: limit) // FIXME(model_predict)
//   testAlmostEqual(name, Matrix(f_predss).T, Matrix(_f_predss).T, tol: Tol(abs: 0.02), prec: 4, limit: limit) // FIXME(model_predict)
//   testAlmostEqual(name, Matrix(f_predss).T, Matrix(_f_predss).T, tol: Tol(),          prec: 4, limit: limit) // XXX Shows diffs
// }

// exit(1) // XXX Debug

//
// Search.species_proba
//

test("Search.species_proba: search_recs.sample(10) start=0 end=10") { name, data in
  let feats     = data["feats"].arrayValue.map { $0.arrayValue.map { $0.floatValue } }
  let _f_predss = data["f_predss"].arrayValue.map { $0.arrayValue.map { $0.floatValue } }
  let f_predss  = search.species_proba(feats)
  testAlmostEqual(name, Matrix(f_predss), Matrix(_f_predss), limit: (10, 10))
}

//
// Projection._feat/_agg
//

test("Projection._feat/_agg: XC416346 start=0 end=10") { name, data in
  let proj  = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _feat = data["feat"].arrayValue.map { $0.floatValue }
  let agg   = projection._agg(proj)
  let feat  = projection._feat(agg)
  testAlmostEqual(name, feat, _feat, limit: 15)
}

test("Projection._feat/_agg: small example") { name, data in
  let proj  = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _feat = data["feat"].arrayValue.map { $0.floatValue }
  let agg   = projection._agg(proj)
  let feat  = projection._feat(agg)
  testAlmostEqual(name, feat, _feat, tol: Tol(abs: 1e-7), limit: 15)
  testAlmostEqual(name, feat, _feat, tol: Tol(rel: 1e-4), limit: 15)
}

//
// Projection._proj
//

// FIXME? tol
test("Projection._proj: XC416346 start=0 end=10") { name, data in
  let patches = Matrix(data["patches"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _proj   = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let proj    = projection._proj(patches)
  testAlmostEqual(name, proj, _proj, tol: Tol(abs: 1e-5), limit: (10, 10))
  testAlmostEqual(name, proj, _proj, tol: Tol(rel: 0.50), limit: (10, 10)) // FIXME? Or just an edge case of np.almost_equal?
}

// WARNING tol
test("Projection._proj: small example") { name, data in
  let patches = Matrix(data["patches"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _proj   = Matrix(data["proj"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let proj    = projection._proj(patches)
  testAlmostEqual(name, proj, _proj, tol: Tol(abs: 1e-6), limit: (10, 10))
  testAlmostEqual(name, proj, _proj, tol: Tol(rel: 1e-2), limit: (10, 10))
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
  testAlmostEqual(name, patches, _patches)
}

test("Features._patches: small example") { name, data in
  let f           = data["f"].arrayValue.map { $0.floatValue }
  let t           = data["t"].arrayValue.map { $0.floatValue }
  let S           = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let _patches    = Matrix(data["patches"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let spectro     = (f: f, t: t, S: S)
  let patches     = features._patches(spectro)
  testAlmostEqual(name, patches, _patches)
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
  testAlmostEqual(name, S, _S, tol: Tol(abs: 1e-3))
  testAlmostEqual(name, S, _S, tol: Tol(rel: 1e-3))
}

// WARNING tol
test("Features._spectro: generate audio from hann window (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let audio       = floor(scipy.signal.hann(512 * 3).slice(to: -100) * 1000) // floor to match .astype(np.int16)
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: Tol(abs: 1e-4))
  testAlmostEqual(name, S, _S, tol: Tol(rel: 1e-1))
}

// WARNING tol
test("Features._spectro: XC416346 start=1.15 end=1.35 (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S)
}

// WARNING tol
test("Features._spectro: XC416346 start=1.15 end=1.35 (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: Tol(abs: 1e-5))
  testAlmostEqual(name, S, _S, tol: Tol(rel: 1e-2))
}

// WARNING tol
test("Features._spectro: XC416346 start=0 end=10 (denoise=false)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: Tol(rel: 1e-4))
}

// WARNING tol
test("Features._spectro: XC416346 start=0 end=10 (denoise=true)") { name, data in
  let sample_rate = data["sample_rate"].intValue
  let f_bins      = data["f_bins"].intValue
  let denoise     = data["denoise"].boolValue
  let audio       = data["audio"].arrayValue.map { $0.floatValue }
  let _S          = Matrix(data["S"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let S           = features._spectro(audio, sample_rate: sample_rate, f_bins: f_bins, denoise: denoise).S
  testAlmostEqual(name, S, _S, tol: Tol(abs: 1e-5))
}

//
// Features._spectro_denoise
//

test("Features._spectro_denoise") { name, data in
  let X = Matrix(data["X"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  let Y = Matrix(data["Y"].arrayValue.map { $0.arrayValue.map { $0.floatValue } })
  testAlmostEqual(name, features._spectro_denoise(X), Y, tol: Tol(abs: 1e-7))
}
