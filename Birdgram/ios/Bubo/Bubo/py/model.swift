// Like python sp14/model.py (but we drop the "sp14." namespace)

// import AudioKit // FIXME How to build for macos? (works for ios)
import Foundation
import SigmaSwiftStatistics
import Surge

public typealias F_Preds = [Float]                  // n_sp
public typealias Feat    = [Float]                  // k*a
public typealias Agg     = Dictionary<String, Feat> // a -> (k,)
public typealias Proj    = Matrix<Float>            // (k,t)
public typealias Patches = Matrix<Float>            // (f*p,t)

public class Search: Loadable {

  public let projection:  Projection
  public let classes_:    [String]
  public let classifier_: sk.base.Classifier

  public var features: Features { get { return projection.features } }

  public convenience required init(load props: FileProps) throws {
    let timer = Timer()
    self.init(
      projection:  try props.at("projection"),
      classes_:    try props.at("classes_"),
      classifier_: try props.at("classifier_") as (
        // sk.multiclass.OneVsRestClassifier<sk.linear_model.LogisticRegressionOneClass>
        OvRLogReg // Avoid many-small-files bottleneck at mobile startup (see payloads.py for details)
      )
    )
    _Log.info(String(format: "Search.init: time[%.3f], props.path[%@]", timer.time(), props.path))
  }

  public init(
    projection:  Projection,
    classes_:    [String],
    classifier_: sk.base.Classifier
  ) {
    self.projection  = projection
    self.classes_    = classes_
    self.classifier_ = classifier_
  }

  // Like py search_recs.f_preds / sqlite search_recs.f_preds_*
  //  - Full featurize+predict pipeline from audio samples
  //  - denoise=true to match api.recs.recs_featurize_slice_audio
  //  - Like sqlite search_recs.f_preds_* (from py sg.search_recs, sg.feat_info, api.recs.get_feat_info)
  //  - py species_proba is what's used to populate sqlite search_recs.f_preds_* (via sg.search_recs, api.recs.get_feat_info)
  public func f_preds(_ samples: [Float], sample_rate: Int) -> F_Preds? {

    // Debug: perf
    let timer = Timer()
    var debugTimes: Array<(String, Double)> = [] // (Array of tuples b/c Dictionary is ordered by key i/o insertion)
    func debugTimed<X>(_ k: String, _ f: () -> X) -> X { let x = f(); debugTimes.append((k, timer.lap())); return x }

    // Featurize: audio -> spectro
    let spectro = debugTimed("spectro") { features._spectro(samples, sample_rate: sample_rate) }
    if spectro.S.isEmpty {
      _Log.info("Search.f_preds: Empty spectro: samples[\(samples.count)] -> S[\(spectro.S.shape)] (e.g. <nperseg)")
      return nil
    }
    // Featurize: spectro -> feat
    let patches = debugTimed("patches") { features._patches(spectro) }
    let proj    = debugTimed("proj")    { projection._proj(patches) }
    let agg     = debugTimed("agg")     { projection._agg(proj) }
    let feat    = debugTimed("feat")    { projection._feat(agg) }
    // Predict
    let f_preds = debugTimed("f_preds") { species_proba([feat]).only() }

    // Debug: perf
    debugTimes.append(("total", sum(debugTimes.map { $0.1 })))
    _Log.debug(String(format: "Spectro.f_preds: debugTimes: %@",
      debugTimes.map { (k, v) in "\(k)=\(Int(v * 1000))" }.joined(separator: ", ")
    ))

    return f_preds
  }

  public func species_proba(_ feats: [Feat]) -> [F_Preds] {
    let X        = Matrix(feats)         // Rows: samples, columns: features
    let y        = species_proba(X)
    let f_predss = y.map { F_Preds($0) } // Rows: samples, columns: class probs
    precondition(f_predss.count == feats.count, "f_predss.count[\(f_predss.count)] == feats.count[\(feats.count)]")
    return f_predss
  }

  public func species_proba(_ feats: Matrix<Float>) -> Matrix<Float> {
    return classifier_.predict_proba(feats) // Rows: samples, columns: class probs
  }

}

public class Projection: Loadable {

  public let features: Features
  public let agg_funs: [String]
  public let skm_:     sp14.skm.SKM

  public convenience required init(load props: FileProps) throws {
    let timer = Timer()
    self.init(
      features: try props.at("features"),
      agg_funs: try props.at("agg_funs"),
      skm_:     try props.at("skm_")
    )
    _Log.info(String(format: "Projection.init: time[%.3f], props.path[%@]", timer.time(), props.path))
  }

  public init(
    features: Features,
    agg_funs: [String],
    skm_:     sp14.skm.SKM
  ) {
    self.features = features
    self.agg_funs = agg_funs
    self.skm_     = skm_
  }

  public func _feat(_ agg: Agg) -> Feat {
    // Deterministic and unsurprising order for feature vectors: follow the order of agg_config.agg_funs
    return agg_funs.flatMap { k in agg[k]! }
  }

  public func _agg(_ proj: Proj) -> Agg {
    let _agg_funs: Dictionary<String, (Proj) -> Feat> = [
      "mean": { X in X.map { mean($0) } },
      "std":  { X in X.map { std($0) } },
      "max":  { X in X.map { max($0) } },
    ]
    return Dictionary(uniqueKeysWithValues: agg_funs.map { k in
      guard let feat = _agg_funs[k]?(proj) else { preconditionFailure("Unknown agg_fun[\(k)]") }
      return (k, feat)
    })
  }

  public func _proj(_ patches: Patches) -> Proj {
    return skm_.transform(patches)
  }

}

public class Features: Loadable {

  public let sample_rate:  Int
  public let f_bins:       Int
  public let hop_length:   Int
  public let frame_length: Int
  public let frame_window: String
  public let patch_length: Int
  public let nperseg:      Int
  public let overlap:      Double
  public let window:       String
  public let n_mels:       Int
  public let denoise:      Bool

  // State
  let mel_basis_cache: Dictionary<Int, Matrix<Float>>

  public convenience required init(load props: FileProps) throws {
    let timer = Timer()
    self.init(
      sample_rate:  try props.at("sample_rate"),
      f_bins:       try props.at("f_bins"),
      hop_length:   try props.at("hop_length"),
      frame_length: try props.at("frame_length"),
      frame_window: try props.at("frame_window"),
      patch_length: try props.at("patch_length")
    )
    _Log.info(String(format: "Features.init: time[%.3f], props.path[%@]", timer.time(), props.path))
  }

  public init(
    sample_rate:  Int,
    f_bins:       Int,
    hop_length:   Int,
    frame_length: Int,
    frame_window: String,
    patch_length: Int
  ) {

    // Like Features config
    self.sample_rate  = sample_rate
    self.f_bins       = f_bins
    self.hop_length   = hop_length
    self.frame_length = frame_length
    self.frame_window = frame_window
    self.patch_length = patch_length
    // Like Features._spectro_nocache
    self.nperseg      = frame_length
    self.overlap      = 1 - Double(hop_length) / Double(frame_length)
    self.window       = frame_window
    self.n_mels       = f_bins
    self.denoise      = true

    // Extracted as constant to remove bottleneck in spectro()
    //  - TODO Cache mappings for new values of n_mels dynamically
    self.mel_basis_cache = Dictionary(uniqueKeysWithValues: [
      40, // Used in tests (to match py test cases)
      80, // Used in mobile (RecordScreen)
    ].map { [sample_rate, nperseg] n_mels in
      (n_mels, librosa.filters.mel(sample_rate, n_fft: nperseg, n_mels: n_mels))
    })

  }

  public func _patches(_ spectro: Melspectro) -> Patches {
    let p = patch_length
    let (_, _, S) = spectro
    return Matrix(
      (0..<(S.columns - (p - 1))).map { i in
        S[columns: i..<(i+p)].flatten()
      }
    ).T
  }

  public func _spectro(
    _ xs:             [Float],
    sample_rate:      Int, // Don't default, so we can explicitly check sample_rate against caller's expectation
    f_bins  _f_bins:  Int?  = nil,
    denoise _denoise: Bool? = nil
  ) -> Melspectro {
    precondition(sample_rate == self.sample_rate, "Impl hardcoded for sampleRate=\(self.sample_rate), got: \(sample_rate)")
    // let timer = Timer() // XXX Perf

    // Params
    let f_bins  = _f_bins  ?? self.f_bins
    let denoise = _denoise ?? self.denoise
    let n_mels  = f_bins
    guard let mel_basis = mel_basis_cache[n_mels] else {
      preconditionFailure("n_mels[\(n_mels)] not in mel_basis_cache[\(mel_basis_cache.keys)]")
    }

    // Like Melspectro
    // let mels_div      = 2 // Overridden by n_mels
    let scaling       = "spectrum"  // Return units X**2 ('spectrum'), not units X**2/Hz ('density')
    let mode          = "magnitude" // Return |STFT(x)**2|, not STFT(x)**2 (because "humans can't hear complex phase")
    // _Log.debug(String(format: "[time] Features._spectro: config: %d", Int(1000 * timer.lap()))) // XXX Perf

    // STFT(xs)
    //  - TODO Compute fs/ts in scipy.signal.spectrogram
    var (_fs, _ts, S) = scipy.signal.spectrogram(
      xs,
      sample_rate: sample_rate,
      window:      window,
      nperseg:     nperseg,
      noverlap:    Int(overlap * Double(nperseg)),
      scaling:     scaling,
      mode:        mode
    )
    // _Log.debug(String(format: "Features._spectro: scipy.signal.spectrogram: %@", show(["S.shape": S.shape]))) // XXX Debug
    // _Log.debug(String(format: "[time] Features._spectro: scipy.signal.spectrogram: %d", Int(1000 * timer.lap()))) // XXX Perf

    // HACK Apply unknown transforms to match librosa.feature.melspectrogram
    //  - Like Melspectro
    S = Float(nperseg / 2) * S // No leads on this one...
    S = S**2                   // Like energy->power, but spectro already gives us power instead of energy...
    // _Log.debug(String(format: "Features._spectro: unknown transforms: %@", show(["S.shape": S.shape]))) // XXX Debug
    // _Log.debug(String(format: "[time] Features._spectro: unknown transforms: %d", Int(1000 * timer.lap()))) // XXX Perf

    // Linear freq -> mel-scale freq
    //  - Like Melspectro
    S = mel_basis * S
    // _Log.debug(String(format: "Features._spectro: mel_basis: %@", show(["S.shape": S.shape]))) // XXX Debug
    // _Log.debug(String(format: "[time] Features._spectro: mel_basis: %d", Int(1000 * timer.lap()))) // XXX Perf

    // Linear power -> log power
    //  - Like Melspectro
    S = librosa.power_to_db(S)
    // _Log.debug(String(format: "Features._spectro: librosa.power_to_db: %@", show(["S.shape": S.shape]))) // XXX Debug
    // _Log.debug(String(format: "[time] Features._spectro: librosa.power_to_db: %d", Int(1000 * timer.lap()))) // XXX Perf

    // [NOTE fs currently unused in mobile]
    // Mel-scale fs to match S[i]
    //  - Like Melspectro
    //  - TODO Blocked on non-mock fs from scipy.signal.spectrogram
    // fs = librosa.mel_frequencies(n_mels, min(fs), max(fs))

    // Denoise
    //  - Like Features._spectro_denoise
    if denoise {
      S = _spectro_denoise(S)
    }
    // _Log.debug(String(format: "Features._spectro: _spectro_denoise: %@", show(["S.shape": S.shape]))) // XXX Debug
    // _Log.debug(String(format: "[time] Features._spectro: _spectro_denoise: %d", Int(1000 * timer.lap()))) // XXX Perf

    // // XXX Debug
    // _Log.debug(String(format: "Features._spectro: S: %@", [
    //   "xs.count[\(xs.count)]",
    //   "xs[\(xs.slice(to: 10))]",
    //   // "xs.quantiles[\(Stats.quantiles(xs, bins: 5))]", // XXX Slow (sorting)
    //   "S.shape[\(S.shape)]",
    //   "S[\(S.grid.slice(to: 10))]",
    //   // "S.quantiles[\(Stats.quantiles(S.grid, bins: 5))]", // XXX Slow (sorting)
    // ].joined(separator: ", ")))

    return (_fs, _ts, S)

  }

  public func _spectro_denoise(_ _S: Matrix<Float>) -> Matrix<Float> {
    var S = _S
    S = SpectroLike.norm_rms(S)
    S = SpectroLike.clip_below_median_per_freq(S)
    return S
  }

}
