// (See Birdgram/Search.swift)

import Foundation

import Bubo // Before Bubo_Pods imports
import AudioKit // For AKAudioFile
import Promises
import Surge

@objc(RNSearch)
class RNSearch: RCTEventEmitter, RNProxy {

  typealias Proxy = SearchBirdgram
  var proxy: Proxy?

  // (See Birdgram/Spectro.swift)
  @objc static override func requiresMainQueueSetup() -> Bool {
    return false
  }

  @objc override func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return [:]
  }

  @objc open override func supportedEvents() -> [String] {
    return []
  }

  @objc func create(
    _ modelsSearch: Props,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromiseNoProxy(resolve, reject, "create") {
      self.proxy = try SearchBirdgram(
        load: FileProps(fromJs: modelsSearch)
      )
      SearchBirdgram.singleton = self.proxy // TODO(refactor_native_deps) For Spectro.searchBirdgram
    }
  }

  @objc func f_preds(
    _ audioPath: String,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "f_preds") { _proxy -> Array<Float>? in
      return try _proxy.f_preds(
        audioPath
      )
    }
  }

}

// TODO Move these into Bubo/Model.swift, which is blocked on dropping UIImage (ios, no macos) -> CGImage (ios, macos)
//  - This blocks testing (since we can bin/test Bubo but not Birdgram, b/c macos vs. ios)
//  - https://stackoverflow.com/questions/1320988/saving-cgimageref-to-a-png-file
//  - https://stackoverflow.com/questions/43846885/swift-3-cgimage-to-png
//  - In the meantime, do audio->spectro here and spectro->f_preds in Bubo Features

public class SearchBirdgram: Loadable {

  // TODO(refactor_native_deps) For Spectro.searchBirdgram
  public static var singleton: SearchBirdgram? = nil

  // Deps
  public let search:           Search
  public let featuresBirdgram: FeaturesBirdgram

  public convenience required init(load props: FileProps) throws {
    let timer = Timer()
    self.init(
      search: try Search(load: props)
    )
    Log.info(String(format: "SearchBirdgram.init: time[%.3f], props.path[%@]", timer.time(), props.path))
  }

  public init(search: Search) {
    self.search           = search
    self.featuresBirdgram = FeaturesBirdgram(features: search.features)
  }

  // f_preds from audio w/ denoise, to match api.recs.recs_featurize_slice_audio
  //  - Like sqlite search_recs.f_preds_* (from py sg.search_recs, sg.feat_info, api.recs.get_feat_info)
  public func f_preds(
    _ audioPath: String
  ) throws -> Array<Float>? {
    let timer = Timer()

    // Read samples <- audioPath
    let (samples, sampleRate) = try featuresBirdgram.samplesFromAudioPath(audioPath)
    if samples.count == 0 {
      Log.info("SearchBirdgram.f_preds: No samples in audioPath[\(audioPath)]")
      return nil
    }

    // Featurize + predict
    //  - (f_preds <- feat <- agg <- proj <- patches <- spectro <- samples)
    let f_preds = search.f_preds(samples, sample_rate: sampleRate)

    Log.info(String(format: "SearchBirdgram.f_preds: time[%.3f], audioPath[%@]", timer.time(), audioPath))
    return f_preds
  }

}

public class FeaturesBirdgram {

  // Deps
  public let features: Features

  public init(features: Features) {
    self.features = features
  }

  // Fails if input sampleRate != config sample_rate
  // Converts input to 1ch
  public func samplesFromAudioPath(_ audioPath: String) throws -> (
    samples: [Float],
    sampleRate: Int
  ) {
    let file = try AKAudioFile(forReading: URL(fileURLWithPath: audioPath))
    guard let floatChannelData = file.floatChannelData else { throw AppError("Null floatChannelData in file[\(file.url)]") }
    let Samples = Matrix(floatChannelData)
    // Log.debug("Spectro.samplesFromAudioPath: Samples.shape[\(Samples.shape)]") // XXX Debug
    let samples: [Float] = (Samples.shape.0 == 1
      ? Samples.grid               // 1ch -> 1ch
      : Samples.T.map { mean($0) } // 2ch -> 1ch [NOTE Slow-ish: record produces 1ch, but re-rendering xc recs will produce 2ch]
    )
    // Require sampleRate = config sample_rate
    let sampleRate = Int(file.processingFormat.sampleRate)
    if sampleRate != features.sample_rate {
      // Throw i/o crashing (e.g. via downstream precondition in features._spectro)
      throw AppError("sampleRate[\(sampleRate)] must be \(features.sample_rate) (in \(file.url))")
    }
    return (
      samples:    samples,
      sampleRate: sampleRate
    )
  }

}
