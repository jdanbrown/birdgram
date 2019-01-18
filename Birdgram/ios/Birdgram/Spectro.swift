// To expose a method from swift to js:
//  - app/native/Spectro.ts      - add js Spectro.f() calling objc NativeModules.RNSpectro.f()
//  - ios/Birdgram/Spectro.m     - add objc extern for swift RNSpectro.f()
//  - ios/Birdgram/Spectro.swift - add swift RNSpectro.f() calling Spectro.f()
//  - ios/Birdgram/Spectro.swift - add swift Spectro.f()

import os // For os_log
import Foundation

import Bubo // Before Bubo_Pods imports
import AudioKit
import Promises
import Surge
import SwiftyJSON

// Docs
//  - https://facebook.github.io/react-native/docs/native-modules-ios
//  - https://facebook.github.io/react-native/docs/communication-ios
//  - react-native/React/Base/RCTBridgeModule.h -- many code comments not covered in the docs
//
// Examples
//  - https://github.com/carsonmcdonald/AVSExample-Swift/blob/master/AVSExample/SimplePCMRecorder.swift
//
// NOTE
//  - Returning values from methods requires callbacks/promises (prefer promises)
//    - https://facebook.github.io/react-native/docs/native-modules-ios#promises
//  - Can't map `throws` to js, must reject() via promise (or error() via callback)
//    - Like: can't map `return` to js, must resolve() via promise (or success() via callback)
//  - @objc for functions with args(/ only if they return a promise?) requires `_ foo` on first arg
//    - https://stackoverflow.com/a/39840952/397334

@objc(RNSpectro)
class RNSpectro: RCTEventEmitter, RNProxy {

  typealias Proxy = Spectro
  var proxy: Proxy?

  // requiresMainQueueSetup / methodQueue / dispatch_async
  //  - https://stackoverflow.com/a/51014267/397334
  //  - https://facebook.github.io/react-native/docs/native-modules-ios#threading
  //  - QUESTION Should we avoid blocking the main queue on long spectro operations?
  @objc static override func requiresMainQueueSetup() -> Bool {
    return false
  }

  // Static constants exported to js once at init time (e.g. later changes will be ignored)
  //  - https://facebook.github.io/react-native/docs/native-modules-ios#exporting-constants
  @objc override func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return [:]
  }

  @objc open override func supportedEvents() -> [String] {
    return [
      "audioChunk",
      "spectroFilePath",
    ]
  }

  // XXX Debug
  @objc func debugPrintNative(_ msg: String) {
    os_log("%@", msg) // (print() doesn't show up in device logs, even though it somehow shows up in xcode logs)
  }

  @objc func create(
    _ opts: Props,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromiseNoProxy(resolve, reject, "create") {
      self.proxy = try Spectro.create(
        emitter:          self,
        f_bins:           propsGetRequired(opts, "f_bins"),
        // TODO Clean up unused params
        sampleRate:       propsGetOptional(opts, "sampleRate"),
        channels:         propsGetOptional(opts, "channels"),
        bytesPerPacket:   propsGetOptional(opts, "bytesPerPacket"),
        framesPerPacket:  propsGetOptional(opts, "framesPerPacket"),
        bytesPerFrame:    propsGetOptional(opts, "bytesPerFrame"),
        channelsPerFrame: propsGetOptional(opts, "channelsPerFrame"),
        bitsPerChannel:   propsGetOptional(opts, "bitsPerChannel")
      )
    }
  }

  @objc func start(
    _ opts: Props,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "start") { _proxy in
      try _proxy.start(
        outputPath:  propsGetRequired(opts, "outputPath"),
        refreshRate: propsGetRequired(opts, "refreshRate")
      )
    }
  }

  @objc func stop(
    _ resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromiseAsync(resolve, reject, "stop") { _proxy in _proxy.stop() }
  }

  @objc func stats(
    _ resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "stats") { _proxy in _proxy.stats() }
  }

  @objc func renderAudioPathToSpectroPath(
    _ audioPath: String,
    spectroPath: String,
    opts: Props,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "renderAudioPathToSpectroPath") { _proxy -> Props? in
      guard let imageFile = try _proxy.renderAudioPathToSpectroPath(
        audioPath,
        spectroPath,
        f_bins: propsGetRequired(opts, "f_bins"),
        denoise: propsGetRequired(opts, "denoise")
      ) else {
        return nil
      }
      return [
        "path":   imageFile.path,
        "width":  imageFile.width,
        "height": imageFile.height,
      ]
    }
  }

  @objc func editAudioPathToAudioPath(
    _ props: Props,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "editAudioPathToAudioPath") { _proxy -> Void in
      try _proxy.editAudioPathToAudioPath(
        parentAudioPath: propsGetRequired(props, "parentAudioPath"),
        editAudioPath:   propsGetRequired(props, "editAudioPath"),
        draftEdit:       DraftEdit.fromJson(Json.loads(propsGetRequired(props, "draftEdit")))
      )
    }
  }

  @objc func chunkImageFile(
    _ path: String,
    chunkWidth: Int,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromiseNoProxy(resolve, reject, "chunkImageFile") { () -> Array<Props> in
      return (try Birdgram.chunkImageFile(
        path,
        chunkWidth: chunkWidth
      )).map { imageFile in [
        "path":   imageFile.path,
        "width":  imageFile.width,
        "height": imageFile.height,
      ]}
    }
  }

}

// Leaned heavily on these very simple and clear examples to make this thing work
//  - https://github.com/carsonmcdonald/AVSExample-Swift/blob/master/AVSExample/SimplePCMRecorder.swift
//  - https://github.com/goodatlas/react-native-audio-record/blob/master/ios/RNAudioRecord.m
//  - https://github.com/chadsmith/react-native-microphone-stream/blob/master/ios/MicrophoneStream.m
//  - https://github.com/rochars/wavefile
//    - e.g. a-law:  https://github.com/rochars/wavefile/blob/846f66c/dist/wavefile.js#L2456
//    - e.g. mu-law: https://github.com/rochars/wavefile/blob/846f66c/dist/wavefile.js#L2490
public class Spectro {

  static func create(
    emitter:          RCTEventEmitter,
    f_bins:           Int,
    // TODO Clean up unused params
    sampleRate:       Double?,
    channels:         UInt32?,
    bytesPerPacket:   UInt32?,
    framesPerPacket:  UInt32?,
    bytesPerFrame:    UInt32?,
    channelsPerFrame: UInt32?,
    bitsPerChannel:   UInt32?
  ) throws -> Spectro {
    Log.info("Spectro.create")
    let mSampleRate       = sampleRate       ?? 44100
    let mBitsPerChannel   = bitsPerChannel   ?? 16
    let mChannelsPerFrame = channelsPerFrame ?? channels ?? 2
    let mBytesPerPacket   = bytesPerPacket   ?? (mBitsPerChannel / 8 * mChannelsPerFrame)
    let mBytesPerFrame    = bytesPerFrame    ?? mBytesPerPacket // Default assumes PCM
    let mFramesPerPacket  = framesPerPacket  ?? 1 // 1 for uncompressed
    let mFormatFlags      = (
      // TODO Understand this. Was crashing without it. Blindly copied from RNAudioRecord.m (react-native-audio-record)
      mBitsPerChannel == 8
        ? kLinearPCMFormatFlagIsPacked
        : (kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked)
    )
    return try Spectro(
      emitter:           emitter,
      f_bins:            f_bins,
      streamDescription: AudioStreamBasicDescription(
        // TODO Clean up unused params
        // https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
        // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/1572096-audio_data_format_identifiers
        // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/mpeg-4_audio_object_type_constants
        mSampleRate:       mSampleRate,
        // TODO kAudioFormatMPEG4AAC [how to specify bitrate? else just try aac_he_v2]
        //  - Check out examples using AVAudioFile(...).{fileFormat,processingFormat}.formatDescription
        //  - e.g. scratch/read-audio-file.swift
        mFormatID:         kAudioFormatLinearPCM,
        mFormatFlags:      mFormatFlags,
        mBytesPerPacket:   mBytesPerPacket,
        mFramesPerPacket:  mFramesPerPacket,
        mBytesPerFrame:    mBytesPerFrame,
        mChannelsPerFrame: mChannelsPerFrame,
        mBitsPerChannel:   mBitsPerChannel,
        mReserved:         0
      )
    )
  }

  // Deps
  let searchBirdgram:   SearchBirdgram
  var featuresBirdgram: FeaturesBirdgram { get { return searchBirdgram.featuresBirdgram } }
  var features:         Features         { get { return featuresBirdgram.features } }

  // Params
  let emitter:    RCTEventEmitter
  let f_bins:     Int
  let numBuffers: Int
  // - TODO Simplify streamDescription:AudioStreamBasicDescription -> format:AVAudioFormat
  //    - https://developer.apple.com/documentation/avfoundation/avaudioformat -- "this class wraps AudioStreamBasicDescription"
  //    - Simplify our 7 input params to just (sampleRate, channels, bitDepth), consumed via AVAudioFormat(settings:)
  //    - AudioQueueNewInput still needs an AudioStreamBasicDescription, which we can get from AVAudioFormat.streamDescription
  //    - Will need to review the streamDescription fields we rely on for recording (mBytesPerPacket, bitsPerChannel)
  var streamDescription: AudioStreamBasicDescription

  // State
  var outputPath:      String?               = nil
  var queue:           AudioQueueRef?        = nil
  var buffers:         [AudioQueueBufferRef] = []
  var audioFile:       AudioFileID?          = nil
  var nPacketsWritten: UInt32                = 0
  var nPathsSent:      UInt32                = 0
  var samplesBuffer:   [Float]               = [] // Pad audio chunks to ≥nperseg with past audio, else gaps in streaming spectro stft
  var spectroRange:    Interval<Float>       = Spectro.zeroSpectroRange
  static let zeroSpectroRange = Interval.bottom // Unit for union

  // Getters for streamDescription: AVAudioFormat / settings
  //  - General settings: https://developer.apple.com/documentation/avfoundation/avaudioplayer/general_audio_format_settings
  var sampleRate:                NSNumber      { get { return settingsGetRequired(AVSampleRateKey) } }
  var numberOfChannels:          NSNumber      { get { return settingsGetRequired(AVNumberOfChannelsKey) } }
  var formatId:                  AudioFormatID { get { return settingsGetRequired(AVFormatIDKey) } }
  //  - LinearPCM settings: https://tinyurl.com/y957wjur (linear_pcm_format_settings)
  //    - TODO Make these optional for avFormatId != kAudioFormatLinearPCM (e.g. mp4 i/o wav)
  var linearPCMBitDepth:         NSNumber      { get { return settingsGetRequired(AVLinearPCMBitDepthKey) } }
  var linearPCMIsBigEndian:      Bool          { get { return settingsGetRequired(AVLinearPCMIsBigEndianKey) } }
  var linearPCMIsFloat:          Bool          { get { return settingsGetRequired(AVLinearPCMIsFloatKey) } }
  var linearPCMIsNonInterleaved: Bool          { get { return settingsGetRequired(AVLinearPCMIsNonInterleaved) } }
  //  - Caveats for AVAudioFormat<->settings
  //    - https://developer.apple.com/documentation/avfoundation/avaudioformat/1389347-init
  //    - https://developer.apple.com/documentation/avfoundation/avaudioformat/1386904-settings
  //    - https://developer.apple.com/documentation/avfoundation/avaudioformat/1387931-init
  var settings:    [String: Any] { get { return audioFormat.settings }}
  var audioFormat: AVAudioFormat { get {
    guard let audioFormat = AVAudioFormat(
      streamDescription: &streamDescription,
      channelLayout:     nil // nil means use mono/stereo layout for 1/2 channels
    ) else { preconditionFailure("nil AVAudioFormat from streamDescription[\(streamDescription)]") }
    return audioFormat
  }}
  func settingsGetRequired<X>(_ k: String) -> X {
    let v = settings[k]
    guard let x = v as? X else { preconditionFailure("Invalid setting[\(k)]: \(v as Any)") }
    return x
  }

  init(
    emitter:           RCTEventEmitter,
    f_bins:            Int,
    numBuffers:        Int = 3, // ≥3 on iphone? [https://books.google.com/books?id=jiwEcrb_H0EC&pg=PA160]
    streamDescription: AudioStreamBasicDescription
  ) throws {
    Log.info(String(format: "Spectro.init: %@", [
      "f_bins":            f_bins,
      "numBuffers":        numBuffers,
      "streamDescription": streamDescription,
    ]))

    // TODO(refactor_native_deps) Refactor so that all native singletons are created together at App init, so deps can be passed in
    //  - Search is currently created at App init
    //  - Spectro is currently re-created on each startRecording(), and needs Search as a dep
    guard let searchBirdgram = SearchBirdgram.singleton else { throw AppError("Spectro.init: SearchBirdgram.singleton is nil") }
    self.searchBirdgram = searchBirdgram

    self.emitter           = emitter
    self.f_bins            = f_bins
    self.numBuffers        = numBuffers
    self.streamDescription = streamDescription

  }

  deinit {
    Log.info("Spectro.deinit")

    // Stop recording + dealloc queue [which also deallocs its buffers, I hope?]
    if let _queue = queue {
      AudioQueueStop(_queue, true) // (No checkStatus)
      AudioQueueDispose(_queue, true) // (No checkStatus)
    }

    // Close audio file
    if let _audioFile = audioFile {
      AudioFileClose(_audioFile) // (No checkStatus)
    }

  }

  func start(
    outputPath:  String,
    refreshRate: Double // Hz
  ) throws -> Void {
    Log.info(String(format: "Spectro.start: %@", pretty([
      "outputPath": outputPath,
      "f_bins":     f_bins,
      "numBuffers": numBuffers,
      "queue":      queue as Any,
      "buffers":    buffers,
    ])))

    // Noop if already recording
    guard queue == nil else { return }

    // Set audio session mode for recording
    Log.debug("Spectro.start: AVAudioSession.setCategory(.playAndRecord)")
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord, mode: .default, options: [])

    // Reset audio file state
    self.outputPath = outputPath
    audioFile       = nil
    nPacketsWritten = 0
    nPathsSent      = 0
    samplesBuffer   = []
    spectroRange    = Spectro.zeroSpectroRange

    // Create audio file to record to
    let outputUrl  = NSURL(fileURLWithPath: outputPath)
    let fileType   = kAudioFileWAVEType // TODO .mp4 [Timesink! Need to muck with format/streamDescription + general trial and error]
    Log.debug(String(format: "Spectro.start: AudioFileCreateWithURL: %@", pretty([
      "outputUrl": outputUrl,
      "fileType": fileType,
      "streamDescription": streamDescription,
    ])))
    try checkStatus(AudioFileCreateWithURL(
      outputUrl,
      fileType,
      &streamDescription,
      .eraseFile, // NOTE Silently overwrite existing files, else weird hangs _after_ recording starts when file already exists
      &audioFile
    ))

    // Allocate audio queue
    Log.debug(String(format: "Spectro.start: AudioQueueNewInput: %@", pretty([
      "streamDescription": streamDescription,
    ])))
    try checkStatus(AudioQueueNewInput(
      &streamDescription,
      { (selfOpaque, inAQ, inBuffer, inStartTime, inNumberPacketDescriptions, inPacketDescs) -> Void in
        let selfTyped = Unmanaged<Spectro>.fromOpaque(selfOpaque!).takeUnretainedValue()
        return selfTyped.onAudioData(inAQ, inBuffer, inStartTime, inNumberPacketDescriptions, inPacketDescs)
      },
      UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()),
      nil, // inCallbackRunLoop:     nil = run callback on one of the audio queue's internal threads
      nil, // inCallbackRunLoopMode: nil = kCFRunLoopCommonModes
      0,   // inFlags: Must be 0 (reserved)
      &queue
    ))
    let _queue = queue!

    // Calculate bufferSize from refreshRate
    //  = bytes/buffer
    //  = bytes/s / (buffer/s)
    //  = bytes/sample * sample/s / (buffer/s)
    //  = mBytesPerPacket * mSampleRate / refreshRate
    let bufferSize = UInt32(Double(streamDescription.mBytesPerPacket) * streamDescription.mSampleRate / Double(refreshRate))

    // Allocate buffers for audio queue
    buffers = []
    for _ in 0..<numBuffers {
      var buffer: AudioQueueBufferRef?
      Log.debug(String(format: "Spectro.start: AudioQueueAllocateBuffer: %@", show([
        "numBuffers": numBuffers,
        "queue": _queue,
        "bufferSize": bufferSize,
      ])))
      try checkStatus(AudioQueueAllocateBuffer(_queue, bufferSize, &buffer))
      let _buffer = buffer!
      Log.debug(String(format: "Spectro.start: AudioQueueEnqueueBuffer: %@", show([
        "numBuffers": numBuffers,
        "queue": _queue,
        "buffer": _buffer,
      ])))
      try checkStatus(AudioQueueEnqueueBuffer(_queue, _buffer, 0, nil))
      buffers.append(_buffer)
    }

    // Start recording
    Log.debug(String(format: "Spectro.start: AudioQueueStart: %@", show([
      "queue": queue,
    ])))
    try checkStatus(AudioQueueStart(_queue, nil))

  }

  func stop() -> Promise<String?> {
    return Promise { () -> String? in
      Log.info("Spectro.stop")

      // Noop unless recording
      guard let _queue      = self.queue      else { return nil }
      guard let _audioFile  = self.audioFile  else { return nil } // Should be defined if queue is, but let's not risk races
      guard let _outputPath = self.outputPath else { return nil } // Should be defined if queue is, but let's not risk races

      // Stop recording
      Log.debug("Spectro.stop: AudioQueueStop + AudioQueueDispose")
      try checkStatus(AudioQueueStop(_queue, true))
      try checkStatus(AudioQueueDispose(_queue, true))

      // Reset audio session mode for playback
      Log.debug("Spectro.stop: AVAudioSession.setCategory(.playback)")
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playback, mode: .default, options: [])

      // Reset audio queue state
      self.queue = nil

      // Close audio file (but don't reset its state until the next .start(), so we can continue reading it)
      try checkStatus(AudioFileClose(_audioFile))

      return _outputPath
    }
  }

  func onAudioData(
    _ inQueue:      AudioQueueRef,
    _ inBuffer:     AudioQueueBufferRef,
    _ pStartTime:   UnsafePointer<AudioTimeStamp>,
    _ numPackets:   UInt32,
    _ pPacketDescs: UnsafePointer<AudioStreamPacketDescription>? // nil for uncompressed formats
  ) -> Void {
    let timer = Timer()
    var debugTimes: Array<(String, Double)> = [] // (Array of tuples b/c Dictionary is ordered by key i/o insertion)

    Log.info(String(format: "Spectro.onAudioData: %@", show([
      // "queue": queue,         // For debug
      // "audioFile": audioFile, // For debug
      "inQueue": inQueue,
      "inBuffer": inBuffer,
      // "startTime": pStartTime.pointee, // Lots of info I don't care about
      "numPackets": numPackets,
      "pPacketDescs": pPacketDescs,
    ])))

    // Noop unless recording
    guard queue != nil               else { return }
    guard queue == inQueue           else { return } // This would be a stop/start race, in which case audioFile is wrong
    guard let _audioFile = audioFile else { return } // Should be defined if queue is, but let's not risk races

    do {

      // Append to audioFile
      if numPackets > 0 {
        var ioNumPackets = numPackets
        try checkStatus(AudioFileWritePackets(
          _audioFile,
          false, // Don't cache the writen data [what does this mean?]
          inBuffer.pointee.mAudioDataByteSize,
          pPacketDescs,
          Int64(nPacketsWritten),
          &ioNumPackets, // in: num packets to write; out: num packets actually written
          inBuffer.pointee.mAudioData
        ))
        nPacketsWritten += ioNumPackets
      }
      debugTimes.append(("file", timer.lap()))

      // Read samples from inBuffer->mAudioData
      //  - 16-bit (because mBitsPerChannel)
      //  - Signed [QUESTION Empirically correct, but what determines signed vs. unsigned?]
      //  - TODO Probably need more checks here
      typealias Sample = Int16
      assert(streamDescription.mBitsPerChannel == 16, "Expected 16bit PCM data, got: \(streamDescription)")
      let nSamples = Int(inBuffer.pointee.mAudioDataByteSize) / (Sample.bitWidth / 8)

      // Don't emit event if no samples (e.g. flushing empty buffers on stop)
      if (nSamples > 0) {

        // samples
        let samples: [Float] = [Sample](UnsafeBufferPointer(
          start: inBuffer.pointee.mAudioData.bindMemory(to: Sample.self, capacity: nSamples),
          count: nSamples
        )).map {
          Float($0)
        }
        // Log.debug(String(format: "Spectro.onAudioData: samples[%d]: %@", // XXX Debug
        //   samples.count, show(samples.slice(to: 20), prec: 0)
        // ))
        debugTimes.append(("samples", timer.lap()))

        // Pad audio chunks to ≥nperseg with past audio, else gaps in streaming spectro stft
        //  - [Lots of pencil and paper...]
        //  - TODO Write tests for this gnar (tested manually)
        let nperseg      = features.nperseg
        let hop_length   = features.hop_length
        let _nPreSamples = samplesBuffer.count
        samplesBuffer    = samplesBuffer + samples
        if samplesBuffer.count < nperseg {
          Log.debug(String(format: "Spectro.onAudioData: %@", [
            "samplesBuffer[\(_nPreSamples)]+samples[\(samples.count)]->\(samplesBuffer.count) -> ",
            "waiting for ≥nperseg[\(nperseg)]",
          ].joined()))
        } else {
          let ready        = samplesBuffer.count / hop_length * hop_length
          let next         = (samplesBuffer.count / hop_length * hop_length) - nperseg + hop_length
          let samplesReady = Array(samplesBuffer.slice(to: ready))
          let _nPreNext    = samplesBuffer.count
          samplesBuffer    = Array(samplesBuffer.slice(from: next))
          Log.debug(String(format: "Spectro.onAudioData: %@", [
            "samplesBuffer[\(_nPreSamples)]+samples[\(samples.count)]->\(_nPreNext) -> ",
            "(ready[\(ready)], next[\(next)]) -> ",
            "samplesBuffer[\(_nPreNext)-\(next)->\(samplesBuffer.count)] (",
            "nperseg[\(nperseg)], hop_length[\(hop_length)]",
            ")",
          ].joined()))

          // S: spectro(samples)
          //  - (fs/ts are mocked as [] since we don't use them yet)
          let (_, _, S) = features._spectro(
            samplesReady,
            sample_rate: Int(streamDescription.mSampleRate),
            f_bins: f_bins,
            // Denoise doesn't make sense for streaming chunks in isolation:
            //  - Median filtering only makes sense when Δt is long enough for variation, which small chunks don't have
            //  - RMS norm would dampen variance for loud chunks and expand variance for quiet chunks, which is undesirable
            //  - We could probably devise a streaming denoise approach, but would it even be helpful to the user? [Probably not]
            denoise: false
          )
          debugTimes.append(("S", timer.lap()))

          // Accumulate spectroRange over time from each recorded chunk
          //  - This is the simplest adaptive approach [think carefully about user benefit vs. dev cost before complexifying this]
          spectroRange = spectroRange | Interval(min(S.grid), max(S.grid))
          Log.debug(String(format: "Spectro.onAudioData: %@", [
            "S[\(S.shape)]",
            "spectroRange[\(spectroRange)]",
            // "quantiles[\(Stats.quantiles(S.grid, bins: 3))]" // XXX Slow (sorting)
          ].joined(separator: ", ")))

          // Skip empty spectros (e.g. spectrogram returned an Nx0 matrix b/c samples.count < nperseg)
          //  - (This might not be necessary anymore since we added samplesBuffer, but let's keep it for safety)
          if S.isEmpty {
            Log.info("Spectro.onAudioData: Skipping image for empty spectro: samples[\(samples.count)] -> S[\(S.shape)]")
          } else {
            // Spectro -> image file
            let path = FileManager.default.temporaryDirectory.path / "\(DispatchTime.now().uptimeNanoseconds).png"
            let imageFile = try matrixToImageFile(
              path,
              S,
              range: spectroRange,
              colors: Colors.magma_r,
              timer: timer, debugTimes: &debugTimes // XXX Debug
            )
            // Image file path -> js (via rn event)
            nPathsSent += 1
            emitter.sendEvent(withName: "spectroFilePath", body: [
              "spectroFilePath": imageFile.path as Any,
              "width":           imageFile.width,
              "height":          imageFile.height,
              "nSamples":        samples.count,
              "debugTimes":      Array(debugTimes.map { (k, v) in ["k": k, "v": v] }),
            ] as Props)
          }

        }

      }

      // XXX Debug
      // Log.debug(String(format: "Spectro.onAudioData: debugTimes: %@", debugTimes.map { (k, v) in (k, Int(v * 1000)) }.description))

      // Re-enqueue consumed buffer to receive more audio data
      switch AudioQueueEnqueueBuffer(inQueue, inBuffer, 0, nil) {
        case kAudioQueueErr_EnqueueDuringReset: break // Ignore these (harmless?) errors on .stop()
        case let status: try checkStatus(status)
      }

    } catch {
      Log.error("Spectro.onAudioData: Error: \(error)")
    }
  }

  func stats() -> Props {
    return [
      "sampleRate": streamDescription.mSampleRate,
      "channels": streamDescription.mChannelsPerFrame,
      "bitsPerSample": streamDescription.mBitsPerChannel,
      "nPacketsWritten": nPacketsWritten,
      "nPathsSent": nPathsSent,
      "spectroRange": spectroRange.description,
      "outputPath": outputPath as Any,
    ]
  }

  func renderAudioPathToSpectroPath(
    _ audioPath: String,
    _ spectroPath: String,
    f_bins: Int, // Independent of self.f_bins [TODO Confusing, clean up]
    denoise: Bool
  ) throws -> ImageFile? {
    Log.info(String(format: "Spectro.renderAudioPathToSpectroPath: %@", [
      "audioPath": audioPath,
      "spectroPath": spectroPath,
      "f_bins": f_bins,
      "denoise": denoise,
    ]))

    // TODO Expose as params (to js via opts)
    let colors = Colors.magma_r

    // Read samples from file
    let (samples, sampleRate) = try featuresBirdgram.samplesFromAudioPath(audioPath)
    if samples.count == 0 {
      Log.info("Spectro.renderAudioPathToSpectroPath: No samples in audioPath[\(audioPath)]")
      return nil
    }

    // S: stft(samples)
    let (_, _, S) = features._spectro(
      samples,
      sample_rate: sampleRate,
      f_bins:      f_bins, // Allow nonstandard for RecordScreen, e.g. 80 i/o 40
      denoise:     denoise // Allow nonstandard for RecordScreen, e.g. false i/o true
    )

    // Spectro -> image file
    if S.isEmpty {
      Log.info("Spectro.renderAudioPathToSpectroPath: Empty spectro: samples[\(samples.count)] -> S[\(S.shape)] (e.g. <nperseg)")
      return nil
    }
    let imageFile = try matrixToImageFile(
      spectroPath,
      S,
      range: Interval(min(S.grid), max(S.grid)), // [Is min/max a robust behavior in general? Works well for doneRecording, at least]
      colors: colors
    )

    return imageFile

  }

  func editAudioPathToAudioPath(
    parentAudioPath: String,
    editAudioPath: String,
    draftEdit: DraftEdit
  ) throws -> Void {
    Log.info(String(format: "Spectro.editAudioPathToAudioPath: %@", [
      "parentAudioPath": parentAudioPath,
      "editAudioPath": editAudioPath,
      "draftEdit": draftEdit,
    ]))

    // Open parentFile (xc.mp4 | user.wav)
    let parentFile = try AKAudioFile(forReading: URL(fileURLWithPath: parentAudioPath))
    Log.debug(String(format: "Spectro.editAudioPathToAudioPath: parentFile: %@", [
      "path":             parentAudioPath,
      "length":           parentFile.length,
      "fileFormat":       parentFile.fileFormat.settings,
      "processingFormat": parentFile.processingFormat.settings,
    ]))
    // Assert sampleRate = config sample_rate, regardless of rec source
    //  - TODO(import_rec): We'll need to convert the input rec encoding when the user can import arbitrary audio files
    //  - User recs are [currently] .wav 22KHz,1ch as created by Spectro.swift:Spectro.start
    //  - XC recs are [currently] .mp4 22KHz,1ch as created by payloads.py
    if Int(parentFile.processingFormat.sampleRate) != sampleRate.intValue { // HACK Compare as int i/o double to avoid spurious failure
      throw AppError("parentFile sampleRate[\(parentFile.processingFormat.sampleRate))] must be \(sampleRate) (in \(parentFile.url))")
    }

    // Read parentSamples
    let parentSamples: [Float] = try local {
      guard let floatChannelData = parentFile.floatChannelData else {
        throw AppError("Null floatChannelData in parentFile[\(parentFile.url)]")
      }
      let Samples = Matrix(floatChannelData)
      let samples: [Float] = (Samples.shape.0 == 1
        ? Samples.grid               // 1ch -> 1ch
        : Samples.T.map { mean($0) } // 2ch -> 1ch
      )
      Log.debug(String(format: "Spectro.editAudioPathToAudioPath: parentSamples: %@", [
        "(ch,samples)": Samples.shape,
        "samples": samples.count,
      ]))
      return samples
    }

    // Make editFile to write to (.wav)
    //  - Avoid partial writes: write to a tmp file, then mv to editAudioPath to commit
    let tmpPath = NSTemporaryDirectory() / UUID().uuidString
    let editFile = try AVAudioFile(
      forWriting: URL(fileURLWithPath: tmpPath),
      settings:   settings
    )
    Log.debug(String(format: "Spectro.editAudioPathToAudioPath: editFile(prepare): %@", [
      "path":             tmpPath,
      "fileFormat":       editFile.fileFormat.settings,
      "processingFormat": editFile.processingFormat.settings,
    ]))

    // Make editSamples <- parentSamples clips
    //  - TODO Avoid the extra array copy [nontrivial complexity for a small perf gain that's not a bottleneck we care about]
    let editSamples: [Float]
    if let clips = draftEdit.clips {
      editSamples = clips.flatMap { clip -> [Float] in
        let n           = parentSamples.count
        let loSample    = clip.time.lo == -.infinity ? 0 : Int(round(clip.time.lo * sampleRate.doubleValue))
        let hiSample    = clip.time.hi == .infinity  ? n : Int(round(clip.time.hi * sampleRate.doubleValue))
        let gain        = clip.gain ?? 1 // QUESTION Is per-clip gain useful? Junks up the per-freq median clipping part of denoising...
        let clipSamples = (
          Array(parentSamples[loSample..<hiSample]) // Clip parent
          * Float(gain)                             // Apply gain
        )
        Log.debug(String(format: "Spectro.editAudioPathToAudioPath: clip: %@", [
          "clip":          clip,
          "loSample":      loSample,
          "hiSample":      hiSample,
          "gain":          gain,
          "parentSamples": parentSamples.count,
          "clipSamples":   clipSamples.count,
        ]))
        return clipSamples
      }
    } else {
      editSamples = parentSamples
    }

    // Write editSamples -> editFile
    precondition(editFile.processingFormat.channelCount == 1, "Expected ch=1: editFile.processingFormat[\(editFile.processingFormat)]")
    let editFrames = UInt32(editSamples.count) // Assuming frames = samples [NOTE Revisit if we write edits as mp4 i/o wav]
    guard let editBuffer = AVAudioPCMBuffer(
      pcmFormat:     editFile.processingFormat,
      frameCapacity: editFrames
    ) else { preconditionFailure("nil AVAudioPCMBuffer(pcmFormat: \(editFile.processingFormat), frameCapacity: \(editFrames))") }
    guard let editBufferSamples = editBuffer.floatChannelData?[0] else { preconditionFailure("nil editBuffer.floatChannelData") }
    for i in 0..<Int(editFrames) {
      editBufferSamples[i] = editSamples[i]
    }
    editBuffer.frameLength = editFrames // Mark all frameCapacity as valid
    try editFile.write(from: editBuffer)

    // Commit editFile: mv tmp -> editAudioPath
    Log.debug(String(format: "Spectro.editAudioPathToAudioPath: editFile(commit): %@", [
      "editAudioPath": editAudioPath,
    ]))
    try FileManager.default.moveItem(at: editFile.url, to: URL(fileURLWithPath: editAudioPath))

  }

}

//
// TODO Make a home for these
//

// Keep in sync with app/datatypes.ts:DraftEdit
public struct DraftEdit {

  public let clips: Array<Clip>?

  static func fromJson(_ json: JSON) throws -> DraftEdit {
    return DraftEdit(
      clips: try (json["clips"].array as [JSON]?).map { clips in
        try clips.map { try Clip.fromJson($0) }
      }
    )
  }

}

// Keep in sync with app/datatypes.ts:Clip
public struct Clip {

  public let time: Interval<Double>
  public let gain: Double?

  static func fromJson(_ json: JSON) throws -> Clip {
    return Clip(
      time: try Interval<Double>.fromJson(json["time"]),
      gain: json["gain"].double
    )
  }

}

extension Interval {
  static func fromJson(_ json: JSON) throws -> Interval<Double> {
    return Interval<Double>(
      try JsonSafeNumber.unsafe(json["lo"]),
      try JsonSafeNumber.unsafe(json["hi"])
    )
  }
}

public enum JsonSafeNumber {
  static func unsafe(_ json: JSON) throws -> Double {
    if let n = json.double {
      return n
    } else if let s = json.string {
      switch (s) {
        case "NaN":       return Double.nan
        case "Infinity":  return Double.infinity
        case "-Infinity": return -Double.infinity
        default:          throw AppError("Expected NaN/Infinity/-Infinity, got[\(s)] in: \(json)")
      }
    } else {
      throw AppError("Expected number or string for element[\(json)]")
    }
  }
}
