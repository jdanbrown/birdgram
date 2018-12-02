// To expose a method from swift to js:
//  - app/native/Spectro.ts      - add js Spectro.f() calling objc NativeModules.RNSpectro.f()
//  - ios/Birdgram/Spectro.m     - add objc extern for swift RNSpectro.f()
//  - ios/Birdgram/Spectro.swift - add swift RNSpectro.f() calling Spectro.f()
//  - ios/Birdgram/Spectro.swift - add swift Spectro.f()

import Foundation

import Bubo // Before Bubo_Pods imports
import AudioKit
import Promises
import Surge

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

typealias Props = Dictionary<String, Any>

@objc(RNSpectro)
class RNSpectro: RCTEventEmitter {

  //
  // Boilerplate
  //

  var proxy: Proxy?

  func withPromise<X>(
    _ resolve: @escaping RCTPromiseResolveBlock,
    _ reject: @escaping RCTPromiseRejectBlock,
    _ name: String,
    _ f: @escaping () throws -> X
  ) -> Void {
    withPromiseAsync(resolve, reject, name) { () -> Promise<X> in
      return Promise { () -> X in
        return try f()
      }
    }
  }

  func withPromiseAsync<X>(
    _ resolve: @escaping RCTPromiseResolveBlock,
    _ reject: @escaping RCTPromiseRejectBlock,
    _ name: String,
    _ f: () -> Promise<X>
  ) -> Void {
    f().then { x in
      resolve(x)
    }.catch { error in
      let method = "\(type(of: self)).\(name)"
      let stack = Thread.callStackSymbols // TODO How to get stack from error i/o current frame? (which is doubly useless in async)
      reject(
        "\(method)",
        "method[\(method)] error[\(error)] stack[\n\(stack.joined(separator: "\n"))\n]",
        error
      )
    }
  }

  // Static constants exported to js once at init time (e.g. later changes will be ignored)
  //  - https://facebook.github.io/react-native/docs/native-modules-ios#exporting-constants
  @objc override func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return Proxy.constantsToExport()
  }

  @objc open override func supportedEvents() -> [String] {
    return Proxy.supportedEvents()
  }

  func getProp<X>(_ props: Props, _ key: String) throws -> X? {
    guard let x = props[key] else { return nil }
    guard let y = x as? X else { throw AppError("Failed to convert \(key)[\(x)] to type \(X.self)") }
    return y
  }

  //
  // Non-boilerplate
  //

  typealias Proxy = Spectro

  // requiresMainQueueSetup / methodQueue / dispatch_async
  //  - https://stackoverflow.com/a/51014267/397334
  //  - https://facebook.github.io/react-native/docs/native-modules-ios#threading
  //  - QUESTION Should we avoid blocking the main queue on long spectro operations?
  @objc static override func requiresMainQueueSetup() -> Bool {
    return false
  }

  @objc func setup(
    _ opts: Props,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "setup") {
      self.proxy = try Spectro.create(
        emitter:          self,
        outputFile:       self.getProp(opts, "outputFile") ?? throw_(AppError("outputFile is required")),
        // TODO Clean up unused params
        refreshRate:      self.getProp(opts, "refreshRate"),
        bufferSize:       self.getProp(opts, "bufferSize"),
        sampleRate:       self.getProp(opts, "sampleRate"),
        channels:         self.getProp(opts, "channels"),
        bytesPerPacket:   self.getProp(opts, "bytesPerPacket"),
        framesPerPacket:  self.getProp(opts, "framesPerPacket"),
        bytesPerFrame:    self.getProp(opts, "bytesPerFrame"),
        channelsPerFrame: self.getProp(opts, "channelsPerFrame"),
        bitsPerChannel:   self.getProp(opts, "bitsPerChannel")
      )
    }
  }

  @objc func start(
    _ resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    if let _proxy = proxy {
      withPromise(resolve, reject, "start") { try _proxy.start() }
    }
  }

  @objc func stop(
    _ resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    if let _proxy = proxy {
      withPromiseAsync(resolve, reject, "stop") { return _proxy.stop() }
    }
  }

  @objc func stats(
    _ resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    if let _proxy = proxy {
      withPromise(resolve, reject, "stats") { _proxy.stats() }
    }
  }

  @objc func renderAudioPathToSpectroPath(
    _ audioPath: String,
    spectroPath: String,
    opts: Props,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    if let _proxy = proxy {
      withPromise(resolve, reject, "renderAudioPathToSpectroPath") { () -> Props in
        let ret = try _proxy.renderAudioPathToSpectroPath(
          audioPath,
          spectroPath,
          denoise: self.getProp(opts, "denoise")
        )
        return [
          "width":  ret.width,
          "height": ret.height,
        ]
      }
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
class Spectro {

  static func supportedEvents() -> [String] {
    return [
      "audioChunk",
      "spectroFilePath",
    ]
  }

  static func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return [:]
  }

  static func create(
    emitter:          RCTEventEmitter,
    outputFile:       String,
    // TODO Clean up unused params
    refreshRate:      UInt32?,
    bufferSize:       UInt32?,
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
    // let _bufferSize       = bufferSize       ?? 2048 // HACK Manually tuned for (22050hz,1ch,16bit)
    // Calculation: bufferSize
    //  = bytes/buffer
    //  = bytes/s / (buffer/s)
    //  = bytes/sample * sample/s / (buffer/s)
    //  = mBytesPerPacket * mSampleRate / refreshRate
    let _refreshRate      = refreshRate      ?? 2 // Hz
    let _bufferSize       = bufferSize       ?? UInt32(Double(mBytesPerPacket) * mSampleRate / Double(_refreshRate))
    let mFormatFlags      = (
      // TODO Understand this. Was crashing without it. Blindly copied from RNAudioRecord.m (react-native-audio-record)
      mBitsPerChannel == 8
        ? kLinearPCMFormatFlagIsPacked
        : (kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked)
    )
    return Spectro(
      emitter:    emitter,
      outputFile: outputFile,
      bufferSize: _bufferSize,
      format:     AudioStreamBasicDescription(
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

  // Params
  let emitter:    RCTEventEmitter
  let outputFile: String
  let bufferSize: UInt32
  let numBuffers: Int
  var format:     AudioStreamBasicDescription

  // State
  var queue:             AudioQueueRef?        = nil
  var buffers:           [AudioQueueBufferRef] = []
  var audioFile:         AudioFileID?          = nil
  var numPacketsWritten: UInt32                = 0
  var spectroRange:      Interval<Float>       = Spectro.zeroSpectroRange
  static let zeroSpectroRange = Interval.bottom // Unit for union

  // TODO Take full outputPath from caller instead of hardcoding documentDirectory() here
  var outputPath: String { get { return "\(documentsDirectory())/\(outputFile)" } }

  init(
    emitter:    RCTEventEmitter,
    outputFile: String,
    bufferSize: UInt32,
    numBuffers: Int = 3, // ≥3 on iphone? [https://books.google.com/books?id=jiwEcrb_H0EC&pg=PA160]
    format:     AudioStreamBasicDescription
  ) {
    Log.info(String(format: "Spectro.init: %@", [
      "outputFile": outputFile,
      "bufferSize": bufferSize,
      "numBuffers": numBuffers,
      "format": format,
    ]))
    self.emitter    = emitter
    self.outputFile = outputFile
    self.bufferSize = bufferSize
    self.numBuffers = numBuffers
    self.format     = format
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

  func start() throws -> Void {
    Log.info(String(format: "Spectro.start: %@", pretty([
      "outputFile": outputFile,
      "numBuffers": numBuffers,
      "queue": queue as Any,
      "buffers": buffers,
    ])))

    // Noop if already recording
    guard queue == nil else { return }

    // Set audio session mode for recording
    Log.debug("Spectro.start: AVAudioSession.setCategory(.playAndRecord)")
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord, mode: .default, options: [])

    // Reset audio file state
    audioFile         = nil
    numPacketsWritten = 0
    spectroRange      = Spectro.zeroSpectroRange

    // Create audio file to record to
    //  - TODO Take full outputPath from caller instead of hardcoding documentDirectory() here
    let outputUrl  = NSURL(fileURLWithPath: outputPath)
    let fileType   = kAudioFileWAVEType // TODO .mp4 [Timesink! Need muck with format + general trial and error]
    Log.debug(String(format: "Spectro.start: AudioFileCreateWithURL: %@", pretty([
      "outputUrl": outputUrl,
      "fileType": fileType,
      "format": format,
    ])))
    AudioFileCreateWithURL(
      outputUrl,
      fileType,
      &format,
      .eraseFile, // NOTE Silently overwrite existing files, else weird hangs _after_ recording starts when file already exists
      &audioFile
    )

    // Allocate audio queue
    Log.debug(String(format: "Spectro.start: AudioQueueNewInput: %@", pretty([
      "format": format,
    ])))
    try checkStatus(AudioQueueNewInput(
      &format,
      { (selfOpaque, inAQ, inBuffer, inStartTime, inNumberPacketDescriptions, inPacketDescs) -> Void in
        let selfTyped = Unmanaged<Spectro>.fromOpaque(selfOpaque!).takeUnretainedValue()
        return selfTyped.onAudioData(inAQ, inBuffer, inStartTime, inNumberPacketDescriptions, inPacketDescs)
      },
      UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()),
      nil,
      nil,
      0, // Must be 0 (reserved)
      &queue
    ))
    let _queue = queue!

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
      guard let _queue     = self.queue     else { return nil }
      guard let _audioFile = self.audioFile else { return nil } // Should be defined if queue is, but let's not risk races

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

      return self.outputPath
    }
  }

  func onAudioData(
    _ inQueue:      AudioQueueRef,
    _ inBuffer:     AudioQueueBufferRef,
    _ pStartTime:   UnsafePointer<AudioTimeStamp>,
    _ numPackets:   UInt32,
    _ pPacketDescs: UnsafePointer<AudioStreamPacketDescription>? // nil for uncompressed formats
  ) -> Void {
    do {

      let timer = Timer()
      var debugTimes: Array<(String, Double)> = [] // (Array of tuples b/c Dictionary is ordered by key i/o insertion)

      Log.info(String(format: "Spectro.onAudioData: %@", show([
        // "self.queue": self.queue,         // For debug
        // "self.audioFile": self.audioFile, // For debug
        "inQueue": inQueue,
        "inBuffer": inBuffer,
        // "startTime": pStartTime.pointee, // Lots of info I don't care about
        "numPackets": numPackets,
        "pPacketDescs": pPacketDescs,
      ])))

      // Noop unless recording
      guard self.queue != nil               else { return }
      guard self.queue == inQueue           else { return } // This would be a stop/start race, in which case audioFile is wrong
      guard let _audioFile = self.audioFile else { return } // Should be defined if queue is, but let's not risk races

      // Append to audioFile
      if numPackets > 0 {
        var ioNumPackets = numPackets
        try checkStatus(AudioFileWritePackets(
          _audioFile,
          false, // Don't cache the writen data [what does this mean?]
          inBuffer.pointee.mAudioDataByteSize,
          pPacketDescs,
          Int64(numPacketsWritten),
          &ioNumPackets, // in: num packets to write; out: num packets actually written
          inBuffer.pointee.mAudioData
        ))
        numPacketsWritten += ioNumPackets
      }
      debugTimes.append(("file", timer.lap()))

      // XXX [Old] js audio->spectro
      // Send audio samples to js (via event)
      // let bytes: UnsafeMutableRawPointer = inBuffer.pointee.mAudioData
      // let base64: String = NSData(
      //   bytes: bytes,
      //   length: Int(inBuffer.pointee.mAudioDataByteSize)
      // ).base64EncodedString()
      // emitter.sendEvent(withName: "audioChunk", body: base64)

      // Read samples from inBuffer->mAudioData
      //  - 16-bit (because mBitsPerChannel)
      //  - Signed [QUESTION Empirically correct, but what determines signed vs. unsigned?]
      typealias Sample = Int16
      assert(format.mBitsPerChannel == 16, "Expected 16bit PCM data, got: \(format)") // TODO Probably more checks needed here
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
        Log.debug(String(format: "Spectro.onAudioData: samples[%d]: %@", // XXX Debug
          samples.count, show(samples.slice(to: 20), prec: 0)
        ))
        debugTimes.append(("samples", timer.lap()))

        // Denoise doesn't make sense for streaming chunks in isolation:
        //  - Median filtering only makes sense when Δt is long enough for variation, which small chunks don't have
        //  - RMS norm would dampen variance for loud chunks and expand variance for quiet chunks, which is undesirable
        //  - There's probably a streaming denoise approach we could devise, but would it even be helpful to the user? [Probably not]
        let denoise = false

        // S: stft(samples)
        //  - (fs/ts are mocked as [] since we don't use them yet)
        let (_, _, S) = Features.spectro(
          samples,
          sample_rate: Int(format.mSampleRate),
          denoise: denoise
        )
        debugTimes.append(("S", timer.lap()))

        // Accumulate lo/hi over time of each recording
        //  - A very simple adaptive approach to lo/hi [think carefully about user benefit vs. dev cost before complexifying this]
        spectroRange = spectroRange | Interval(min(S.grid), max(S.grid))
        let (lo, hi) = (spectroRange.lo, spectroRange.hi)
        Log.debug(String(format: "Spectro.onAudioData: %@", // XXX Debug
          "S[\(S.shape)], spectroRange[\(spectroRange)], lo[\(lo)], hi[\(hi)], quantiles[\(Stats.quantiles(S.grid, bins: 3))]"
        ))

        // Skip empty spectros (e.g. spectrogram returned an Nx0 matrix b/c samples.count < nperseg)
        if S.isEmpty {
          Log.info("Spectro.onAudioData: Skipping image for empty spectro: samples[\(samples.count)] -> S[\(S.shape)]")
        } else {
          // Spectro -> image file
          let path = FileManager.default.temporaryDirectory.path / "\(DispatchTime.now().uptimeNanoseconds).png"
          let (width, height) = try matrixToImageFile(
            path,
            S.vect { $0.map { x in (x.clamped(lo, hi) - lo) / (hi - lo) }},
            colors: Colors.magma_r,
            timer: timer, debugTimes: &debugTimes // XXX Debug
          )
          // Image file path -> js (via rn event)
          emitter.sendEvent(withName: "spectroFilePath", body: [
            "spectroFilePath": path as Any,
            "width": width,
            "height": height,
            "nSamples": samples.count,
            "debugTimes": Array(debugTimes.map { (k, v) in ["k": k, "v": v] }),
          ] as Props)
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
      "sampleRate": format.mSampleRate,
      "channels": format.mChannelsPerFrame,
      "bitsPerSample": format.mBitsPerChannel,
      "numPacketsWritten": numPacketsWritten,
      "spectroRange": spectroRange.description,
      "outputFile": outputFile,
    ]
  }

  func renderAudioPathToSpectroPath(
    _ audioPath: String,
    _ spectroPath: String,
    denoise: Bool? = nil
  ) throws -> (
    width: Int,
    height: Int
  ) {
    Log.info(String(format: "Spectro.renderAudioPathToSpectroPath: %@", [
      "audioPath": audioPath,
      "spectroPath": spectroPath,
      "denoise": denoise as Any,
    ]))

    // TODO Expose as params (to js via opts)
    let colors = Colors.magma_r

    // Read samples from file
    let file = try AKAudioFile(forReading: URL(fileURLWithPath: audioPath))
    guard let floatChannelData = file.floatChannelData else {
      throw AppError("No floatChannelData in file[\(file.url)]")
    }
    let Samples = Matrix(floatChannelData)
    let samples = transpose(Samples).map { mean($0) } // Convert to 1ch if >1ch
    if samples.count == 0 {
      throw AppError("No samples in file[\(file.url)]")
    }

    // S: stft(samples)
    if file.processingFormat.sampleRate != 22050 {
      // Throw i/o crashing (via precondition in Features.spectro)
      throw AppError("sampleRate[\(file.processingFormat.sampleRate)] must be 22050 for Features.spectro, in file[\(file.url)]")
    }
    let (_, _, S) = Features.spectro(
      samples,
      sample_rate: Int(file.processingFormat.sampleRate),
      denoise: denoise ?? true
    )

    // Spectro -> image file
    if S.isEmpty {
      throw AppError("Empty spectro: samples[\(samples.count)] -> S[\(S.shape)] (probably samples.count < nperseg)")
    }
    let (lo, hi) = (min(S.grid), max(S.grid)) // HACK Is min/max a good behavior in general? Works well for doneRecording, at least
    let dims = try matrixToImageFile(
      spectroPath,
      S.vect { $0.map { x in (x.clamped(lo, hi) - lo) / (hi - lo) }},
      colors: colors
    )

    return dims

  }

}
