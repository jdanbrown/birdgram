import Foundation

import Bubo // Before Bubo_Pods imports
import AudioKit
import Promises

// Docs
//  - react-native/React/Base/RCTBridgeModule.h
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

  func getProp<X>(_ props: Dictionary<String, Any>, _ key: String) throws -> X? {
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
    _ opts: Dictionary<String, Any>,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "setup") {
      self.proxy = try Spectro.create(
        emitter:          self,
        outputFile:       self.getProp(opts, "outputFile") ?? throw_(AppError("outputFile is required")),
        // TODO Clean up unused params
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
      withPromiseAsync(resolve, reject, "stop") { _proxy.stop() }
    }
  }

  @objc func stats(
    _ resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    if let _proxy = proxy {
      withPromise(resolve, reject, "stats") { _proxy.stats() }
    }
  }

  // XXX Dev
  // @objc func hello(
  //   _ x: String, y: String, z: NSNumber,
  //   resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock
  // ) -> Void {
  //   withPromise(resolve, reject, "hello") {
  //     proxy?.hello(x, y, z)
  //   }
  // }

}

class Spectro
  // : _SpectroAudioKit
  : _SpectroBasic
{}

class _SpectroAudioKit {

  static func supportedEvents() -> [String] {
    return [
      "audioChunk"
    ]
  }

  static func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return [:]
  }

  static func create(
    emitter:          RCTEventEmitter,
    outputFile:       String,
    // TODO Clean up unused params
    bufferSize:       UInt32?,
    sampleRate:       Double?,
    channels:         UInt32?,
    bytesPerPacket:   UInt32?,
    framesPerPacket:  UInt32?,
    bytesPerFrame:    UInt32?,
    channelsPerFrame: UInt32?,
    bitsPerChannel:   UInt32?
  ) throws -> Spectro {
    RCTLogInfo("Spectro.create")
    return Spectro(
      emitter:    emitter,
      outputFile: outputFile,
      bufferSize: bufferSize ?? 8192,
      format:     AudioStreamBasicDescription(
        // TODO Clean up unused params
        // https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
        // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/1572096-audio_data_format_identifiers
        // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/mpeg-4_audio_object_type_constants
        mSampleRate:       sampleRate       ?? 44100,
        mFormatID:         kAudioFormatLinearPCM, // TODO kAudioFormatMPEG4AAC [how to specify bitrate? else just try aac_he_v2]
        mFormatFlags:      0,
        mBytesPerPacket:   bytesPerPacket   ?? 2, // TODO Function of bitsPerChannel (and ...)?
        mFramesPerPacket:  framesPerPacket  ?? 1, // 1 for uncompressed
        mBytesPerFrame:    bytesPerFrame    ?? 2, // TODO Function of bitsPerChannel (and ...)?
        mChannelsPerFrame: channelsPerFrame ?? channels ?? 1,
        mBitsPerChannel:   bitsPerChannel   ?? 16,
        mReserved:         0
      )
    )
  }

  // Params
  let emitter:    RCTEventEmitter
  let outputFile: String
  let format:     AudioStreamBasicDescription

  // State
  var recorder: AKNodeRecorder?
  var recPath:  String?

  init(
    emitter:    RCTEventEmitter,
    outputFile: String,
    bufferSize: UInt32,
    format:     AudioStreamBasicDescription
  ) {
    RCTLogInfo("Spectro.init")
    self.emitter    = emitter
    self.outputFile = outputFile
    self.format     = format
  }

  deinit {
    RCTLogInfo("Spectro.deinit")
    do { let _ = try _stop() } catch { RCTLogWarn("Spectro.deinit: Ignoring error from .stop: \(error)") }
  }

  func start() throws -> Void {
    RCTLogInfo("Spectro.start")

    if (recorder?.isRecording ?? false) {
      try _stop()
    }

    // WARNING Crash on app startup [wat]
    // AKSettings.bufferLength = .huge // 2^11 = 2048 samples (92.9ms @ 22050hz)
    // AKSettings.recordingBufferLength = .huge

    // Settings
    try AKSettings.session.setCategory(.playAndRecord, mode: .default, options: [])

    // WARNING Set AKSettings, not AudioKit.format
    //  - IDK WTF AudioKit.format is for, but AKAudioFile() uses params from AKSettings.format and ignores AudioKit.format
    AKSettings.sampleRate = format.mSampleRate
    AKSettings.channelCount = format.mChannelsPerFrame
    //  - XXX DONT USE AudioKit.format, use AKSettings (above)
    // AudioKit.format = AVAudioFormat(
    //   // WARNING .pcmFormatInt16 crashes AudioKit.start(); convert manually [https://stackoverflow.com/a/9153854/397334]
    //   commonFormat: .pcmFormatFloat32, // TODO Convert float32 -> uint16 downstream (+ assert format.mBitsPerChannel == 16)
    //   sampleRate: format.mSampleRate,
    //   channels: format.mChannelsPerFrame,
    //   interleaved: false
    // )!

    RCTLogTrace("Spectro.start: AudioKit.inputDevices: \(String(describing: AudioKit.inputDevices))")
    RCTLogTrace("Spectro.start: AudioKit.inputDevice: \(String(describing: AudioKit.inputDevice))")
    RCTLogTrace("Spectro.start: AudioKit.format: \(AudioKit.format)")
    RCTLogTrace("Spectro.start: AKSettings.session: \(AKSettings.session)")

    let mic = AKMicrophone()
    recorder = try AKNodeRecorder(node: AKMixer(mic)) // WARNING AKMixer before .installTap else crash
    let output = AKBooster(mic, gain: 0) // Playback silent output else speaker->mic feedback

    let bufSize: UInt32 = AKSettings.recordingBufferLength.samplesCount
    RCTLogTrace("Spectro.start: Installing tap: bufSize[\(bufSize)]")
    AKMixer(mic).avAudioUnitOrNode.installTap( // WARNING AKMixer before .installTap else crash
       onBus: 0, bufferSize: bufSize, format: nil
    ) { (buffer: AVAudioPCMBuffer!, time: AVAudioTime!) -> Void in
      RCTLogTrace(String(format:"Spectro.start: Tap: %@, %d, %d, %@",
        buffer, buffer.frameLength, buffer.stride, String(describing: self.recorder?.recordedDuration)
        // TODO TODO De-risk this (else we should question whether to stick with AudioKit)
      ))
    }

    recPath = nil
    try recorder!.reset()  // Calls .removeTap + resets .internalAudioFile
    try recorder!.record() // Calls .installTap

    RCTLogTrace(String(format: "Spectro.start: audioFile.url[%s]", String(describing: recorder!.audioFile?.url))) // XXX Dev

    AudioKit.output = output
    try AudioKit.start()

  }

  func _stop() throws -> Void {
    RCTLogTrace("Spectro._stop")
    try AudioKit.stop()
    recorder?.stop()
    try AKSettings.session.setCategory(.playback, mode: .default, options: [])
  }

  func stop() -> Promise<String?> {
    return Promise<String?> { fulfill, reject in
      RCTLogTrace("Spectro.stop: recordedDuration: \(String(describing: self.recorder?.recordedDuration))")

      // Guards before stopping
      guard let recorder = self.recorder else {
        return fulfill(nil)
      }
      guard recorder.isRecording else {
        return fulfill(nil)
      }

      RCTLogTrace(String(format: "Spectro.stop: Stopping: %@", [ // XXX Dev
        "AudioKit.format": AudioKit.format,
        "audioFile.fileFormat": recorder.audioFile!.fileFormat,
      ]))

      // Stop recording
      try self._stop()

      RCTLogTrace(String(format: "Spectro.stop: Stopped: %@", [ // XXX Dev
        "AudioKit.format": AudioKit.format,
        "audioFile.fileFormat": recorder.audioFile!.fileFormat,
      ]))

      // Guards before export
      guard recorder.recordedDuration > 1e-6 else {
        RCTLogInfo("Spectro.stop: Skipping export for empty recordedDuration[\(recorder.recordedDuration)]")
        return fulfill(nil)
      }
      let (filename, ext) = pathSplitExt(pathBasename(self.outputFile))
      guard ext == "mp4" else {
        throw AppError("Output file extension must be mp4, got \(ext) (from outputFile: \(self.outputFile))")
      }
      guard AKSettings.sampleRate == 44100 else {
        throw AppError("sampleRate[\(AKSettings.sampleRate)] not supported, must be 44100 (see comments in Spectro.swift)")
      }

      // Export <temp>/*.caf -> <documents>/*.mp4
      //  - NOTE Can't make sampleRate=22050 work, going with 44100
      //    - Maybe related? https://github.com/AudioKit/AudioKit/issues/1009
      //  - NOTE sampleRate/channels are set by AKSettings.sampleRate/.channelCount, which we set above (in .start)
      //    - via AKAudioFile() with no args, which reads from AKSettings (WARNING and not AudioKit.format)
      //  - NOTE AKAudioFile.exportAsynchronously seems to trigger fewer mysterious failures then AKConverter
      //    - e.g. AKConverter kept failing with https://www.osstatus.com/search/results?search=2003334207
      recorder.audioFile!.exportAsynchronously(
        name: filename,
        baseDir: .documents, // Poop: .custom -> "not implemented yet" + NSError.fileCreateError [AKAudioFile.swift]
        exportFormat: .mp4
      ) { (outputFile: AKAudioFile?, error: NSError?) -> Void in
        if let _error = error {
          reject(_error)
        } else if let _outputFile = outputFile {
          self.recPath = _outputFile.url.path
          RCTLogInfo(String(format: "Spectro.stop: Exported recorded audio: %@", [
            "fileFormat": _outputFile.fileFormat,
            "settings": _outputFile.fileFormat.settings,
            "duration": recorder.recordedDuration,
            "recPath": self.recPath!,
          ]))
          fulfill(self.recPath)
        }
      }

    }
  }

  func onAudioData(
    _ inAQ:                       AudioQueueRef,
    _ inBuffer:                   AudioQueueBufferRef,
    _ inStartTime:                UnsafePointer<AudioTimeStamp>,
    _ inNumberPacketDescriptions: UInt32,
    _ inPacketDescs:              UnsafePointer<AudioStreamPacketDescription>?
  ) -> Void {
    RCTLogInfo("Spectro.onAudioData")
    // TODO(swift_spectro)
  }

  func stats() -> Dictionary<String, Any> {
    return [
      "sampleRate": format.mSampleRate,
      "channels": format.mChannelsPerFrame,
      // "bitsPerSample": format.mBitsPerChannel, // TODO
      "duration": recorder?.recordedDuration as Any,
      "path": recPath as Any,
    ]
  }

  // XXX Dev
  // func hello(_ x: String, _ y: String, _ z: NSNumber) -> String {
  //   return SpectroImpl.hello(x, y, z)
  // }

}

// Leaned heavily on these very simple and clear examples to make this thing work
//  - https://github.com/carsonmcdonald/AVSExample-Swift/blob/master/AVSExample/SimplePCMRecorder.swift
//  - https://github.com/goodatlas/react-native-audio-record/blob/master/ios/RNAudioRecord.m
//  - https://github.com/chadsmith/react-native-microphone-stream/blob/master/ios/MicrophoneStream.m
class _SpectroBasic {

  static func supportedEvents() -> [String] {
    return [
      "audioChunk"
    ]
  }

  static func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return [:]
  }

  static func create(
    emitter:          RCTEventEmitter,
    outputFile:       String,
    // TODO Clean up unused params
    bufferSize:       UInt32?,
    sampleRate:       Double?,
    channels:         UInt32?,
    bytesPerPacket:   UInt32?,
    framesPerPacket:  UInt32?,
    bytesPerFrame:    UInt32?,
    channelsPerFrame: UInt32?,
    bitsPerChannel:   UInt32?
  ) throws -> Spectro {
    RCTLogInfo("Spectro.create")
    let mSampleRate       = sampleRate       ?? 44100
    let mBitsPerChannel   = bitsPerChannel   ?? 16
    let mChannelsPerFrame = channelsPerFrame ?? channels ?? 2
    let mBytesPerPacket   = bytesPerPacket   ?? (mBitsPerChannel / 8 * mChannelsPerFrame)
    let mBytesPerFrame    = bytesPerFrame    ?? mBytesPerPacket // Default assumes PCM
    let mFramesPerPacket  = framesPerPacket  ?? 1 // 1 for uncompressed
    let _bufferSize       = bufferSize       ?? 2048 // HACK Manually tuned for (22050hz,1ch,16bit)
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
        mFormatID:         kAudioFormatLinearPCM, // TODO kAudioFormatMPEG4AAC [how to specify bitrate? else just try aac_he_v2]
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

  // TODO Take full outputPath from caller instead of hardcoding documentDirectory() here
  var outputPath: String { get { return "\(documentsDirectory())/\(outputFile)" } }

  init(
    emitter:    RCTEventEmitter,
    outputFile: String,
    bufferSize: UInt32,
    numBuffers: Int = 3, // ≥3 on iphone? [https://books.google.com/books?id=jiwEcrb_H0EC&pg=PA160]
    format:     AudioStreamBasicDescription
  ) {
    RCTLogInfo("Spectro.init")
    self.emitter    = emitter
    self.outputFile = outputFile
    self.bufferSize = bufferSize
    self.numBuffers = numBuffers
    self.format     = format
  }

  deinit {
    RCTLogInfo("Spectro.deinit")

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
    RCTLogInfo(String(format: "Spectro.start: %@", pretty([
      "outputFile": outputFile,
      "numBuffers": numBuffers,
      "queue": queue as Any,
      "buffers": buffers,
    ])))

    // Noop if already recording
    guard queue == nil else { return }

    // Set audio session mode for recording
    RCTLogTrace("Spectro.start: AVAudioSession.setCategory(.playAndRecord)")
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord, mode: .default, options: [])

    // Reset audio file state
    audioFile         = nil
    numPacketsWritten = 0

    // Create audio file to record to
    //  - TODO Take full outputPath from caller instead of hardcoding documentDirectory() here
    let outputUrl  = NSURL(fileURLWithPath: outputPath)
    let fileType   = kAudioFileWAVEType // TODO .mp4 [Timesink! Need muck with format + general trial and error]
    RCTLogTrace(String(format: "Spectro.start: AudioFileCreateWithURL: %@", pretty([
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
    RCTLogTrace(String(format: "Spectro.start: AudioQueueNewInput: %@", pretty([
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
      RCTLogTrace(String(format: "Spectro.start: AudioQueueAllocateBuffer: %@", show([
        "numBuffers": numBuffers,
        "queue": _queue,
        "bufferSize": bufferSize,
      ])))
      try checkStatus(AudioQueueAllocateBuffer(_queue, bufferSize * 2, &buffer)) // TODO Why *2?
      let _buffer = buffer!
      RCTLogTrace(String(format: "Spectro.start: AudioQueueEnqueueBuffer: %@", show([
        "numBuffers": numBuffers,
        "queue": _queue,
        "buffer": _buffer,
      ])))
      try checkStatus(AudioQueueEnqueueBuffer(_queue, _buffer, 0, nil))
      buffers.append(_buffer)
    }

    // Start recording
    RCTLogTrace(String(format: "Spectro.start: AudioQueueStart: %@", show([
      "queue": queue,
    ])))
    try checkStatus(AudioQueueStart(_queue, nil))

  }

  func stop() -> Promise<String?> {
    return Promise { () -> String? in
      RCTLogInfo("Spectro.stop")

      // Noop unless recording
      guard let _queue     = self.queue     else { return nil }
      guard let _audioFile = self.audioFile else { return nil } // Should be defined if queue is, but let's not risk races

      // Stop recording
      RCTLogTrace("Spectro.start: AudioQueueStop + AudioQueueDispose")
      try checkStatus(AudioQueueStop(_queue, true))
      try checkStatus(AudioQueueDispose(_queue, true))

      // Reset audio session mode for playback
      RCTLogTrace("Spectro.stop: AVAudioSession.setCategory(.playback)")
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
      RCTLogInfo(String(format: "Spectro.onAudioData: %@", show([
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

      // Compute spectro chunks from audio samples
      //  - TODO Switch js -> Surge [after e2e with existing js spectro code]
      let bytes: UnsafeMutableRawPointer = inBuffer.pointee.mAudioData
      let base64: String = NSData(
        bytes: bytes,
        length: Int(inBuffer.pointee.mAudioDataByteSize)
      ).base64EncodedString()
      emitter.sendEvent(withName: "audioChunk", body: base64)

      // Re-enqueue consumed buffer to receive more audio data
      switch AudioQueueEnqueueBuffer(inQueue, inBuffer, 0, nil) {
      case kAudioQueueErr_EnqueueDuringReset: break // Ignore these (harmless?) errors on .stop()
      case let status: try checkStatus(status)
      }

    } catch {
      RCTLogError("Spectro.onAudioData: Error: \(error)")
    }
  }

  func stats() -> Dictionary<String, Any> {
    return [
      "sampleRate": format.mSampleRate,
      "channels": format.mChannelsPerFrame,
      "bitsPerSample": format.mBitsPerChannel,
      "numPacketsWritten": numPacketsWritten,
      "outputFile": outputFile,
    ]
  }

  // XXX Dev
  // func hello(_ x: String, _ y: String, _ z: NSNumber) -> String {
  //   return SpectroImpl.hello(x, y, z)
  // }

}
