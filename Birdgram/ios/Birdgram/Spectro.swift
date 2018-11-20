import Foundation

import Bubo
import AudioKit // After Bubo

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

  var _proxy: Proxy?
  func proxy() throws -> Proxy {
    guard let x = _proxy else { throw AppError("Proxy is nil") }
    return x
  }

  func withPromise<X>(
    _ resolve: RCTPromiseResolveBlock,
    _ reject: RCTPromiseRejectBlock,
    _ name: String,
    _ f: () throws -> X
  ) -> Void {
    do {
      resolve(try f())
    } catch {
      let method = "\(type(of: self)).\(name)"
      let stack = Thread.callStackSymbols
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
    resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "setup") {
      _proxy = try Spectro.create(
        emitter:          self,
        outputPath:       self.getProp(opts, "outputPath") ?? throw_(AppError("outputPath is required")),
        bufferSize:       self.getProp(opts, "bufferSize"),
        sampleRate:       self.getProp(opts, "sampleRate"),
        bytesPerPacket:   self.getProp(opts, "bytesPerPacket"),
        framesPerPacket:  self.getProp(opts, "framesPerPacket"),
        bytesPerFrame:    self.getProp(opts, "bytesPerFrame"),
        channelsPerFrame: self.getProp(opts, "channelsPerFrame"),
        bitsPerChannel:   self.getProp(opts, "bitsPerChannel")
      )
    }
  }

  @objc func start(
    _ resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "start") {
      try proxy().start()
    }
  }

  @objc func stop(
    _ resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "stop") {
      try proxy().stop()
    }
  }

  // XXX Dev
  @objc func hello(
    _ x: String, y: String, z: NSNumber,
    resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock
  ) -> Void {
    withPromise(resolve, reject, "hello") {
      try proxy().hello(x, y, z)
    }
  }

}

class Spectro {

  static func supportedEvents() -> [String] {
    return [
      "audioData"
    ]
  }

  static func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return [:]
    // return [
    //   "one": 1,
    //   "two": "dos",
    // ]
  }

  static func create(
    emitter:          RCTEventEmitter,
    outputPath:       String,
    bufferSize:       UInt32?,
    sampleRate:       Double?,
    bytesPerPacket:   UInt32?,
    framesPerPacket:  UInt32?,
    bytesPerFrame:    UInt32?,
    channelsPerFrame: UInt32?,
    bitsPerChannel:   UInt32?
  ) throws -> Spectro {
    RCTLogInfo("Spectro.create")
    return Spectro(
      emitter:    emitter,
      outputPath: outputPath,
      bufferSize: bufferSize ?? 8192,
      desc:       AudioStreamBasicDescription(
        // https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
        // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/1572096-audio_data_format_identifiers
        // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/mpeg-4_audio_object_type_constants
        mSampleRate:       sampleRate       ?? 44100,
        mFormatID:         kAudioFormatLinearPCM, // TODO kAudioFormatMPEG4AAC [how to specify bitrate? else just try aac_he_v2]
        mFormatFlags:      0,
        mBytesPerPacket:   bytesPerPacket   ?? 2, // TODO Function of bitsPerChannel (and ...)?
        mFramesPerPacket:  framesPerPacket  ?? 1, // 1 for uncompressed
        mBytesPerFrame:    bytesPerFrame    ?? 2, // TODO Function of bitsPerChannel (and ...)?
        mChannelsPerFrame: channelsPerFrame ?? 1,
        mBitsPerChannel:   bitsPerChannel   ?? 16,
        mReserved:         0
      )
    )
  }

  // Params
  let emitter:    RCTEventEmitter
  let outputPath: String
  let bufferSize: UInt32
  var desc:       AudioStreamBasicDescription
  // let queue:      UnsafeMutablePointer<AudioQueueRef>
  // let buffer:     UnsafeMutablePointer<AudioQueueBufferRef>
  var queue:      AudioQueueRef?
  var buffer:     AudioQueueBufferRef?

  init(
    emitter:    RCTEventEmitter,
    outputPath: String,
    bufferSize: UInt32,
    desc:       AudioStreamBasicDescription
  ) {
    RCTLogInfo("Spectro.init")

    self.emitter    = emitter
    self.outputPath = outputPath
    self.bufferSize = bufferSize
    self.desc       = desc

    // self.queue   = UnsafeMutablePointer<AudioQueueRef>.allocate(capacity: 1)
    // self.buffer  = UnsafeMutablePointer<AudioQueueBufferRef>.allocate(capacity: 1)

    // let numBuffers = 3 // TODO Param
    // self.queue   = UnsafeMutablePointer<AudioQueueRef>.allocate(capacity: 1)
    // self.buffers = Array<AudioQueueBufferRef>(repeatedValue: nil, count: numBuffers)

  }

  deinit {
    RCTLogInfo("Spectro.deinit")

    if let _queue = self.queue { AudioQueueStop(_queue, true) } // Ignore OSStatus

    // queue.deallocate()
    // buffer.deallocate()

  }

  func start() throws -> Void {
    RCTLogInfo("Spectro.start")

    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord, mode: .default, options: [])
    try session.overrideOutputAudioPort(AVAudioSession.PortOverride.speaker)

    try checkStatus(AudioQueueNewInput(
      &desc,
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
    try checkStatus(AudioQueueAllocateBuffer(
      queue!,
      bufferSize * 2, // TODO Why *2?
      &buffer
    ))
    try checkStatus(AudioQueueEnqueueBuffer(
      queue!,
      buffer!,
      0,
      nil
    ))
    try checkStatus(AudioQueueStart(queue!, nil))

  }

  func stop() throws -> String {
    RCTLogInfo("Spectro.stop")

    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playback, mode: .default, options: [])

    try checkStatus(AudioQueueStop(queue!, true))

    return outputPath
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

  // XXX Dev
  func hello(_ x: String, _ y: String, _ z: NSNumber) -> String {
    return SpectroImpl.hello(x, y, z)
  }

}
