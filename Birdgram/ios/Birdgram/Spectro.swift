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

class Spectro {

  static func supportedEvents() -> [String] {
    return [
      "audioData"
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
      desc:       AudioStreamBasicDescription(
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
        mChannelsPerFrame: channelsPerFrame ?? 1,
        mBitsPerChannel:   bitsPerChannel   ?? 16,
        mReserved:         0
      )
    )
  }

  // Params
  let emitter:    RCTEventEmitter
  let outputFile: String
  let desc:       AudioStreamBasicDescription

  // State
  var recorder: AKNodeRecorder?
  var recPath:  String?

  init(
    emitter:    RCTEventEmitter,
    outputFile: String,
    bufferSize: UInt32,
    desc:       AudioStreamBasicDescription
  ) {
    RCTLogInfo("Spectro.init")
    self.emitter    = emitter
    self.outputFile = outputFile
    self.desc       = desc
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
    AKSettings.sampleRate = desc.mSampleRate
    AKSettings.channelCount = desc.mChannelsPerFrame
    //  - XXX DONT USE AudioKit.format, use AKSettings (above)
    // AudioKit.format = AVAudioFormat(
    //   // WARNING .pcmFormatInt16 crashes AudioKit.start(); convert manually [https://stackoverflow.com/a/9153854/397334]
    //   commonFormat: .pcmFormatFloat32, // TODO Convert float32 -> uint16 downstream (+ assert desc.mBitsPerChannel == 16)
    //   sampleRate: desc.mSampleRate,
    //   channels: desc.mChannelsPerFrame,
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
      "sampleRate": desc.mSampleRate,
      "channels": desc.mChannelsPerFrame,
      // "bitsPerSample": desc.mBitsPerChannel, // TODO
      "duration": recorder?.recordedDuration as Any,
      "path": recPath as Any,
    ]
  }

  // XXX Dev
  // func hello(_ x: String, _ y: String, _ z: NSNumber) -> String {
  //   return SpectroImpl.hello(x, y, z)
  // }

}

// XXX after de-risking and committing to AudioKit
// class _Spectro {
//
//   static func supportedEvents() -> [String] {
//     return [
//       "audioData"
//     ]
//   }
//
//   static func constantsToExport() -> Dictionary<AnyHashable, Any> {
//     return [:]
//     // return [
//     //   "one": 1,
//     //   "two": "dos",
//     // ]
//   }
//
//   static func create(
//     emitter:          RCTEventEmitter,
//     outputFile:       String,
//     bufferSize:       UInt32?,
//     sampleRate:       Double?,
//     bytesPerPacket:   UInt32?,
//     framesPerPacket:  UInt32?,
//     bytesPerFrame:    UInt32?,
//     channelsPerFrame: UInt32?,
//     bitsPerChannel:   UInt32?
//   ) throws -> Spectro {
//     RCTLogInfo("Spectro.create")
//     return Spectro(
//       emitter:    emitter,
//       outputFile: outputFile,
//       bufferSize: bufferSize ?? 8192,
//       desc:       AudioStreamBasicDescription(
//         // https://developer.apple.com/documentation/coreaudio/audiostreambasicdescription
//         // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/1572096-audio_data_format_identifiers
//         // https://developer.apple.com/documentation/coreaudio/core_audio_data_types/mpeg-4_audio_object_type_constants
//         mSampleRate:       sampleRate       ?? 44100,
//         mFormatID:         kAudioFormatLinearPCM, // TODO kAudioFormatMPEG4AAC [how to specify bitrate? else just try aac_he_v2]
//         mFormatFlags:      0,
//         mBytesPerPacket:   bytesPerPacket   ?? 2, // TODO Function of bitsPerChannel (and ...)?
//         mFramesPerPacket:  framesPerPacket  ?? 1, // 1 for uncompressed
//         mBytesPerFrame:    bytesPerFrame    ?? 2, // TODO Function of bitsPerChannel (and ...)?
//         mChannelsPerFrame: channelsPerFrame ?? 1,
//         mBitsPerChannel:   bitsPerChannel   ?? 16,
//         mReserved:         0
//       )
//     )
//   }
//
//   // Params
//   let emitter:    RCTEventEmitter
//   let outputFile: String
//   let bufferSize: UInt32
//   var desc:       AudioStreamBasicDescription
//   // let queue:      UnsafeMutablePointer<AudioQueueRef>
//   // let buffer:     UnsafeMutablePointer<AudioQueueBufferRef>
//   var queue:      AudioQueueRef?
//   var buffer:     AudioQueueBufferRef?
//
//   init(
//     emitter:    RCTEventEmitter,
//     outputFile: String,
//     bufferSize: UInt32,
//     desc:       AudioStreamBasicDescription
//   ) {
//     RCTLogInfo("Spectro.init")
//
//     self.emitter    = emitter
//     self.outputFile = outputFile
//     self.bufferSize = bufferSize
//     self.desc       = desc
//
//     // self.queue   = UnsafeMutablePointer<AudioQueueRef>.allocate(capacity: 1)
//     // self.buffer  = UnsafeMutablePointer<AudioQueueBufferRef>.allocate(capacity: 1)
//
//     // let numBuffers = 3 // TODO Param
//     // self.queue   = UnsafeMutablePointer<AudioQueueRef>.allocate(capacity: 1)
//     // self.buffers = Array<AudioQueueBufferRef>(repeatedValue: nil, count: numBuffers)
//
//   }
//
//   deinit {
//     RCTLogInfo("Spectro.deinit")
//
//     if let _queue = self.queue { AudioQueueStop(_queue, true) } // Ignore OSStatus
//
//     // queue.deallocate()
//     // buffer.deallocate()
//
//   }
//
//   func start() throws -> Void {
//     RCTLogInfo("Spectro.start")
//
//     let session = AVAudioSession.sharedInstance()
//     try session.setCategory(.playAndRecord, mode: .default, options: [])
//     try session.overrideOutputAudioPort(AVAudioSession.PortOverride.speaker)
//
//     try checkStatus(AudioQueueNewInput(
//       &desc,
//       { (selfOpaque, inAQ, inBuffer, inStartTime, inNumberPacketDescriptions, inPacketDescs) -> Void in
//         let selfTyped = Unmanaged<Spectro>.fromOpaque(selfOpaque!).takeUnretainedValue()
//         return selfTyped.onAudioData(inAQ, inBuffer, inStartTime, inNumberPacketDescriptions, inPacketDescs)
//       },
//       UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()),
//       nil,
//       nil,
//       0, // Must be 0 (reserved)
//       &queue
//     ))
//     try checkStatus(AudioQueueAllocateBuffer(
//       queue!,
//       bufferSize * 2, // TODO Why *2?
//       &buffer
//     ))
//     try checkStatus(AudioQueueEnqueueBuffer(
//       queue!,
//       buffer!,
//       0,
//       nil
//     ))
//     try checkStatus(AudioQueueStart(queue!, nil))
//
//   }
//
//   func stop() throws -> String {
//     RCTLogInfo("Spectro.stop")
//
//     let session = AVAudioSession.sharedInstance()
//     try session.setCategory(.playback, mode: .default, options: [])
//
//     try checkStatus(AudioQueueStop(queue!, true))
//
//     return outputFile
//   }
//
//   func onAudioData(
//     _ inAQ:                       AudioQueueRef,
//     _ inBuffer:                   AudioQueueBufferRef,
//     _ inStartTime:                UnsafePointer<AudioTimeStamp>,
//     _ inNumberPacketDescriptions: UInt32,
//     _ inPacketDescs:              UnsafePointer<AudioStreamPacketDescription>?
//   ) -> Void {
//     RCTLogInfo("Spectro.onAudioData")
//     // TODO(swift_spectro)
//   }
//
//   // XXX Dev
//   func hello(_ x: String, _ y: String, _ z: NSNumber) -> String {
//     return SpectroImpl.hello(x, y, z)
//   }
//
// }
