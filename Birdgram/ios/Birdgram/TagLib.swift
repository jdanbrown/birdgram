// (See Birdgram/Spectro.swift)

import Foundation

import Bubo // Before Bubo_Pods imports
import Promises

@objc(RNTagLib)
class RNTagLib: RCTEventEmitter, RNProxy {

  typealias Proxy = Void
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

  @objc func readComment(
    _ audioPath: String,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromiseNoProxy(resolve, reject, "readComment") {
      return try TagLibBirdgram.readComment(
        audioPath
      )
    }
  }

  @objc func writeComment(
    _ audioPath: String,
    value: String,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromiseNoProxy(resolve, reject, "writeComment") {
      return try TagLibBirdgram.writeComment(
        audioPath,
        value
      )
    }
  }

}

public enum TagLibBirdgram {

  public static func readComment(_ audioPath: String) throws -> String? {
    let timer = Timer()
    let value = try TagLib.readComment(audioPath)
    Log.info(String(format: "TagLibBirdgram.readComment: time[%.3f], audioPath[%@]", timer.time(), audioPath))
    return value
  }

  public static func writeComment(_ audioPath: String, _ value: String) throws {
    Log.debug(String(format: "TagLibBirdgram.writeComment: audioPath[%@], value[%@]", audioPath, value.slice(to: 1000))) // XXX Debug
    let timer = Timer()
    try TagLib.writeComment(audioPath, value)
    Log.info(String(format: "TagLibBirdgram.writeComment: time[%.3f], audioPath[%@], value[%@]",
      timer.time(), audioPath, value.slice(to: 1000)
    ))
  }

}
