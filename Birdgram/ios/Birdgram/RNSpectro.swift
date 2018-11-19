import Foundation

import Bubo
import AudioKit // After Bubo

// NOTE
//  - Returning values from methods requires callbacks/promises (prefer promises)
//    - https://facebook.github.io/react-native/docs/native-modules-ios#promises
//  - @objc for functions with args(/ only if they return a promise?) requires `_ foo` on first arg
//    - https://stackoverflow.com/a/39840952/397334

@objc(RNSpectro)
class RNSpectro: NSObject {

  // requiresMainQueueSetup + methodQueue
  //  - https://stackoverflow.com/a/51014267/397334
  //  - QUESTION Avoid blocking the main queue on long spectro operations?
  //    - https://facebook.github.io/react-native/docs/native-modules-ios#threading
  @objc static func requiresMainQueueSetup() -> Bool {
    return false
  }

  // Static constants exported to js once at init time (e.g. later changes will be ignored)
  //  - https://facebook.github.io/react-native/docs/native-modules-ios#exporting-constants
  @objc func constantsToExport() -> Dictionary<AnyHashable, Any> {
    return [
      "one": 1,
      "two": 2,
    ]
  }

  @objc func foo(
    _ x: String, y: String, z: NSNumber,
    resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock
  ) -> Void {
    // let b = buboValue;
    // let a = AudioKit.format.sampleRate;
    // resolve("x[\(x)], y[\(y)], z[\(z)], b[\(b)], a[\(a)]")
    resolve(Spectro.foo(x: x, y: y, z: z))
  }

}
