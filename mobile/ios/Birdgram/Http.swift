// (See Birdgram/Spectro.swift)

import Foundation

import Bubo // Before Bubo_Pods imports
import Alamofire
import Promises

@objc(RNHttp)
class RNHttp: RCTEventEmitter, RNProxy {

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

  @objc func httpFetch(
    _ url: String,
    resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock
  ) -> Void {
    withPromiseNoProxyAsync(resolve, reject, "httpFetch") {
      return Birdgram.httpFetch(
        url
      )
    }
  }

}

// HACK Workaround react-native .fetch() failing to set cookies on redirect for https://ebird.org
//  - Problem? https://github.com/facebook/react-native/pull/22178 -- tried manually applying the PR but no luck
//  - Problem? https://github.com/wkh237/react-native-fetch-blob/issues/232 -- maybe due to "missing expiry time on the Set-Cookie"?
//  - Solution: skip react-native .fetch() and shim in Alamofire for ios native
//
// TODO Expose more args
// TODO Add non-String responses (e.g. Data/Json)
//
// TODO Increase timeout (e.g. ebird /barchart/CR failed at default 60s timeout)
//  - https://github.com/Alamofire/Alamofire/issues/1644
//  - https://stackoverflow.com/questions/41803856/set-timeout-in-alamofire
//
public func httpFetch(
  _ url: String
) -> Promise<String> {
  Log.debug(String(format: "httpFetch.request: %@", [
    "url": url,
  ]))
  return Promise { fulfill, reject -> Void in
    (Alamofire
      .request(url)
      .validate() // Map 4xx/5xx to .failure i/o .success
      .responseString { rep in
        Log.info(String(format: "httpFetch.response: %@", [
          "url": url,
          "status": rep.response?.statusCode as Any,
        ]))
        switch rep.result {
          case .success:            fulfill(rep)
          case .failure(let error): reject(error)
        }
      }
    )
  }.then { (rep: DataResponse<String>) -> String in
    guard let value = rep.result.value else { throw AppError("httpFetch: Failed to .responseString: \(rep)") }
    return value
  }
}
