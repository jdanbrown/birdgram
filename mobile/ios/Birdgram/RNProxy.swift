import Foundation

import Bubo // Before Bubo_Pods imports
import Promises

protocol RNProxy {

  associatedtype Proxy
  var proxy: Proxy? { get }

}

extension RNProxy {

  // TODO Simplify
  func withPromiseNoProxy<X>(
    _ resolve: @escaping RCTPromiseResolveBlock,
    _ reject: @escaping RCTPromiseRejectBlock,
    _ name: String,
    _ f: @escaping () throws -> X
  ) -> Void {
    withPromiseNoProxyAsync(resolve, reject, name) { () -> Promise<X> in
      return Promise { () -> X in
        return try f()
      }
    }
  }

  // TODO Simplify
  func withPromiseNoProxyAsync<X>(
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

  // TODO Simplify
  func withPromise<X>(
    _ resolve: @escaping RCTPromiseResolveBlock,
    _ reject: @escaping RCTPromiseRejectBlock,
    _ name: String,
    _ f: @escaping (Proxy) throws -> X
  ) -> Void {
    withPromiseAsync(resolve, reject, name) { (_proxy: Proxy) -> Promise<X> in
      return Promise { () -> X in
        return try f(_proxy)
      }
    }
  }

  // TODO Simplify
  func withPromiseAsync<X>(
    _ resolve: @escaping RCTPromiseResolveBlock,
    _ reject: @escaping RCTPromiseRejectBlock,
    _ name: String,
    _ f: (Proxy) -> Promise<X>
  ) -> Void {
    if let _proxy = proxy {
      f(_proxy).then { x in
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
    } else {
      let error = AppError("proxy=nil")
      let method = "\(type(of: self)).\(name)"
      let stack = Thread.callStackSymbols // TODO How to get stack from error i/o current frame? (which is doubly useless in async)
      reject(
        "\(method)",
        "method[\(method)] error[\(error)] stack[\n\(stack.joined(separator: "\n"))\n]",
        error
      )
    }
  }

}
