import Foundation

// Based on https://gist.github.com/nicklockwood/c5f075dd9e714ad8524859c6bd46069f
public enum AppError: Error, CustomStringConvertible {

  case message(String)
  case generic(Error)

  public init(_ message: String) {
    self = .message(message)
  }

  public init(_ error: Error) {
    if let error = error as? AppError {
      self = error
    } else {
      self = .generic(error)
    }
  }

  public var description: String {
    switch self {
    case let AppError.message(message):
      return message
    case let AppError.generic(error):
      return (error as CustomStringConvertible).description
    }
  }

}

// Generic <X> i/o Never because Never isn't bottom [https://forums.swift.org/t/pitch-never-as-a-bottom-type/5920]
public func throw_<X>(_ e: Error) throws -> X {
  throw e
}

public func checkStatus(_ status: OSStatus) throws -> Void {
  if (status != 0) {
    throw NSError(domain: NSOSStatusErrorDomain, code: Int(status))
  }
}

public func pathDirname(_ path: String) -> String {
  return (path as NSString).deletingLastPathComponent
}

public func pathBasename(_ path: String) -> String {
  return (path as NSString).lastPathComponent
}

public func pathSplitExt(_ path: String) -> (String, String) {
  return (
    (path as NSString).deletingPathExtension,
    (path as NSString).pathExtension
  )
}

public func documentsDirectory() -> String {
  // https://stackoverflow.com/questions/24055146/how-to-find-nsdocumentdirectory-in-swift
  return NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
}
