import AudioKit
import Foundation
import Surge

public let buboValue = "bubo here"

public enum SpectroImpl {

  public static func hello(_ x: String, _ y: String, _ z: NSNumber) -> String {
    let b = buboValue;
    let a = AudioKit.format.sampleRate;
    return "Hello from Bubo: x[\(x)], y[\(y)], z[\(z)], b[\(b)], a[\(a)]"
  }

}
