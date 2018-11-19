import AudioKit
import Foundation

public let buboValue = "bubo here"

public enum Spectro {

    public static func foo(x: String, y: String, z: NSNumber) -> String {
        let b = buboValue;
        let a = AudioKit.format.sampleRate;
        return "From bubo: x[\(x)], y[\(y)], z[\(z)], b[\(b)], a[\(a)]"
    }

}
