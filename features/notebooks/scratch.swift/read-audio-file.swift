// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
// NOTE Must afterwards import other stuff like AudioKit, else "no such module" [why?]
import Surge; import SigmaSwiftStatistics; import SwiftyJSON; import Yams; import Bubo; import AudioKit

for path in [
  "/users/danb/Desktop/rec-v2-2018-11-22T02-08-51-182Z-136b2e72.wav", // Uncompressed, 1ch
  "/users/danb/Desktop/rec-v2-2018-11-21T07-41-18-480Z-dbabcb08.mp4", // Compressed, 2ch
] {
  print(path)

  // Inspect AKAudioFile / AVAudioFile
  let file = try AKAudioFile(forReading: URL(fileURLWithPath: path))
  print(String(format: "%@", [
    "file.url: \(file.url)",
    "file.length: \(file.length)",
    // "file.fileFormat: \(file.fileFormat)",
    // "file.fileFormat.channelCount: \(file.fileFormat.channelCount)",
    // "file.fileFormat.channelLayout: \(file.fileFormat.channelLayout)",
    // "file.fileFormat.sampleRate: \(file.fileFormat.sampleRate)",
    // "file.fileFormat.formatDescription: \(file.fileFormat.formatDescription)",
    // "file.fileFormat.isStandard: \(file.fileFormat.isStandard)",
    // "file.fileFormat.isInterleaved: \(file.fileFormat.isInterleaved)",
    // "file.fileFormat.settings: \(file.fileFormat.settings)",
    // "file.fileFormat.magicCookie: \(file.fileFormat.magicCookie)",
    "file.processingFormat: \(file.processingFormat)",
    "file.processingFormat.channelCount: \(file.processingFormat.channelCount)",
    // "file.processingFormat.channelLayout: \(file.processingFormat.channelLayout)",
    "file.processingFormat.sampleRate: \(file.processingFormat.sampleRate)",
    // "file.processingFormat.formatDescription: \(file.processingFormat.formatDescription)",
    // "file.processingFormat.isStandard: \(file.processingFormat.isStandard)",
    // "file.processingFormat.isInterleaved: \(file.processingFormat.isInterleaved)",
    // "file.processingFormat.settings: \(String(format: "%@", file.processingFormat.settings))",
    // "file.processingFormat.magicCookie: \(file.processingFormat.magicCookie)",
    "file.maxLevel: \(file.maxLevel)",
  ].map{"  \($0)"}.joined(separator: "\n")))

  // [Not sure when this would happen]
  guard let floatChannelData = file.floatChannelData else {
    print("  ERROR: No floatChannelData")
    continue
  }

  // Inspect samples
  let Samples = Matrix(floatChannelData)
  let samples = transpose(Samples).map { mean($0) } // Convert to 1ch
  let quantiles = Stats.quantiles(samples, bins: 5)
  print(String(format: "%@", [
    "Samples.shape: \(Samples.shape)",
    "samples.count: \(samples.count)",
    "samples: \(samples.slice(to: 8))",
    "samples.quantiles: \(quantiles)",
  ].map{"  \($0)"}.joined(separator: "\n")))

  print()
}
