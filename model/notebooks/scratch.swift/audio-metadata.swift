// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
// NOTE Must afterwards import other stuff like AudioKit, else "no such module" [why?]
import Surge; import SigmaSwiftStatistics; import SwiftyJSON; import Yams; import Bubo

import AVFoundation

for (path, outputExt, outputFileType, preset) in [
  // TODO Clean up: everything but path (the input) is obviated by taglib
  // ("/Users/danb/Dropbox/tmp/foo.mp3", "caf", AVFileType.caf, nil), // XXX determineCompatibleFileTypes doesn't include .mp3! (only .caf and .mov)
  // ("/Users/danb/Dropbox/tmp/foo.mp4", "m4a", AVFileType.m4a, nil),
  // ("/Users/danb/Dropbox/tmp/foo.wav", "wav", AVFileType.wav, nil),  // FIXME No metadata in out.*
  // ("/Users/danb/Dropbox/tmp/foo.wav", "caf", AVFileType.caf, nil),  // FIXME No metadata in out.*
  // ("/Users/danb/Dropbox/tmp/foo.wav", "aiff", AVFileType.aiff, nil), // FIXME No metadata in out.*
  // ("/Users/danb/Dropbox/tmp/foo.wav", "aifc", AVFileType.aifc, nil), // FIXME No metadata in out.*
  // ("/Users/danb/Dropbox/tmp/foo.wav", "m4a", AVFileType.m4a, AVAssetExportPresetAppleM4A),  // FIXME No metadata in out.*
  ("/Users/danb/Dropbox/tmp/foo.wav", "wav", AVFileType.wav, nil), // Redo with taglib (ignores: outputExt, outputFileType, preset)
] as [(String, String, AVFileType, String?)] {
  let asset = AVURLAsset(url: URL(fileURLWithPath: path))
  precondition(asset.isReadable, "Asset not readable: \(asset)") // e.g. file not found

  // Inspect foo.*
  print()
  try print(Yaml.dumps([
    asset.url.path: [
      // "asset": str(asset),
      // "asset.tracks": str(asset.tracks),
      // "asset.trackGroups": str(asset.trackGroups),
      // "asset.containsFragments": str(asset.containsFragments),
      "asset.duration": str(CMTimeGetSeconds(asset.duration)),
      "asset.metadata": asset.metadata.map { str($0) },
      "asset.commonMetadata": asset.commonMetadata.map { str($0) },
      // "asset.availableMetadataFormats": str(asset.availableMetadataFormats),
      "asset.metadata(forFormat:)": Dictionary(uniqueKeysWithValues: asset.availableMetadataFormats.map {
        (str($0), asset.metadata(forFormat: $0).map { str($0) })
      }),
    ],
  ]).trim())

  // let outputMetadata = asset.metadata // FIXME Works for mp4, but not wav [XXX Obviated by taglib]
  let title = AVMutableMetadataItem()
  title.identifier = .commonIdentifierTitle
  title.value = "another title" as NSString
  let outputMetadata = [
    title,
  ]

  // Write comment tag back to input file
  //  - [x] Works on mac! [Test tag max lengths on ios, in case that's different than macos...]
  //  - [x] Works on ios! Max lengths are >1e7 on ios!
  //  - [x] ios r/w interops with osx cli taffy r/w + shncat r (yay! of course it should, but yay!)
  print("Read comment: \(try TagLib.readComment(path) as Any)")
  try TagLib.writeComment(path, String(repeating: "here is a new comment / ", count: 100))
  print("Read comment: \(try TagLib.readComment(path) as Any)")

  //
  // XXX Obviated by taglib
  //  - Edits file metadata in place
  //  - No need to transcode input to output
  //  - No need to mess with ios AVFoundation input/output codec restrictions (see AKConverter for a map of how gnarly it is)
  //

  // let outputURL = URL(fileURLWithPath: "\(path).out.\(outputExt)")

  // // Write out.* with different metadata
  // print()
  // print("Export")
  // try print(Yaml.dumps([[
  //   "outputMetadata": outputMetadata.map { str($0) },
  //   "exportPresets(compatibleWith:)": AVAssetExportSession.exportPresets(compatibleWith: asset).map { str($0) }.sorted(),
  // ]]).trim())
  // guard let export = AVAssetExportSession(
  //   asset: asset,
  //   presetName: preset ?? AVAssetExportPresetPassthrough
  // ) else { preconditionFailure("Failed to AVAssetExportSession") }
  // export.outputFileType = outputFileType
  // export.outputURL      = outputURL
  // export.metadata       = outputMetadata
  // try print(Yaml.dumps([[
  //   "export": str(export),
  //   "outputURL": str(outputURL),
  // ]]).trim())
  // let semaphore = DispatchSemaphore(value: 0)
  // if (FileManager.default.fileExists(atPath: outputURL.path)) {
  //   try FileManager.default.removeItem(at: outputURL) // Allow overwrite
  // }
  // export.determineCompatibleFileTypes { filetypes in
  //   try! print(Yaml.dumps([[
  //     "outputFileType": str(outputFileType),
  //     "determineCompatibleFileTypes": filetypes.map { str($0) },
  //   ]]).trim())
  //   print("Export...")
  //   export.exportAsynchronously {
  //     if (export.status != .completed ) {
  //       preconditionFailure("Export failed: \(export.error!)")
  //     } else {
  //       print("Export done")
  //       semaphore.signal()
  //     }
  //   }
  // }
  // semaphore.wait()

  // print("AKConverter...")
  // if (FileManager.default.fileExists(atPath: outputURL.path)) {
  //   try FileManager.default.removeItem(at: outputURL) // Allow overwrite
  // }
  // let semaphore = DispatchSemaphore(value: 0)
  // AKConverter(
  //   inputURL: asset.url,
  //   outputURL: outputURL,
  //   options: AKConverter.Options()
  // ).start { error in
  //   if let error = error {
  //     preconditionFailure("Convert failed: \(error)")
  //   } else {
  //     print("Export done")
  //     semaphore.signal()
  //   }
  // }
  // semaphore.wait()

  // // Inspect out.*
  // let out = AVURLAsset(url: outputURL)
  // print()
  // try print(Yaml.dumps([
  //   out.url.path: [
  //     "out.duration": str(CMTimeGetSeconds(out.duration)),
  //     "out.metadata": out.metadata.map { str($0) },
  //     "out.commonMetadata": out.commonMetadata.map { str($0) },
  //     "out.metadata(forFormat:)": Dictionary(uniqueKeysWithValues: out.availableMetadataFormats.map {
  //       (str($0), out.metadata(forFormat: $0).map { str($0) })
  //     }),
  //   ],
  // ]).trim())

  // // get comment
  // print()
  // do {
  //   let file: UnsafeMutablePointer<TagLib_File> = taglib_file_new(path)
  //   if (file != nil && taglib_file_is_valid(file) != 0) {
  //     let tag: UnsafeMutablePointer<TagLib_Tag> = taglib_file_tag(file)
  //     let val: UnsafeMutablePointer<Int8>       = taglib_tag_comment(tag)
  //     let str = NSString(utf8String: val)
  //     print("str[comment]: \(str)")
  //     taglib_free(val)
  //   }
  //   taglib_file_free(file)
  // }

  // // set comment
  // print()
  // do {
  //   let comment = "we set some other comment"
  //   let file: UnsafeMutablePointer<TagLib_File> = taglib_file_new(path)
  //   if (file != nil && taglib_file_is_valid(file) != 0) {
  //     let tag: UnsafeMutablePointer<TagLib_Tag> = taglib_file_tag(file)
  //     comment.withCString { taglib_tag_set_comment(tag, $0) }
  //   }
  //   taglib_file_save(file)
  //   taglib_file_free(file)
  //   print("set: comment")
  // }

}
