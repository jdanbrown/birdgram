import Foundation
import UIKit

import Bubo // Before Bubo_Pods imports
import Surge

extension UIImage {

  public func mapCGImage(f: (CGImage) -> CGImage?) -> UIImage? {
    guard let a = cgImage else { return nil }
    guard let b = f(a)    else { return nil }
    return UIImage(cgImage: b)
  }

  public func cropping(to: CGRect) -> UIImage? {
    return mapCGImage { cgImage in cgImage.cropping(to: to) }
  }

}

public typealias ImageFile = (
  path: String,
  width: Int,
  height: Int
)

public func matrixToImageFile(
  _ path: String,
  _ X: Matrix<Float>,
  range: Interval<Float>,
  colors: Colormap,
  bottomUp: Bool? = nil
) throws -> ImageFile {
  var debugTimes = [(String, Double)]()
  return try matrixToImageFile(
    path,
    X,
    range: range,
    colors: colors,
    bottomUp: bottomUp,
    timer: Timer(), debugTimes: &debugTimes // XXX Debug (For Features._spectro)
  )
}

public func matrixToImageFile(
  _ path: String, // (Same as returned ImageFile.path)
  _ X: Matrix<Float>,
  range: Interval<Float>,
  colors: Colormap,
  bottomUp: Bool? = nil,
  timer: Bubo.Timer, debugTimes: inout Array<(String, Double)> // XXX Debug (for Features._spectro)
) throws -> ImageFile {
  precondition(!X.isEmpty, "matrixToImageFile: X must be nonempty (for path[\(path)])")

  // X -> pixels
  let P            = !(bottomUp ?? true) ? X : X.flipVertically()
  let height       = P.rows
  let width        = P.columns
  let pxF: [Float] = P.grid // .grid is row major
  let pxI: [UInt8] = pxF.map     { x in
    UInt8((range.norm(x).isFiniteOr(0) * 256).clamped(0, 255)) // Infinitesimal edge case: force 1->255 i/o 256
  }
  var pxB: [UInt8] = pxI.flatMap { i in
    colors[Int(i)].bytes
  }
  // Log.debug(String(format: "Image.matrixToImageFile: pxB[%d]: %@", pxB.count, show(pxB.slice(to: 20)))) // XXX Debug
  debugTimes.append(("pxB", timer.lap()))

  // Pixels -> .png file
  //  - HACK Skip if file exists (else bottleneck on every RecordScreen load, since we blindly recompute and overwrite)
  if (pathExists(path)) {
    Log.debug(String(format: "matrixToImageFile: File exists, skipping write: %@", [
      "path": path,
      "width": width,
      "height": height,
    ]))
  } else {

    // Pixels -> .png file
    guard let image = ImageHelper.convertBitmapRGBA8(
      toUIImage: &pxB,
      withWidth: Int32(width),
      withHeight: Int32(height),
      grayscale: false
    ) else {
      throw AppError("Failed to ImageHelper.convertBitmapRGBA8")
    }
    guard let pngData = image.pngData() else { throw AppError("Failed to image.pngData") }
    try pngData.write(to: URL(fileURLWithPath: path), options: [.atomic])
    debugTimes.append(("img", timer.lap()))

  }

  return (path: path, width: width, height: height)

}

// FIXME [Perf] Bottleneck for moderately-sized recs (e.g. rec >30s), and appears hung to the user for long recs (e.g. rec ~5-10m)
public func chunkImageFile(
  _ path: String,
  chunkWidth: Int
) throws -> Array<ImageFile> {
  Log.debug(String(format: "chunkImageFile: %@", [
    "path": path,
    "chunkWidth": chunkWidth,
  ]))

  // Load image from path
  //  - [Perf] ~1ms for image.width[5734], chunkWidth[5]
  guard let image = UIImage(contentsOfFile: path) else { throw AppError("Failed to read UIImage from path: \(path)") }
  // Get CGImage from Image
  //  - Use cgImage.{width,height}:Int i/o image.size.{width,height}:CGFloat so we don't have to think about float/int conversion
  guard let cgImage = image.cgImage else { throw AppError("Nil image.cgImage") }
  // Log.debug(String(format: "chunkImageFile: Loaded: %@", [
  //   "path": path,
  //   "chunkWidth": chunkWidth,
  //   "image": image,
  // ]))

  // Chunk image
  //  - [Perf] ~32ms for image.width[5734], chunkWidth[5]
  let chunksDir = "\(path).chunks-chunkWidth=\(chunkWidth)"
  let chunks: Array<(path: String, image: UIImage)> = (
    try stride(from: 0, to: cgImage.width, by: chunkWidth).map { x in
      // WARNING Avoid ':' in ios paths [they map to dir separator, I think?]
      let path = "\(chunksDir)" / "\(x)-\(cgImage.width)-\(chunkWidth).\(pathSplitExt(path).ext)"
      guard let image = image.cropping(to: CGRect(
        x: x,
        y: 0,
        width: min(chunkWidth, cgImage.width - x),
        height: cgImage.height
      )) else {
        throw AppError("Failed to image.cropping (for outgoing path: \(path))")
      }
      return (path: path, image: image)
    }
  )
  // Log.debug(String(format: "chunkImageFile: Chunked: %@", [
  //   "path": path,
  //   "chunkWidth": chunkWidth,
  //   "chunksDir": chunksDir,
  //   "image": image,
  // ]))

  // Write image chunks to files
  //  - HACK Skip if done file exists (else bottleneck on every RecordScreen load, since we bindly recompute and overwrite)
  let chunksDonePath = "\(chunksDir).done"
  if (pathExists(chunksDonePath)) {
    Log.debug(String(format: "chunkImageFile: Writes already done, skipping: %@", [
      "path": path,
      "chunkWidth": chunkWidth,
      "chunksDir": chunksDir,
      "image": image,
    ]))
  } else {

    // Write image chunks to files
    //  - [Perf] ~5500ms for image.width[5734], chunkWidth[5] <- FIXME bottleneck
    try ensureDir(chunksDir)
    try chunks.forEach { (path, image) in
      guard let pngData = image.pngData() else { throw AppError("Failed to image.pngData (for outgoing path: \(path))") }
      try pngData.write(to: URL(fileURLWithPath: path))
    }
    // Log.debug(String(format: "chunkImageFile: Written: %@", [
    //   "path": path,
    //   "chunkWidth": chunkWidth,
    //   "chunksDir": chunksDir,
    //   "image": image,
    // ]))

    // Mark done
    try touchPath(chunksDonePath)
  }

  // Return an ImageFile for each chunk
  //  - [Perf] ~5ms for image.width[5734], chunkWidth[5]
  let imageFiles: Array<ImageFile> = chunks.map { (path, image) in (
    path:   path,
    width:  image.cgImage!.width,
    height: image.cgImage!.height
  )}
  Log.info(String(format: "chunkImageFile: Done: %@", [
    "path": path,
    "chunkWidth": chunkWidth,
    "chunksDir": chunksDir,
    "image": image,
    "imageFiles.count": imageFiles.count,
  ]))
  return imageFiles

}
