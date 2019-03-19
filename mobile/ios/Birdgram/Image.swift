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
