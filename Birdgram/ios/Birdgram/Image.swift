import Foundation

import Bubo // Before Bubo_Pods imports
import Surge

// Assumes X values are in [0,1), for mapping to color scales
public func matrixToImageFile(
  _ path: String,
  _ X: Matrix<Float>,
  colors: Colormap,
  bottomUp: Bool? = nil
) throws -> (
  width: Int,
  height: Int
) {
  var debugTimes = [(String, Double)]()
  return try matrixToImageFile(
    path,
    X,
    colors: colors,
    bottomUp: bottomUp,
    timer: Timer(), debugTimes: &debugTimes // XXX Debug (For Features.spectro)
  )
}

// Assumes X values are in [0,1), for mapping to color scales
public func matrixToImageFile(
  _ path: String,
  _ X: Matrix<Float>,
  colors: Colormap,
  bottomUp: Bool? = nil,
  timer: Bubo.Timer, debugTimes: inout Array<(String, Double)> // XXX Debug (for Features.spectro)
) throws -> (
  width: Int,
  height: Int
) {
  precondition(!X.isEmpty, "matrixToImageFile: X must be nonempty (for path[\(path)])")

  // X -> pixels
  let P            = !(bottomUp ?? true) ? X : X.flipVertically()
  let height       = P.rows
  let width        = P.columns
  let pxF: [Float] = P.grid // .grid is row major
  let pxI: [UInt8] = pxF.map     { x in UInt8((x * 256).clamped(0, 255)) }
  var pxB: [UInt8] = pxI.flatMap { i in colors[Int(i)].bytes }
  Log.trace(String(format: "Spectro.onAudioData: pxB[%d]: %@", pxB.count, show(pxB.slice(to: 20)))) // XXX Debug [XXX Bottleneck]
  debugTimes.append(("pxB", timer.lap()))

  // Pixels -> .png file
  if let image = ImageHelper.convertBitmapRGBA8(
    toUIImage: &pxB,
    withWidth: Int32(width),
    withHeight: Int32(height),
    grayscale: false
  ) {
    if let pngData = image.pngData() {
      do {
        try pngData.write(to: URL(fileURLWithPath: path))
      } catch {
        Log.error("Spectro.onAudioData: Failed to pngData.write(): \(error)")
      }
    } else {
      Log.error("Spectro.onAudioData: Failed to image.pngData()")
    }
  } else {
    Log.error("Spectro.onAudioData: Failed to ImageHelper.convertBitmapRGBA8")
  }
  debugTimes.append(("img", timer.lap()))

  return (width: width, height: height)

}
