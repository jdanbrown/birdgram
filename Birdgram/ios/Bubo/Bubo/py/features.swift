// Like python features.py

import Foundation
import SigmaSwiftStatistics
import Surge

public typealias Spectro    = scipy.signal.Spectrogram
public typealias Melspectro = Spectro

// public func Melspectro(...) -> Melspectro {
//   ...
// }

// public func Spectro(...) -> Spectro {
//   ...
// }

public enum SpectroLike {

  // Normalize by RMS (like "normalize a spectro by its RMS energy" from [SP14])
  //  - Like SpectroLike.norm_rms
  public static func norm_rms(_ S: Matrix<Float>) -> Matrix<Float> {
    return S / sqrt(mean(S.grid ** 2))
  }

  // For each freq bin (row), subtract the median and then zero out negative values
  //  - Like SpectroLike.clip_below_median_per_freq
  public static func clip_below_median_per_freq(_ S: Matrix<Float>) -> Matrix<Float> {
    let S_demedianed = S.mapRows { fs -> [Float] in // (.mapRows i/o Matrix(contents) to preserve rows=0/columns=0)
      let median = fs.count > 0 ? Sigma.median(fs.map { Double($0) })! : .nan
      return fs - Float(median)
    }
    return np.clip(S_demedianed, 0, nil)
  }

}
