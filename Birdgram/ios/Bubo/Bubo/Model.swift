// Like python sp14/model.py (+ sp14/features.py, for now)

// import AudioKit // XXX Failing to build for macos
import Foundation
import SigmaSwiftStatistics
import Surge

public enum Features {

  // Like Features._spectro_nocache
  public static func spectro(
    _ xs:        [Float],
    sample_rate: Int
  ) -> Melspectro {

    // Like Features config
    assert(sample_rate == 22050, "Spectro params are hardcoded for sampleRate=22050, got: \(sample_rate)")
    let f_bins        = 40
    let hop_length    = 256 // (12ms @ 22050hz)
    let frame_length  = 512 // (23ms @ 22050hz)
    let frame_window  = "hann"
    // let patch_length  = 4   // (46ms @ 22050hz) // For model predict
    // Like Features._spectro_nocache
    let denoise       = true
    let nperseg       = frame_length
    let overlap       = 1 - Double(hop_length) / Double(frame_length)
    let window        = frame_window
    let n_mels        = f_bins
    // Like Melspectro
    // let mels_div      = 2 // Overridden by n_mels
    let scaling       = "spectrum"  // Return units X**2 ('spectrum'), not units X**2/Hz ('density')
    let mode          = "magnitude" // Return |STFT(x)**2|, not STFT(x)**2 (because "humans can't hear complex phase")

    // STFT(xs)
    //  - TODO Compute fs/ts in scipy.signal.spectrogram
    var (_fs, _ts, S) = scipy.signal.spectrogram(
      xs,
      sample_rate: sample_rate,
      window:      window,
      nperseg:     nperseg,
      noverlap:    Int(overlap * Double(nperseg)),
      scaling:     scaling,
      mode:        mode
    )

    // HACK Apply unknown transforms to match librosa.feature.melspectrogram
    //  - Like Melspectro
    S = Float(nperseg / 2) * S // No leads on this one...
    S = S**2                   // Like energy->power, but spectro already gives us power instead of energy...

    // Linear freq -> mel-scale freq
    //  - Like Melspectro
    let mel_basis = librosa.filters.mel(sample_rate: sample_rate, n_fft: nperseg, n_mels: n_mels)
    S = mel_basis * S

    // Linear power -> log power
    //  - Like Melspectro
    S = librosa.power_to_db(S)

    // Mel-scale fs to match S[i]
    //  - Like Melspectro
    // fs = librosa.mel_frequencies(n_mels, min(fs), max(fs)) // TODO Need fs from scipy.signal.spectrogram

    // Denoise
    //  - Like Features._spectro_denoise
    if denoise {
      S = _spectro_denoise(S)
    }

    return (_fs, _ts, S)

  }

  public static func _spectro_denoise(_ S: Matrix<Float>) -> Matrix<Float> {
    return clip_below_median_per_freq(norm_rms(S))
  }

}

//
// TODO -> Features.swift
//

public typealias Melspectro = ([Float], [Float], Matrix<Float>)
public typealias Spectro    = ([Float], [Float], Matrix<Float>)

// public func Melspectro(...) -> Melspectro {
//   ...
// }

// public func Spectro(...) -> Spectro {
//   ...
// }

// Normalize by RMS (like "normalize a spectro by its RMS energy" from [SP14])
//  - Like SpectroLike.norm_rms
public func norm_rms(_ S: Matrix<Float>) -> Matrix<Float> {
  return S / sqrt(mean(S.grid ** 2))
}

// For each freq bin (row), subtract the median and then zero out negative values
//  - Like SpectroLike.clip_below_median_per_freq
public func clip_below_median_per_freq(_ S: Matrix<Float>) -> Matrix<Float> {
  let fss = Array(S) // freqs = rows = Matrix.Iterator
  let S_demedianed = Matrix(fss.map { fs -> [Float] in
    if let median = Sigma.median(fs.map { Double($0) }) {
      return fs - Float(median)
    } else {
      assert(fs.count == 0)
      return []
    }
  })
  return np.clip(S_demedianed, 0, nil)
}
