// Like python sp14/model.py (+ sp14/features.py, for now)

// import AudioKit // XXX Failing to build for macos
import Foundation
import SigmaSwiftStatistics
import Surge

public class Features {
  private init() {} // (Disallow instances to enforce class singleton via statics)

  // Like Features config
  static let sample_rate   = 22050
  static let f_bins        = 40
  static let hop_length    = 256 // (12ms @ 22050hz)
  static let frame_length  = 512 // (23ms @ 22050hz)
  static let frame_window  = "hann"
  // static let patch_length  = 4   // (46ms @ 22050hz) // For model predict
  // Like Features._spectro_nocache
  static let nperseg       = frame_length
  static let overlap       = 1 - Double(hop_length) / Double(frame_length)
  static let window        = frame_window
  static let n_mels        = f_bins

  // Extracted as constant to remove bottleneck in spectro()
  static let mel_basis = librosa.filters.mel(sample_rate, n_fft: nperseg, n_mels: n_mels)

  // Like Features._spectro_nocache_from_audio
  public static func spectro(
    _ xs:        [Float],
    sample_rate: Int,
    denoise:     Bool = true
  ) -> Melspectro {
    assert(sample_rate == Features.sample_rate, "Impl hardcoded for sampleRate=\(Features.sample_rate), got: \(sample_rate)")
    // let timer = Timer() // XXX Perf

    // Like Melspectro
    // let mels_div      = 2 // Overridden by n_mels
    let scaling       = "spectrum"  // Return units X**2 ('spectrum'), not units X**2/Hz ('density')
    let mode          = "magnitude" // Return |STFT(x)**2|, not STFT(x)**2 (because "humans can't hear complex phase")
    // print(String(format: "[time] Features.spectro: config: %d", Int(1000 * timer.lap()))) // XXX Perf

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
    // print(String(format: "Features.spectro: scipy.signal.spectrogram: %@", show(["S.shape": S.shape]))) // XXX Debug
    // print(String(format: "[time] Features.spectro: scipy.signal.spectrogram: %d", Int(1000 * timer.lap()))) // XXX Perf

    // HACK Apply unknown transforms to match librosa.feature.melspectrogram
    //  - Like Melspectro
    S = Float(nperseg / 2) * S // No leads on this one...
    S = S**2                   // Like energy->power, but spectro already gives us power instead of energy...
    // print(String(format: "Features.spectro: unknown transforms: %@", show(["S.shape": S.shape]))) // XXX Debug
    // print(String(format: "[time] Features.spectro: unknown transforms: %d", Int(1000 * timer.lap()))) // XXX Perf

    // Linear freq -> mel-scale freq
    //  - Like Melspectro
    S = mel_basis * S
    // print(String(format: "Features.spectro: mel_basis: %@", show(["S.shape": S.shape]))) // XXX Debug
    // print(String(format: "[time] Features.spectro: mel_basis: %d", Int(1000 * timer.lap()))) // XXX Perf

    // Linear power -> log power
    //  - Like Melspectro
    S = librosa.power_to_db(S)
    // print(String(format: "Features.spectro: librosa.power_to_db: %@", show(["S.shape": S.shape]))) // XXX Debug
    // print(String(format: "[time] Features.spectro: librosa.power_to_db: %d", Int(1000 * timer.lap()))) // XXX Perf

    // [NOTE fs currently unused in mobile]
    // Mel-scale fs to match S[i]
    //  - Like Melspectro
    //  - TODO Blocked on non-mock fs from scipy.signal.spectrogram
    // fs = librosa.mel_frequencies(n_mels, min(fs), max(fs))

    // Denoise
    //  - Like Features._spectro_denoise
    if denoise {
      S = _spectro_denoise(S)
    }
    // print(String(format: "Features.spectro: _spectro_denoise: %@", show(["S.shape": S.shape]))) // XXX Debug
    // print(String(format: "[time] Features.spectro: _spectro_denoise: %d", Int(1000 * timer.lap()))) // XXX Perf

    return (_fs, _ts, S)

  }

  public static func _spectro_denoise(_ S: Matrix<Float>) -> Matrix<Float> {
    return SpectroLike.clip_below_median_per_freq(
      SpectroLike.norm_rms(S)
    )
  }

}

//
// TODO -> Features.swift
//

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
    let S_demedianed = S.mapRows { fs in // (.mapRows i/o Matrix(contents) to preserve rows=0/columns=0)
      let median = fs.count > 0 ? Sigma.median(fs.map { Double($0) })! : .nan
      return fs - Float(median)
    }
    return np.clip(S_demedianed, 0, nil)
  }

}