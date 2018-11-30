import Foundation
import Surge

// Functions ported from python librosa (0.6.1)
//  - https://github.com/librosa/librosa
public enum librosa {

  public enum filters {

    // Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
    //  - https://github.com/librosa/librosa/blob/0.6.1/librosa/filters.py#L180
    public static func mel(
      _ sr:   Int, // (sample_rate -- keep names consistent with py librosa)
      n_fft:  Int,
      n_mels: Int  = 128,
      htk:    Bool = false // Implementation assumes false
    ) -> Matrix<Float> {
      let sample_rate = sr
      precondition(!htk, "Implementation assumes htk=false")

      let fmin = Float(0.0)
      let fmax = Float(sample_rate) / 2

      // Initialize the weights
      var weights = Matrix<Float>(
        rows:          n_mels,
        columns:       1 + n_fft / 2,
        repeatedValue: 0
      )

      // Center freqs of each FFT bin
      let fftfreqs = fft_frequencies(sample_rate, n_fft: n_fft)

      // 'Center freqs' of mel bands - uniformly spaced between limits
      let mel_f = mel_frequencies(n_mels + 2, fmin, fmax, htk: htk)

      let fdiff = np.diff(mel_f)
      let ramps = np.subtract_outer(mel_f, fftfreqs)

      for i in 0..<n_mels {
        // lower and upper slopes for all bins
        let lower = ramps[row: i]   / fdiff[i] * -1
        let upper = ramps[row: i+2] / fdiff[i+1]
        // ... then intersect them with each other and zero
        weights[row: i] = np.maximum(np.minimum(lower, upper), 0)
      }

      // Slaney-style mel is scaled to be approx constant energy per channel
      let enorm = 2.0 / (mel_f.slice(from: 2, to: 2+n_mels) .- mel_f.slice(from: 0, to: n_mels))
      weights = elmul(weights, np.broadcast_to(column: enorm, weights.shape))

      return weights

    }

  }

  // Alternative implementation of `np.fft.fftfreqs`
  //  - https://github.com/librosa/librosa/blob/0.6.1/librosa/core/time_frequency.py#L758
  public static func fft_frequencies(
    _ sr:  Int = 22050, // (sample_rate -- keep names consistent with py librosa)
    n_fft: Int = 2048
  ) -> [Float] {
    let sample_rate = sr
    return np.linspace(
      0,
      Float(sample_rate) / 2,
      1 + n_fft / 2
    )
  }

  // Compute an array of acoustic frequencies tuned to the mel scale
  //  - https://github.com/librosa/librosa/blob/0.6.1/librosa/core/time_frequency.py#L828
  public static func mel_frequencies(
    _ n_mels: Int   = 128,
    _ fmin:   Float = 0.0,
    _ fmax:   Float = 11025.0,
    htk:      Bool  = false
  ) -> [Float] {
    precondition(!htk, "Implementation assumes htk=false")
    let min_mel = hz_to_mel(fmin)
    let max_mel = hz_to_mel(fmax)
    let mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels)
  }

  // Convert Hz to Mels (scalar)
  //  - https://github.com/librosa/librosa/blob/0.6.1/librosa/core/time_frequency.py#L589
  //  - TODO [Float] (in addition to Float)
  public static func hz_to_mel(_ freq: Float) -> Float {

    // The linear part
    let f_min = Float(0.0)
    let f_sp  = Float(200.0 / 3)
    var mels  = (freq - f_min) / f_sp

    // The log part
    let min_log_hz  = Float(1000.0)
    let min_log_mel = (min_log_hz - f_min) / f_sp
    let logstep     = Float(log(6.4) / 27.0)
    if freq >= min_log_hz {
      mels = min_log_mel + log(freq / min_log_hz) / logstep
    }

    return mels

  }

  // Convert Mels to Hz (vector)
  //  - https://github.com/librosa/librosa/blob/0.6.1/librosa/core/time_frequency.py#L644
  //  - TODO Float (in addition to [Float])
  public static func mel_to_hz(_ mels: [Float]) -> [Float] {
    // Avoid losing precision
    //  - TODO Probably bigger numerical stability issues, given tol:1e-5 in the tests...
    let mels_d = mels.map { Double($0) }

    // The linear part
    let f_min = Double(0.0)
    let f_sp  = Double(200.0 / 3)
    var freqs = mels_d * f_sp + f_min

    // The log part
    let min_log_hz  = Double(1000.0)
    let min_log_mel = (min_log_hz - f_min) / f_sp
    let logstep     = Double(log(6.4) / 27.0)
    for i in 0..<freqs.count {
      if mels_d[i] >= min_log_mel {
        freqs[i] = min_log_hz * exp(logstep * (mels_d[i] - min_log_mel))
      }
    }

    return freqs.map { Float($0) }

  }

  // Convert a power spectrogram (amplitude squared) to decibel (dB) units
  //  - Computes the scaling `10 * log10(S / ref)` in a numerically stable way
  //  - https://github.com/librosa/librosa/blob/0.6.1/librosa/core/spectrum.py#L766
  public static func power_to_db(
    _ S:    Matrix<Float>,
    ref:    Float         = 1.0,
    amin:   Float         = 1e-10,
    top_db: Float?        = 80.0
  ) -> Matrix<Float> {
    precondition(amin > 0)
    let magnitude = S
    let ref_value = abs(ref)
    var log_spec  = 10.0 * magnitude.vect { log10(np.maximum($0, amin)) }
    log_spec      = log_spec.vect { $0 - 10.0 * log10(max(amin, ref_value)) }
    if let _top_db = top_db {
      precondition(_top_db >= 0)
      log_spec = log_spec.vect { np.maximum($0, max($0) - _top_db) }
    }
    return log_spec
  }

}
