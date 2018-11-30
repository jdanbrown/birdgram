import Foundation
import Surge

// Functions ported from python scipy
//  - Based on https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/spectral.py#L563
//  - Based on https://github.com/tensorflow/magenta-js/blob/53a6cdd/music/src/transcription/audio_utils.ts
public enum scipy {

  public enum signal {

    public typealias Spectrogram = (
      f: [Float],
      t: [Float],
      S: Matrix<Float>
    )

    // Compute a spectrogram with consecutive Fourier transforms
    //  - https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/spectral.py#L563
    //  - Perf: if we need to optimize, consider dropping a level of abstraction to eliminate lots of memcpy's, e.g.
    //    - https://forum.openframeworks.cc/t/a-guide-to-speeding-up-your-of-app-with-accelerate-osx-ios/10560
    //  - TODO Should we pad?
    //    - Python Spectro doesn't pad (scipy.signal.spectrogram -> _spectral_helper(boundary=None, padded=False))
    //    - But maybe helpful for the many short streaming spectros in RecordScreen?
    //    - https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/spectral.py#L563
    public static func spectrogram(
      _ xs:            [Float],
      sample_rate:     Int,
      window:          String,
      nperseg:         Int,
      noverlap:        Int,
      detrend:         String = "constant",
      return_onesided: Bool   = true,
      scaling:         String = "density",
      mode:            String = "psd"
    ) -> Spectrogram {
      precondition(window          == "hann",      "Hardcoded impl")
      precondition(detrend         == "constant",  "Hardcoded impl")
      precondition(return_onesided == true,        "Hardcoded impl")
      precondition(scaling         == "spectrum",  "Hardcoded impl")
      precondition(mode            == "magnitude", "Hardcoded impl")

      let hop_length = noverlap
      let win        = hann(nperseg, sym: false)
      let f          = Int(nperseg / 2) + 1
      let t          = xs.count < nperseg ? 0 : (xs.count - nperseg) / hop_length + 1 // Round down b/c no padding
      let detrend    = { (xs: [Float]) -> [Float] in xs - mean(xs) } // detrend="constant"
      let scale      = sqrt(1.0 / sum(win)**2)                       // scaling="spectrum"

      // Compute S_cols (freqs per time segment)
      var S_cols: [[Float]] = []
      if xs.count >= nperseg {
        S_cols.reserveCapacity(t)
        for i in stride(from: 0, through: xs.count - nperseg, by: hop_length) {
          let (start, end) = (i, i + nperseg)
          assert(end <= xs.count)          // No padding
          var seg = xs.slice(from: start, to: end) // Slice segment from xs (nperseg)
          // if i == 0 { sig("strid", seg) } // XXX Debug
          seg     = detrend(seg)           // Detrend
          // if i == 0 { sig("detre", seg) } // XXX Debug
          seg     = seg .* win             // Apply window (elem-wise multiplication)
          // if i == 0 { sig("windo", seg) } // XXX Debug
          var fs  = np.fft.abs_rfft(seg)   // Compute abs(rfft) (after window)
          // if i == 0 { sig("abfft", fs) } // XXX Debug
          fs      = fs * scale             // Scale (commutes with abs)
          // if i == 0 { sig("scale", fs) } // XXX Debug
          S_cols.append(fs)
          assert(fs.count == f)
        }
      }

      // Join S_cols into S matrix (freq x time)
      //  - transpose(Matrix(...)) for vectorized vDSP_mtrans, i/o trying to zip cols->rows in swift
      let S = transpose(Matrix(
        rows:    t,
        columns: f,
        grid:    Array(S_cols.joined())
      ))

      // TODO Compute freq/time labels (not yet needed for mobile)
      let (fs, ts) = ([Float](), [Float]())

      // Checks to match scipy.signal.spectrogram
      assert(S.shape == (f, t), "S.shape[\(S.shape)] == (f,t)[\(f),\(t)]")

      return (fs, ts, S)

    }

    // A Hann window
    //  - https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/windows/windows.py#L708
    //  - https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
    //  - TODO Simplify with vDSP_hann_window? https://developer.apple.com/documentation/accelerate/1450263-vdsp_hann_window
    public static func hann(_ n: Int, sym: Bool = true) -> [Float] {
      if n == 1 {
        return [1] // Match scipy.signal.hann
      } else {
        if !sym {
          return hann(n+1, sym: true).slice(from: 0, to: n)
        } else {
          var win = Array<Float>(repeating: Float.nan, count: n)
          for i in 0..<n {
            win[i] = Float(0.5 * (1 - cos(2 * Double.pi * Double(i) / Double(n - 1))))
          }
          return win;
        }
      }
    }

  }

}

// XXX Debug
private func sig(_ name: String, _ xs: [Float], limit: Int? = 7) {
  print(String(format: "%@ %3d %@", name, xs.count, show(xs.slice(to: limit), prec: 3)))
}
