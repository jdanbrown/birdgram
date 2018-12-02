import Foundation
import SigmaSwiftStatistics

public enum Stats {

  public typealias QuantileMethod = (_ data: [Double], _ probability: Double) -> Double?

  // A more basic version of pd.qcut
  public static func quantiles(_ xs: [Float], bins: Int, method: QuantileMethod = Sigma.quantiles.method7) -> [Float] {
    if xs.isEmpty {
      return [] // HACK Better than crashing (e.g. for the common case of logging stats and moving on)
    } else {
      return toFloats(np.linspace(0, 1, bins).map { q in
        method(toDoubles(xs), Double(q))!
      })
    }
  }

}
