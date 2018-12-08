// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
import Surge
import SigmaSwiftStatistics
import SwiftyJSON
import Yams
import Bubo

func sig(_ name: String, _ xs: [Float], limit: Int? = 7) {
  print(String(format: "%@ %3d %@", name, xs.count, show(xs.slice(to: limit), prec: 3)))
}

func mat(_ name: String, _ X: Matrix<Float>) {
  print(String(format: "%@ %@\n%@", name, String(describing: X.shape), show(X, prec: 3)))
}

//
// np.zeros
//

testEqual("np.zeros", np.zeros(0), [])
testEqual("np.zeros", np.zeros(3), [0,0,0])

//
// np.linspace
//

testEqual("np.linspace", np.linspace(0, 1, 5), [0, 0.25, 0.5, 0.75, 1])

//
// np.diff
//

testEqual("np.diff", np.diff([]), [])
testEqual("np.diff", np.diff([1, 4, 6, 2, 8]), [3, 2, -4, 6])

//
// np.subtract_outer
//

testEqual("np.op_outer", np.op_outer([2,3,5], [7,9]) { $0 * $1 }, Matrix([
  [14, 18],
  [21, 27],
  [35, 45],
]))

//
// np.clip
//

do {
  let xs: [Float] = [1,2,3]
  testEqual("np.clip", np.clip(xs, 0, 10), [1,2,3])
  testEqual("np.clip", np.clip(xs, nil, nil), [1,2,3])
  testEqual("np.clip", np.clip(xs, 0, 1), [1,1,1])
  testEqual("np.clip", np.clip(xs, nil, 1), [1,1,1])
  testEqual("np.clip", np.clip(xs, 2, nil), [2,2,3])
  testEqual("np.clip", np.clip(xs, 2, 2.5), [2,2,2.5])
}

//
// np.minimum, np.maximum
//

do {
  let xs: [Float] = [2,3,4]
  let ys: [Float] = [1,5,2]
  testEqual("np.minimum", np.minimum(xs, ys), [1, 3, 2])
  testEqual("np.maximum", np.maximum(xs, ys), [2, 5, 4])
  testEqual("np.minimum", np.minimum(xs, 2.5), [2, 2.5, 2.5])
  testEqual("np.maximum", np.maximum(xs, 2.5), [2.5, 3, 4])
}

//
// np.broadcast_to
//

testEqual("np.broadcast_to", np.broadcast_to(row: [1,2,3], (2,3)), Matrix([
  [1,2,3],
  [1,2,3],
]))
testEqual("np.broadcast_to", np.broadcast_to(column: [1,2,3], (3,2)), Matrix([
  [1,1],
  [2,2],
  [3,3],
]))

//
// np.random.rand
//

testEqual("np.random.rand", np.random.rand(5).count, 5)

//
// np.fft.abs_rfft
//

// XXX vDSP requires n≥16
// testAlmostEqual("np.fft.abs_rfft", [1],       np.fft.abs_rfft([1]))
// testAlmostEqual("np.fft.abs_rfft", [1,1],     np.fft.abs_rfft([2,0]))
// testAlmostEqual("np.fft.abs_rfft", [1,1,1],   np.fft.abs_rfft([3,0]))
// testAlmostEqual("np.fft.abs_rfft", [1,1,1,1], np.fft.abs_rfft([4,0,0]))

do {

  // Copied from .ipynb
  let xs: [Float] = [0.8871240795107513, 0.4524807356734083, 0.7514738977690576, 0.6398369473812076, 0.1048129828019595, 0.9385910487684322, 0.18325794318946964, 0.19693278956005267, 0.12113768111983592, 0.1143597199071067, 0.730643103017661, 0.9718515637928253, 0.41818231476302403, 0.431682819195905, 0.6911312488550043, 0.7615860726520747]
  let fs: [Float] = [8.395084947957775, 1.7526119715002084, 0.7403942553394491, 1.543562846158875, 1.04011679342957, 0.7685660050530024, 1.6750287409383486, 0.964201961784352, 0.6195584459042496]

  // // Debug
  // sig("xs      ", xs)
  // sig("abs_rfft", np.fft.abs_rfft(xs))
  // sig("fs      ", fs)
  // sig("Δ       ", np.fft.abs_rfft(xs) .- fs)
  // sig("log Δ   ", np.log10(np.fft.abs_rfft(xs) .- fs))

  testAlmostEqual("np.fft.abs_rfft", fs, np.fft.abs_rfft(xs), tol: 1e-6)

}

//
// np.fft.dct
//

// // XXX Not clear if dct is helpful
// print()
// sig("dct2   ", np.fft.dct(xs, .II))
// // Q: dct2 looks vaguely close -- is it a shortcut? [XXX Nope]
// let dct2  = np.fft.dct(xs, .II)
// let dct20 = Array(stride(from: 0, to: dct2.count, by: 2).map { dct2[$0] })
// let dct21 = Array(stride(from: 1, to: dct2.count, by: 2).map { dct2[$0] })
// let nope  = sqrt(pow(dct20, 2) .+ pow(dct21, 2))
// sig(" [nope]", nope)
// sig("dct3   ", np.fft.dct(xs, .III))
// sig("dct4   ", np.fft.dct(xs, .IV))
