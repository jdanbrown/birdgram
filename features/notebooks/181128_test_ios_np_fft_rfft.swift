// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
import Surge
import SigmaSwiftStatistics
import SwiftyJSON
import Yams
import Bubo

func sig(_ name: String, _ xs: [Float]) {
  print(String(format: "%@ %2d %@", name, xs.count, show(xs, prec: 3)))
}

// Copied from .ipynb
let xs: [Float] = [0.8871240795107513, 0.4524807356734083, 0.7514738977690576, 0.6398369473812076, 0.1048129828019595, 0.9385910487684322, 0.18325794318946964, 0.19693278956005267, 0.12113768111983592, 0.1143597199071067, 0.730643103017661, 0.9718515637928253, 0.41818231476302403, 0.431682819195905, 0.6911312488550043, 0.7615860726520747]
let fs: [Float] = [8.395084947957775, 1.7526119715002084, 0.7403942553394491, 1.543562846158875, 1.04011679342957, 0.7685660050530024, 1.6750287409383486, 0.964201961784352, 0.6195584459042496]

sig("xs      ", xs)
// sig("S.fft   ", Surge.fft(xs)) // XXX Nope
sig("abs_rfft", np.fft.abs_rfft(xs))
sig("fs      ", fs)
sig("=       ", fs)

// // Not sure if dct is helpful
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
