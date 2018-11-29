// Run using scheme: Birdgram -> <any>
//  - Key: shift-enter to run current line
//  - Key: shift-cmd-enter to run all lines
//  - Key: ctrl-\ to stop running playground
//  - Key: ctrl-enter to show/hide current result inline [doesn't always work...]

import Bubo // Before pods imports
import AudioKit
import AudioKitUI
//import SwiftyJSON
import Surge
//import Promises

print(np.linspace(0, 10, 5))

let (sample_rate, n_fft) = (22050, 512)
print(np.linspace(0, Float(sample_rate) / 2, 1 + n_fft / 2).count)
print(np.linspace(0, Float(sample_rate) / 2, 1 + n_fft / 2)[250...])

do { let (n,w,h) = (5119, 512, 256); Array(stride(from: 0, to: n - w, by: h)).count }
do { let (n,w,h) = (5120, 512, 256); Array(stride(from: 0, to: n - w, by: h)).count }
do { let (n,w,h) = (5121, 512, 256); Array(stride(from: 0, to: n - w, by: h)).count }

do { let (n,w,h) = (5119, 512, 256); Array(stride(from: 0, through: n - w, by: h)).count }
do { let (n,w,h) = (5120, 512, 256); Array(stride(from: 0, through: n - w, by: h)).count }
do { let (n,w,h) = (5121, 512, 256); Array(stride(from: 0, through: n - w, by: h)).count }

Array(stride(from: 0, to: 5210 - 512, by: 256)).count

3 <= -Float.infinity
3 <= Float.infinity
Float.greatestFiniteMagnitude <= Float.infinity
Float.infinity <= Float.infinity
Float.infinity < Float.infinity
Float.leastNormalMagnitude
Float.leastNonzeroMagnitude

let xs = [0,1,2,3,4,5,6,7,8,9]
let hop_length = 3
print(Array(stride(from: 0, to: xs.count, by: hop_length)))
print(Array(stride(from: 0, to: xs.count - hop_length, by: hop_length)))

let xs: [Float] = [1,2,4,7,0]
let ys: [Float] = Array(xs[1...])
let zs: [Float] = Array(xs[..<(xs.count - 1)])
ys .- zs

print(fft(xs))

exp(2.0)
log(2.0)
log10(2.0)
Float(3.0) / 2
0...5
print(Array(stride(from: 0.0, through: 1.0, by: 1.0/5.0)))

print(Matrix([1,2,3,4].chunked(2)).vop { xs in log10(xs as [Float]) })

Matrix([1,2,3,4].chunked(2)).map { (x: ArraySlice<Double>) in x*2.0 }

max(1,2,3)
abs(-3)
2.0 * Matrix([1,2,3,4].chunked(2))

print(Matrix([1,2,3,4].chunked(2)))
print(Matrix(rows: 0, columns: 0, repeatedValue: 0 as Float))
// Matrix([] as [[Double]])

// Surge
Surge.sum([1,2,3,4] as [Double])
[1.0,2,3] • [4.0,5,6]

// Surge: Arithmetic
let n = [-1.0, 2.0, 3.0, 4.0, 5.0]
let sum = Surge.sum(n)
let a = [1.0, 3.0, 5.0, 7.0]
let b = [2.0, 4.0, 6.0, 8.0]
let product = mul(a, b)

// Surge: Matrix
let A = Matrix([[1, 1], [1, -1]])
let C = Matrix([[3], [1]])
let B = inv(A) * C
print(A, B, C)
print(A * transpose(A))

// Surge: FFT
let count = 64
let frequency = 4.0
let amplitude = 3.0
let x = (0..<count).map { 2.0 * Double.pi / Double(count) * Double($0) * frequency }
print("x", show(x))
print("fft(x)", show(fft(x)))
print("sin(x)", show(sin(x)))
print("fft(sin(x))", show(fft(sin(x))))

////

print(FileManager.default.temporaryDirectory.path)

let xss = [[1,2],[3,4]]
//xss.joined()

URL(fileURLWithPath: "/foo/bar")

print(String(describing: ["a":1, "b":2]))
print(String(format: "%@", ["a":1, "b":2]))

//JSON(3)

print(AKAudioFile().fileFormat)
print(AudioKit.format)
AudioKit.format.sampleRate
AudioKit.format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 22050, channels: 1, interleaved: false)!
AudioKit.format.sampleRate
print(AKSettings.audioFormat)
AKSettings.sampleRate = 22050
AKSettings.channelCount = 1
print(AKSettings.audioFormat)
print(AudioKit.format)
print(AKAudioFile().fileFormat)

print(AKAudioFile().fileFormat.settings)
print(AKAudioFile(writeIn: .temp, name: nil, settings: [
    "AVSampleRateKey": 22050,
    "AVNumberOfChannelsKey": 1,
]).fileFormat.settings)

AudioKit.format.sampleRate
AKSettings.sampleRate

let path = "foo/bar/baz.ext"
(path as NSString).deletingLastPathComponent
(path as NSString).lastPathComponent
(path as NSString).pathExtension
(path as NSString).deletingPathExtension
((path as NSString).lastPathComponent as NSString).deletingPathExtension

print(FileManager.default.temporaryDirectory)
print(FileManager.default.documentsDirectory)

let url = URL(string: "file:///private/var/mobile/Containers/Data/Application/6ECF062D-2057-42D0-9EBB-D467FAF8BF74/tmp/410D8E2B-0307-41E2-8A38-D78B103F3E97.caf")!
url.path

let path = "/foo/bar/baz.ext"
String(path.reversed())
String(path.reversed().split(separator: "/", maxSplits: 1)[0].reversed())
String(path.reversed().split(separator: "/", maxSplits: 1)[1].reversed())

print(AKAudioFile().url)

let x: Int? = 3
String(format: "%@ %@", "foo", String(describing: x))

var microphone = AKMicrophone()
AudioKit.output = AKBooster(microphone, gain: 0.0) // "Zero out the mic to prevent feedback"
try AudioKit.start()

asdf

var oscillator = AKOscillator()
var oscillator2 = AKOscillator()
AudioKit.output = AKMixer(oscillator, oscillator2)
AudioKit.start()
oscillator.amplitude = random(0.5, 1)
oscillator.frequency = random(220, 880)
oscillator.start()
// ...
oscillator.stop()

print(Thread.callStackSymbols.joined(separator: "\n"))

func f(_ g: () -> Int) -> Int { return g() * 2 }
f { 4 }
f { return 3 }

let s = "foo"
s.self
type(of: s)

//AppError("foo")
//Array<AudioQueueBufferRef>(repeating: nil, count: 3)

"x: \(Spectro.self)"

let p = UnsafeMutablePointer<String>.allocate(capacity: 1)
var v = "foo"
//p.pointee // Error
p.initialize(from: &v, count: 1)
p.pointee
p.deallocate()

import Foundation
NSError(domain: "foo", code: -5000)

let d = ["a": 1, "b": 2]
d["c"] ?? 3

import UIKit

import Bubo // Before pods imports
import AudioKit
import AudioKitUI
import Surge

Surge.sum([1,2,3,4] as [Double])

buboValue
Spectro.foo(x: "xx", y: "yy", z: 42)
AudioKit.format.sampleRate

print(Bundle.main.url(forResource: "leadloop.wav", withExtension: nil)!)
print(Bundle.main.path(forResource: "leadloop.wav", ofType: nil)!)

let url = Bundle.main.url(forResource: "leadloop.wav", withExtension: nil)!
let file = try AKAudioFile(forReading: url)
let data = try Data(contentsOf: url)

let player = AKPlayer(url: url)!
player.isLooping = true

AudioKit.output = player
try AudioKit.start()
player.play()
let fft = AKFFTTap(player)

AKPlaygroundLoop(every: 0.1) {
    if let max = fft.fftData.max() {
        let index = fft.fftData.index(of: max)
    }
}

import PlaygroundSupport
PlaygroundPage.current.needsIndefiniteExecution = true
