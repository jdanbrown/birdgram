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

CFURL(

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
