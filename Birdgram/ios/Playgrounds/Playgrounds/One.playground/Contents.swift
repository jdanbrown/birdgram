// Run using scheme: Birdgram -> <any>
//  - Key: shift-enter to run current line
//  - Key: shift-cmd-enter to run all lines
//  - Key: ctrl-\ to stop running playground
//  - Key: ctrl-enter to show/hide current result inline [doesn't always work...]

import Bubo // Before pods imports
import AudioKit
import AudioKitUI
import Surge

let d = ["a": 1, "b": 2]
d["a"] as? Int

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
