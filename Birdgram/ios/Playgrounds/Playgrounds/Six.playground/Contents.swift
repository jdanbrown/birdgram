// Run using scheme: Birdgram -> <any>
//  - Key: shift-enter to run current line
//  - Key: shift-cmd-enter to run all lines
//  - Key: ctrl-\ to stop running playground
//  - Key: ctrl-enter to show/hide current result inline [doesn't always work...]

import UIKit

import Bubo
import AudioKit // After Bubo
import AudioKitUI // After Bubo

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
