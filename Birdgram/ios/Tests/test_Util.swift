import Bubo

//
// Collection.slice
//

let xs = [5,6,7]
testEqual("slice", xs.slice(), xs)
testEqual("slice", xs.slice(from: 0), [5,6,7])
testEqual("slice", xs.slice(from: 1), [6,7])
testEqual("slice", xs.slice(from: -1), [7])
testEqual("slice", xs.slice(from: xs.count), [])
testEqual("slice", xs.slice(to: 0), [])
testEqual("slice", xs.slice(to: xs.count), xs)
testEqual("slice", xs.slice(to: -1), [5,6])
testEqual("slice", xs.slice(from: 1, to: -1), [6])
testEqual("slice", xs.slice(from: -2, to: -1), [6])
testEqual("slice", xs.slice(from: -8, to: 20), [5,6,7])
testEqual("slice", xs.slice(from: -8), [5,6,7])
testEqual("slice", xs.slice(to: 20), [5,6,7])
