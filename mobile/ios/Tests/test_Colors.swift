import Foundation
import Bubo

testEqual("Colors.magma.count", Colors.magma.count, 256)
testEqual("Colors.magma[0]",    Colors.magma[0],   Color(0x00, 0x00, 0x04, 0xff))
testEqual("Colors.magma[255]",  Colors.magma[255], Color(0xfc, 0xfd, 0xbf, 0xff))
