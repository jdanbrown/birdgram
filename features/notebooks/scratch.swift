// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
import Surge; import SigmaSwiftStatistics; import SwiftyJSON; import Yams; import Bubo

%%
// XXX PULL HAIR OUT LIGHT EVERYTHING ON FIRE RUN AWAY

%%
print("foo")

%%
import Accelerate
import Foundation
print(ProcessInfo.processInfo.environment["REPL_SWIFT_PATH"])
print(ProcessInfo.processInfo.environment["foo"])
print(ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"])
print(ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"])
print(ProcessInfo.processInfo.environment["SWIFT_INCLUDE_PATHS"])
print(ProcessInfo.processInfo.environment["SWIFT_OTHER_FLAGS"])
print(CommandLine.arguments)

%%
import Surge

%%
[1,2] .* [3,4]

%%
fft([1,2,3,4])

%%
// NOTE Must first import every module from Bubo's "Linked Frameworks and Libraries", else "error: Couldn't lookup symbols"
// import Surge
// import SigmaSwiftStatistics
// import SwiftyJSON
// import Yams
// import Bubo

%%
print([1,3])
debugPrint([1,3])
dump([1,3])
3

%%
import Foundation
ProcessInfo.processInfo.environment["foo"]

%%
3

%% {once: true}
foo_bubo()
nowSeconds()

////
