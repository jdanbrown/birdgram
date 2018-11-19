#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(RNSpectro, NSObject)

RCT_EXTERN_METHOD(
  foo:(NSString *)x
  y:(NSString *)y
  z:(nonnull NSNumber *)z
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

@end
