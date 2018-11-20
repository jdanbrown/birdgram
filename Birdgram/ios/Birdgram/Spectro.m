#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(RNSpectro, RCTEventEmitter)

RCT_EXTERN_METHOD(
  setup:(NSDictionary *)opts
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

RCT_EXTERN_METHOD(
  start:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

RCT_EXTERN_METHOD(
  stop:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

// XXX Dev
RCT_EXTERN_METHOD(
  hello:(NSString *)x
  y:(NSString *)y
  z:(nonnull NSNumber *)z
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

@end
