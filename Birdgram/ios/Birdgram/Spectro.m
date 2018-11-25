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

RCT_EXTERN_METHOD(
  stats:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

@end
