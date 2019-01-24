// (See Birdgram/Spectro.m)

#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(RNTagLib, RCTEventEmitter)

RCT_EXTERN_METHOD(
  readComment:(NSString *)audioPath
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

RCT_EXTERN_METHOD(
  writeComment:(NSString *)audioPath
  value:(NSString *)value
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

RCT_EXTERN_METHOD(
  audioFiletype:(NSString *)audioPath
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

@end
