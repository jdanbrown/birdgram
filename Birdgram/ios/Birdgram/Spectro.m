// To expose a method from swift to js:
//  - app/native/Spectro.ts      - add js Spectro.f() calling objc NativeModules.RNSpectro.f()
//  - ios/Birdgram/Spectro.m     - add objc extern for swift RNSpectro.f()
//  - ios/Birdgram/Spectro.swift - add swift RNSpectro.f() calling Spectro.f()
//  - ios/Birdgram/Spectro.swift - add swift Spectro.f()

#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(RNSpectro, RCTEventEmitter)

// XXX Debug
RCT_EXTERN_METHOD(
  debugPrintNative:(NSString *)msg
);

RCT_EXTERN_METHOD(
  create:(NSDictionary *)opts
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

RCT_EXTERN_METHOD(
  renderAudioPathToSpectroPath:(NSString *)audioPath
  spectroPath:(NSString *)spectroPath
  opts:(NSDictionary *)opts
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

RCT_EXTERN_METHOD(
  chunkImageFile:(NSString *)path
  chunkWidth:(NSInteger *)chunkWidth
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

RCT_EXTERN_METHOD(
  preds:(NSDictionary *)modelsSearch
  audioPath:(NSString *)audioPath
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

@end
