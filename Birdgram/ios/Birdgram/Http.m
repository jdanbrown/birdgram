// (See Birdgram/Spectro.m)

#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(RNHttp, RCTEventEmitter)

RCT_EXTERN_METHOD(
  httpFetch:(NSString *)url
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

@end
