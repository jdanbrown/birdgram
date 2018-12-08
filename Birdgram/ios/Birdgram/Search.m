// (See Birdgram/Search.m)

#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(RNSearch, RCTEventEmitter)

RCT_EXTERN_METHOD(
  create:(NSDictionary *)modelsSearch
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

RCT_EXTERN_METHOD(
  preds:(NSString *)audioPath
  resolve:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
);

@end
