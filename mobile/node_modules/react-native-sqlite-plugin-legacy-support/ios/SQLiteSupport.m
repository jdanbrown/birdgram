#import "SQLiteSupport.h"

#import "SQLitePlugin.h"

static SQLitePlugin * plugin = nil;

@implementation SQLiteSupport

+ (void) initialize
{
    plugin = [[SQLitePlugin alloc] init];
    [plugin pluginInitialize];
}

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(sampleMethod:(NSString *)stringArgument numberParameter:(nonnull NSNumber *)numberArgument callback:(RCTResponseSenderBlock)callback)
{
    // TODO: Implement some real useful functionality
    callback(@[[NSString stringWithFormat: @"numberArgument: %@ stringArgument: %@", numberArgument, stringArgument]]);
}

RCT_EXPORT_METHOD(echoStringValue:(NSArray *)arguments callback:(RCTResponseSenderBlock)success error:(RCTResponseSenderBlock)error)
{
    // echo error callback not expected, ignored here.
    [plugin echoIt: arguments error: nil success: ^(NSObject * aa) {
        success(@[aa]);
    }];
}

RCT_EXPORT_METHOD(open:(NSArray *)arguments callback:(RCTResponseSenderBlock)success error:(RCTResponseSenderBlock)error)
{
    NSLog(@"open method");
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [plugin openNow: arguments
                  error: ^(NSObject * e) {
                    error(@[e]);
		  }
                success: ^() {
                    success(@[]);
                }];
    });
}

RCT_EXPORT_METHOD(backgroundExecuteSqlBatch:(NSArray *)arguments callback:(RCTResponseSenderBlock)success error:(RCTResponseSenderBlock)error)
{
    NSLog(@"batch method");
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [plugin executeSqlBatchNow: arguments error: nil success: ^(NSObject * aa) {
    NSLog(@"bg success cb with object");
    //NSLog(aa);
                    success(@[aa]);
                }];
    });
}

RCT_EXPORT_METHOD(close:(NSArray *)arguments callback:(RCTResponseSenderBlock)success error:(RCTResponseSenderBlock)error)
{
    NSLog(@"close method");
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [plugin closeNow: arguments
                  error: ^(NSObject * e) {
                    error(e);
		  }
                success: ^() {
                    success(@[]);
                }];
    });
}

RCT_EXPORT_METHOD(delete:(NSArray *)arguments callback:(RCTResponseSenderBlock)success error:(RCTResponseSenderBlock)error)
{
    NSLog(@"delete method");
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [plugin deleteNow: arguments
                  error: ^(NSObject * e) {
                    error(e);
		  }
                success: ^() {
                    success(@[]);
                }];
    });
}

@end
