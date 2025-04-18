// From http://paulsolt.com/2010/09/ios-converting-uiimage-to-rgba8-bitmaps-and-back/

/*
 * The MIT License
 *
 * Copyright (c) 2011 Paul Solt, PaulSolt@gmail.com
 *
 * https://github.com/PaulSolt/UIImage-Conversion/blob/master/MITLicense.txt
 *
 */

#import "ImageHelper.h"


@implementation ImageHelper


+ (unsigned char *) convertUIImageToBitmapRGBA8:(UIImage *) image {

  CGImageRef imageRef = image.CGImage;

  // Create a bitmap context to draw the uiimage into
  CGContextRef context = [self newBitmapRGBA8ContextFromImage:imageRef];

  if(!context) {
    return NULL;
  }

  size_t width = CGImageGetWidth(imageRef);
  size_t height = CGImageGetHeight(imageRef);

  CGRect rect = CGRectMake(0, 0, width, height);

  // Draw image into the context to get the raw image data
  CGContextDrawImage(context, rect, imageRef);

  // Get a pointer to the data
  unsigned char *bitmapData = (unsigned char *)CGBitmapContextGetData(context);

  // Copy the data and release the memory (return memory allocated with new)
  size_t bytesPerRow = CGBitmapContextGetBytesPerRow(context);
  size_t bufferLength = bytesPerRow * height;

  unsigned char *newBitmap = NULL;

  if(bitmapData) {
    newBitmap = (unsigned char *)malloc(sizeof(unsigned char) * bytesPerRow * height);

    if(newBitmap) {  // Copy the data
      for(int i = 0; i < bufferLength; ++i) {
        newBitmap[i] = bitmapData[i];
      }
    }

    free(bitmapData);

  } else {
    NSLog(@"Error getting bitmap pixel data\n");
  }

  CGContextRelease(context);

  return newBitmap;
}

+ (CGContextRef) newBitmapRGBA8ContextFromImage:(CGImageRef) image {
  CGContextRef context = NULL;
  CGColorSpaceRef colorSpace;
  uint32_t *bitmapData;

  size_t bitsPerPixel = 32;
  size_t bitsPerComponent = 8;
  size_t bytesPerPixel = bitsPerPixel / bitsPerComponent;

  size_t width = CGImageGetWidth(image);
  size_t height = CGImageGetHeight(image);

  size_t bytesPerRow = width * bytesPerPixel;
  size_t bufferLength = bytesPerRow * height;

  colorSpace = CGColorSpaceCreateDeviceRGB();

  if(!colorSpace) {
    NSLog(@"Error allocating color space RGB\n");
    return NULL;
  }

  // Allocate memory for image data
  bitmapData = (uint32_t *)malloc(bufferLength);

  if(!bitmapData) {
    NSLog(@"Error allocating memory for bitmap\n");
    CGColorSpaceRelease(colorSpace);
    return NULL;
  }

  //Create bitmap context

  context = CGBitmapContextCreate(bitmapData,
                                  width,
                                  height,
                                  bitsPerComponent,
                                  bytesPerRow,
                                  colorSpace,
                                  kCGImageAlphaPremultipliedLast);  // RGBA
  if(!context) {
    free(bitmapData);
    NSLog(@"Bitmap context not created");
  }

  CGColorSpaceRelease(colorSpace);

  return context;
}

+ (UIImage *) convertBitmapRGBA8ToUIImage:(unsigned char *) buffer
                                withWidth:(int) width
                               withHeight:(int) height
                                grayscale:(bool) grayscale {

  // Grayscale vs. RGBA
  size_t bitsPerComponent       = grayscale ? 8 : 8;
  size_t componentsPerPixel     = grayscale ? 1 : 4;
  CGBitmapInfo bitmapInfo       = (grayscale ? kCGImageAlphaNone : kCGImageAlphaPremultipliedLast) | kCGBitmapByteOrderDefault;
  CGColorSpaceRef colorSpaceRef = grayscale ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB();

  size_t bitsPerPixel = bitsPerComponent * componentsPerPixel;
  size_t bytesPerRow = componentsPerPixel * width;
  size_t bufferLength = componentsPerPixel * width * height;
  CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, buffer, bufferLength, NULL);
  CGColorRenderingIntent renderingIntent = kCGRenderingIntentDefault;

  if(colorSpaceRef == NULL) {
    NSLog(@"Error allocating color space");
    CGDataProviderRelease(provider);
    return nil;
  }

  CGImageRef iref = CGImageCreate(width,
                                  height,
                                  bitsPerComponent,
                                  bitsPerPixel,
                                  bytesPerRow,
                                  colorSpaceRef,
                                  bitmapInfo,
                                  provider,  // data provider
                                  NULL,    // decode
                                  YES,      // should interpolate
                                  renderingIntent);

  uint32_t* pixels = (uint32_t*)malloc(bufferLength);

  if(pixels == NULL) {
    NSLog(@"Error: Memory not allocated for bitmap");
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpaceRef);
    CGImageRelease(iref);
    return nil;
  }

  CGContextRef context = CGBitmapContextCreate(pixels,
                                               width,
                                               height,
                                               bitsPerComponent,
                                               bytesPerRow,
                                               colorSpaceRef,
                                               bitmapInfo);

  if(context == NULL) {
    NSLog(@"Error context not created");
    free(pixels);
  }

  UIImage *image = nil;
  if(context) {

    CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, width, height), iref);

    CGImageRef imageRef = CGBitmapContextCreateImage(context);

    // Support both iPad 3.2 and iPhone 4 Retina displays with the correct scale
    if([UIImage respondsToSelector:@selector(imageWithCGImage:scale:orientation:)]) {
      float scale = [[UIScreen mainScreen] scale];
      image = [UIImage imageWithCGImage:imageRef scale:scale orientation:UIImageOrientationUp];
    } else {
      image = [UIImage imageWithCGImage:imageRef];
    }

    CGImageRelease(imageRef);
    CGContextRelease(context);
  }

  CGColorSpaceRelease(colorSpaceRef);
  CGImageRelease(iref);
  CGDataProviderRelease(provider);

  if(pixels) {
    free(pixels);
  }
  return image;
}

@end
