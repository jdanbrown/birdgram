//
//  Bubo.h
//  Bubo
//
//  Created by Dan Brown on 2018-11-18.
//  Copyright © 2018 Dan Brown. All rights reserved.
//

// #import <UIKit/UIKit.h> // XXX Replaced with Foundation.h for macos + ios build
#import <Foundation/Foundation.h>

//! Project version number for Bubo.
FOUNDATION_EXPORT double BuboVersionNumber;

//! Project version string for Bubo.
FOUNDATION_EXPORT const unsigned char BuboVersionString[];

// Import all "public" headers for the framework (not "project" or "private")
//  - i.e. must be 1-1 with Bubo -> Build Phases -> Headers -> Public
//  - If you encounter "include of non-modular header" errors, make sure the headers in xcode are Public i/o Private or Project
#import "tag_c.h"
#import "NSURL+TagLib.h"    // Unused
#import "TagLib+CoverArt.h" // Unused
