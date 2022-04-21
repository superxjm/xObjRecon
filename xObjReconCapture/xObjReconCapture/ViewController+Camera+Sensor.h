//
//  ViewController+Sensor.h
//  xObjReconCapture
//
//  Created by xujiamin on 2018/3/1.
//  Copyright © 2018年 xujiamin. All rights reserved.
//

#ifndef ViewController_Sensor_h
#define ViewController_Sensor_h

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>
#define HAS_LIBCXX

#import "ViewController.h"
#import <Structure/Structure.h>

@interface ViewController (Camera) <AVCaptureVideoDataOutputSampleBufferDelegate>

- (void) startColorCamera;
- (void) stopColorCamera;
- (void) setColorCameraParametersForInit;
- (void) setColorCameraParametersForScanning;
- (void) setLensPositionWithValue:(float)value lockVideoDevice:(bool)lockVideoDevice;

- (bool) videoDeviceSupportsHighResColor;

@end

@interface ViewController (Sensor) <STSensorControllerDelegate>
    
- (STSensorControllerInitStatus)connectToStructureSensorAndStartStreaming;
- (void)setupStructureSensor;
- (BOOL)isStructureConnectedAndCharged;

@end

#endif /* ViewController_Sensor_h */
