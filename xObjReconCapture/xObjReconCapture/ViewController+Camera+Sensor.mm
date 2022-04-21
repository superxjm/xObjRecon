//
//  ViewController+Sensor.m
//  xObjReconCapture
//
//  Created by xujiamin on 2018/3/1.
//  Copyright © 2018年 xujiamin. All rights reserved.
//

#import "ViewController.h"
#import "ViewController+Camera+Sensor.h"

#import <Structure/Structure.h>

@implementation ViewController (Camera)

- (bool)queryCameraAuthorizationStatusAndNotifyUserIfNotGranted
{
    const NSUInteger numCameras = [[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo] count];
    
    if (0 == numCameras)
        return false; // This can happen even on devices that include a camera, when camera access is restricted globally.
    
    AVAuthorizationStatus authStatus = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    
    if (authStatus != AVAuthorizationStatusAuthorized)
    {
        NSLog(@"Not authorized to use the camera!");
        
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                                 completionHandler:^(BOOL granted)
         {
             // This block fires on a separate thread, so we need to ensure any actions here
             // are sent to the right place.
             
             // If the request is granted, let's try again to start an AVFoundation session.
             // Otherwise, alert the user that things won't go well.
             if (granted)
             {
                 dispatch_async(dispatch_get_main_queue(), ^(void) {
                     
                     [self startColorCamera];
                     
                     //_appStatus.colorCameraIsAuthorized = true;
                     //[self updateAppStatusMessage];
                     
                 });
             }
         }];
        
        return false;
    }
    return true;
    
}

- (void)selectCaptureFormat:(NSDictionary*)demandFormat
{
    AVCaptureDeviceFormat * selectedFormat = nil;
    
    for (AVCaptureDeviceFormat* format in self._videoDevice.formats)
    {
        double formatMaxFps = ((AVFrameRateRange *)[format.videoSupportedFrameRateRanges objectAtIndex:0]).maxFrameRate;
        
        CMFormatDescriptionRef formatDesc = format.formatDescription;
        FourCharCode fourCharCode = CMFormatDescriptionGetMediaSubType(formatDesc);
        
        CMVideoFormatDescriptionRef videoFormatDesc = formatDesc;
        CMVideoDimensions formatDims = CMVideoFormatDescriptionGetDimensions(videoFormatDesc);
        
        NSNumber * widthNeeded  = demandFormat[@"width"];
        NSNumber * heightNeeded = demandFormat[@"height"];
        
        if ( widthNeeded && widthNeeded .intValue!= formatDims.width )
            continue;
        
        if( heightNeeded && heightNeeded.intValue != formatDims.height )
            continue;
        
        // we only support full range YCbCr for now
        if(fourCharCode != (FourCharCode)'420f')
            continue;
        
        
        selectedFormat = format;
        break;
    }
    
    self._videoDevice.activeFormat = selectedFormat;
}

- (void)setLensPositionWithValue:(float)value lockVideoDevice:(bool)lockVideoDevice
{
    if(!self._videoDevice) return; // Abort if there's no videoDevice yet.
    
    if(lockVideoDevice && ![self._videoDevice lockForConfiguration:nil]) {
        return; // Abort early if we cannot lock and are asked to.
    }
    
    [self._videoDevice setFocusModeLockedWithLensPosition:value completionHandler:nil];
    
    if(lockVideoDevice)
        [self._videoDevice unlockForConfiguration];
}

- (bool)doesStructureSensorSupport24FPS
{
    bool ret = false;
    
    if (_sensorController)
        ret = 0 >= (long)[[_sensorController getFirmwareRevision] compare:@"2.0" options:NSNumericSearch];
    
    return ret;
}

- (bool)videoDeviceSupportsHighResColor
{
    // High Resolution Color format is width 2592, height 1936.
    // Most recent devices support this format at 30 FPS.
    // However, older devices may only support this format at a lower framerate.
    // In your Structure Sensor is on firmware 2.0+, it supports depth capture at FPS of 24.
    
    AVCaptureDevice *testVideoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if (testVideoDevice == nil)
        assert(0);
    
    const BOOL structureSensorSupports24FPS = [self doesStructureSensorSupport24FPS];
    
    for (AVCaptureDeviceFormat* format in testVideoDevice.formats)
    {
        AVFrameRateRange* firstFrameRateRange = ((AVFrameRateRange *)[format.videoSupportedFrameRateRanges objectAtIndex:0]);
        
        double formatMinFps = firstFrameRateRange.minFrameRate;
        double formatMaxFps = firstFrameRateRange.maxFrameRate;
        
        if (   (formatMaxFps < 15) // Max framerate too low.
            || (formatMinFps > 30) // Min framerate too high.
            || (formatMaxFps == 24 && !structureSensorSupports24FPS && formatMinFps > 15) // We can neither do the 24 FPS max framerate, nor fall back to 15.
            )
            continue;
        
        CMFormatDescriptionRef formatDesc = format.formatDescription;
        FourCharCode fourCharCode = CMFormatDescriptionGetMediaSubType(formatDesc);
        
        CMVideoFormatDescriptionRef videoFormatDesc = formatDesc;
        CMVideoDimensions formatDims = CMVideoFormatDescriptionGetDimensions(videoFormatDesc);
        
        if ( 2592 != formatDims.width )
            continue;
        
        if ( 1936 != formatDims.height )
            continue;
        
        // we only support full range YCbCr for now
        if(fourCharCode != (FourCharCode)'420f')
            continue;
        
        // All requirements met.
        return true;
    }
    
    // No acceptable high-res format was found.
    return false;
}

- (void)setupColorCamera
{
    // Early-return if the capture session was already setup.
    
    if (self._avCaptureSession)
        return;
    
#if 0
    // Ensure that camera access was properly granted.
    bool cameraAccessAuthorized = [self queryCameraAuthorizationStatusAndNotifyUserIfNotGranted];
    
    if (!cameraAccessAuthorized)
    {
        [self updateAppStatusMessageWithColorCameraAuthorization:false];
        
        return;
    }
#endif
    
    // Set up the capture session.
    
    self._avCaptureSession = [[AVCaptureSession alloc] init];
    
    [self._avCaptureSession beginConfiguration];
    
    // Set preset session size.
    
    // Capture color frames at VGA resolution.
    //[self._avCaptureSession setSessionPreset:AVCaptureSessionPreset640x480];
    // InputPriority allows us to select a more precise format (below)
    [self._avCaptureSession setSessionPreset:AVCaptureSessionPresetInputPriority];
    
    // Create a video device.
    self._videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    
    assert(self._videoDevice != nil);
    
    NSError *error = nil;
    
    // Use auto-exposure, and auto-white balance and set the focus to infinity.
    
    if([self._videoDevice lockForConfiguration:&error])
    {
        int imageWidth = -1;
        int imageHeight = -1;
        
        // High-resolution uses 2592x1936, which is close to a 4:3 aspect ratio.
        // Other aspect ratios such as 720p or 1080p are not yet supported.
        imageWidth = 2592;
        imageHeight = 1936;
    
        // Low resolution uses VGA.
        //imageWidth = 640;
        //imageHeight = 480;
        
        // Select capture format
        [self selectCaptureFormat:@{ @"width": @(imageWidth),
                                     @"height": @(imageHeight)}];
        
        // Allow exposure to change
        if ([self._videoDevice isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure])
        {
            //[self._videoDevice setExposureMode:AVCaptureExposureModeContinuousAutoExposure];
            [self._videoDevice setExposureMode:AVCaptureExposureModeLocked];
        }
        
        // Allow white balance to change
        if ([self._videoDevice isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance])
        {
            //[self._videoDevice setWhiteBalanceMode:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance];
            [self._videoDevice setWhiteBalanceMode:AVCaptureWhiteBalanceModeLocked];
        }
        
        // Set focus at the maximum position allowable (e.g. "near-infinity") to get the
        // best color/depth alignment.
        //[self setLensPositionWithValue:_options.lensPosition lockVideoDevice:false];
        [self._videoDevice setFocusMode:AVCaptureFocusModeContinuousAutoFocus];
        
        CGFloat currentISO = self._videoDevice.ISO;
        NSLog([NSString stringWithFormat:@"currentISO: %f", self._videoDevice.ISO]);
        CGFloat newISO = currentISO;// / 2;
        [self._videoDevice setExposureModeCustomWithDuration:AVCaptureExposureDurationCurrent ISO:newISO completionHandler:^(CMTime syncTime) {}];
        
        [self._videoDevice unlockForConfiguration];
    }
    
    // Create the video capture device input.
    
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:self._videoDevice error:&error];
    
    if (error)
    {
        NSLog(@"Cannot initialize AVCaptureDeviceInput");
        assert(0);
    }
    
    // Add the input to the capture session.
    
    [self._avCaptureSession addInput:input];
    
    //  Create the video data output.
    
    AVCaptureVideoDataOutput* dataOutput = [[AVCaptureVideoDataOutput alloc] init];
    
    // We don't want to process late frames.
    
    [dataOutput setAlwaysDiscardsLateVideoFrames:YES];
    
    // Use kCVPixelFormatType_420YpCbCr8BiPlanarFullRange format.
    
    [dataOutput setVideoSettings:@{ (NSString*)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) }];
    
    // Dispatch the capture callbacks on the main thread, where OpenGL calls can be made synchronously.
    
    [dataOutput setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
    
    // Add the output to the capture session.
    
    [self._avCaptureSession addOutput:dataOutput];
    
    // Enforce 30 FPS capture rate.
    if([self._videoDevice lockForConfiguration:&error])
    {
        [self._videoDevice setActiveVideoMaxFrameDuration:CMTimeMake(1, 30)];
        [self._videoDevice setActiveVideoMinFrameDuration:CMTimeMake(1, 30)];
        [self._videoDevice unlockForConfiguration];
    }
    
    [self._avCaptureSession commitConfiguration];
}

- (void)startColorCamera
{
    if (self._avCaptureSession && [self._avCaptureSession isRunning])
        return;
    
    // Re-setup so focus is lock even when back from background
    if (self._avCaptureSession == nil)
        [self setupColorCamera];
    
    // Start streaming color images.
    [self._avCaptureSession startRunning];
}

- (void)stopColorCamera
{
    if ([self._avCaptureSession isRunning])
    {
        // Stop the session
        [self._avCaptureSession stopRunning];
    }
    
    self._avCaptureSession = nil;
    self._videoDevice = nil;
}

- (void)setColorCameraParametersForInit
{
    NSError *error;
    
    [self._videoDevice lockForConfiguration:&error];
    
    // Auto-exposure
    if ([self._videoDevice isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure])
        [self._videoDevice setExposureMode:AVCaptureExposureModeContinuousAutoExposure];
    
    // Auto-white balance.
    if ([self._videoDevice isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance])
        [self._videoDevice setWhiteBalanceMode:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance];

    
    [self._videoDevice unlockForConfiguration];
    
}

- (void)setColorCameraParametersForScanning
{
    NSError *error;
    
    [self._videoDevice lockForConfiguration:&error];
    
    // Exposure locked to its current value.
    if ([self._videoDevice isExposureModeSupported:AVCaptureExposureModeLocked])
        [self._videoDevice setExposureMode:AVCaptureExposureModeLocked];
    
    // White balance locked to its current value.
    if ([self._videoDevice isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeLocked])
        [self._videoDevice setWhiteBalanceMode:AVCaptureWhiteBalanceModeLocked];
    
    [self._videoDevice unlockForConfiguration];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    // Pass color buffers directly to the driver, which will then produce synchronized depth/color pairs.
    [_sensorController frameSyncNewColorBuffer:sampleBuffer];
}

@end

@implementation ViewController (Sensor)

#pragma mark -  Structure Sensor delegates

- (void)setupStructureSensor
{
    [self setMsgToTextView: @"Setup The Sensor\n"];
    
    // Get the sensor controller singleton
    _sensorController = [STSensorController sharedController];
    
    // Set ourself as the delegate to receive sensor data.
    _sensorController.delegate = self;
}

- (BOOL)isStructureConnectedAndCharged
{
    return [_sensorController isConnected] && ![_sensorController isLowPower];
}

- (void)sensorDidConnect
{
    NSLog(@"[Structure] Sensor connected!");
    [self setMsgToTextView: @"Sensor Connected\n"];
    
    [self connectToStructureSensorAndStartStreaming];
}

- (void)sensorDidLeaveLowPowerMode
{
    
}

- (void)sensorBatteryNeedsCharging
{

}

- (void)sensorDidStopStreaming:(STSensorControllerDidStopStreamingReason)reason
{
    if (reason == STSensorControllerDidStopStreamingReasonAppWillResignActive)
    {
        [self stopColorCamera];
        NSLog(@"[Structure] Stopped streaming because the app will resign its active state.");
    }
    else
    {
        NSLog(@"[Structure] Stopped streaming for an unknown reason.");
    }
}

- (void)sensorDidDisconnect
{
    // If we receive the message while in background, do nothing. We'll check the status when we
    // become active again.
    if ([[UIApplication sharedApplication] applicationState] != UIApplicationStateActive)
        return;
    
    NSLog(@"[Structure] Sensor disconnected!");
    
#if 0
    // Reset the scan on disconnect, since we won't be able to recover afterwards.
    if (_slamState.scannerState == ScannerStateScanning)
    {
        [self resetButtonPressed:self];
    }
#endif
    
    [self stopColorCamera];
}


- (STSensorControllerInitStatus)connectToStructureSensorAndStartStreaming
{
    // Try connecting to a Structure Sensor.
    STSensorControllerInitStatus result = [_sensorController initializeSensorConnection];
    
    if (result == STSensorControllerInitStatusSuccess || result == STSensorControllerInitStatusAlreadyInitialized)
    {
        // Even though _useColorCamera was set in viewDidLoad by asking if an approximate calibration is guaranteed,
        // it's still possible that the Structure Sensor that has just been plugged in has a custom or approximate calibration
        // that we couldn't have known about in advance.
        
        STCalibrationType calibrationType = [_sensorController calibrationType];
        
        assert(calibrationType == STCalibrationTypeApproximate || calibrationType == STCalibrationTypeDeviceSpecific);
        
        // Start streaming depth data.
        [self setMsgToTextView: @"Start Sensor \n"];
        [self startStructureSensorStreaming];
    }
    else
    {
        switch (result)
        {
            case STSensorControllerInitStatusSensorNotFound:
                NSLog(@"[Structure] No sensor found"); break;
            case STSensorControllerInitStatusOpenFailed:
                NSLog(@"[Structure] Error: Open failed."); break;
            case STSensorControllerInitStatusSensorIsWakingUp:
                NSLog(@"[Structure] Error: Sensor still waking up."); break;
            default: {}
        }
    }
    
    return result;
}

- (void)startStructureSensorStreaming
{
    if (![self isStructureConnectedAndCharged])
        return;
    
    // Tell the driver to start streaming.
    NSError *error = nil;
    BOOL optionsAreValid = FALSE;
    
    // We can use either registered or unregistered depth.
    _structureStreamConfig = STStreamConfigDepth640x480;
    //_structureStreamConfig = STStreamConfigRegisteredDepth640x480;
    //_structureStreamConfig = STStreamConfigInfrared640x488;
    
    if (_options.useHardwareRegisteredDepth)
    {
        // We are using the color camera, so let's make sure the depth gets synchronized with it.
        // If we use registered depth, we also need to specify a fixed lens position value for the color camera.
        optionsAreValid = [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(_structureStreamConfig),
                                                                         kSTFrameSyncConfigKey : @(STFrameSyncDepthAndRgb),
                                                                         kSTColorCameraFixedLensPositionKey: @(_options.lensPosition)}
                                                                 error:&error];
    }
    else
    {
        // We are using the color camera, so let's make sure the depth gets synchronized with it.
        optionsAreValid = [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(_structureStreamConfig),
                                                                         kSTFrameSyncConfigKey : @(STFrameSyncDepthAndRgb)}
                                                                 error:&error];
    }
    
    [self startColorCamera];
    
    if (!optionsAreValid)
    {
        NSLog(@"Error during streaming start: %s", [[error localizedDescription] UTF8String]);
        return;
    }
    
    NSLog(@"[Structure] Streaming started.");
    
    // Notify and initialize streaming dependent objects.
    [self onStructureSensorStartedStreaming];
}

- (void)onStructureSensorStartedStreaming
{

}

- (void)resizeColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer
{
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(yCbCrSampleBuffer);
    
    // get image size
    size_t cols = CVPixelBufferGetWidth(pixelBuffer);
    size_t rows = CVPixelBufferGetHeight(pixelBuffer);
    //NSLog([NSString stringWithFormat:@"row: %d col: %d\n",rows, cols]);
    
    // allocate memory for RGBA image for the first time
    if(_resizedColorImageBufferYCbCr == nil)
    {
        _resizedColorImageBufferYCbCr = (uint8_t*)malloc(cols * rows * 4);
        _colorImageBufferOnlyY = (uint8_t*)malloc(cols * rows * 4);
    }
    
    // color space
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // get y plane
    uint8_t* yData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    
    // get cbCr plane
    uint8_t* cbCrData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    
    size_t yBytePerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    size_t cbcrBytePerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    //NSLog([NSString stringWithFormat:@"yBytePerRow: %d cbcrBytePerRow: %d\n",yBytePerRow, cbcrBytePerRow]);
    assert( yBytePerRow==cbcrBytePerRow );
    
    _keyFrameBytePerRow = yBytePerRow;
    _keyFrameRows = rows;
    _keyFrameCols = cols;
    for (int r = 0; r < rows / 4; ++r)
    {
        memcpy(_colorImageBufferOnlyY, yData + yBytePerRow * r, cols / 4);
    }
    
    uint8_t *pYData, *pCbCrData;
    uint8_t* pBuffer = _resizedColorImageBufferYCbCr;
    for (int r = 0; r < rows; r += 4)
    {
        pYData = yData + yBytePerRow * r;
        for (int c = 0; c < cols; c += 4)
        {
            *(pBuffer++) = pYData[c];
        }
    }
    int halfRows = rows * 0.5f, halfCols = cols * 0.5f;
    for (int r = 0; r < halfRows; r += 4)
    {
        pCbCrData = cbCrData + cbcrBytePerRow * r;
        for (int c = 0; c < halfCols; c += 4)
        {
            *(pBuffer++) = pCbCrData[2 * c];
            *(pBuffer++) = pCbCrData[2 * c + 1];
        }
    }
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)sensorDidOutputSynchronizedDepthFrame:(STDepthFrame*)depthFrame
                                   colorFrame:(STColorFrame*)colorFrame
{
    std::unique_lock<std::mutex> lk(_bufMtx);
    //_bufCon.wait(lk);
    getMeasurements(_measurements);
    if (_measurements.size() == 0)
    {
        _measurements.resize(1);
    }
    lk.unlock();
    //ImuMsg &latestImuMst = _measurements.back();
    //double acc = (latestImuMst.acc.x() * latestImuMst.acc.x() +
                  //latestImuMst.acc.y() * latestImuMst.acc.y() +
                  //latestImuMst.acc.z() * latestImuMst.acc.z());
    //[self clearAndSetMsgToTextView: [NSString stringWithFormat:@"acc: %.3f\n", acc]];
    
    //STDepthFrame* depthFrame2 = [depthFrame registeredToColorFrame: colorFrame];
    //colorFrame.registeredToColorFrame
    [self renderDepthFrame:depthFrame];
    //[self renderColorFrame:colorFrame.sampleBuffer];
    
#if 1
    [self resizeColorFrame:colorFrame.sampleBuffer];
    [self renderColorFrame648x484:_resizedColorImageBufferYCbCr];
    //UInt64 startTime = [[NSDate date] timeIntervalSince1970]*1000;
    float blurScore = [self calcBlurScore:_resizedColorImageBufferYCbCr];
    //UInt64 endTime = [[NSDate date] timeIntervalSince1970]*1000;
    //NSLog([NSString stringWithFormat:@"blurScore: %f, time: %d", blurScore, endTime - startTime]);
#endif
#if 0
    if (_measurements.size() > 2)
    {
        [self clearAndSetMsgToTextView: [NSString stringWithFormat:@"%.3f : %.3f : %.3f\n%.3f : %.3f : %.3f\n",
                                     _measurements[0].acc.x(), _measurements[0].acc.y(), _measurements[0].acc.z(),
                                     _measurements[1].acc.x(), _measurements[1].acc.y(), _measurements[1].acc.z()]];
    }
#endif
    if (_isStarted)
    {
        [self clearAndSetMsgToTextView: [NSString stringWithFormat:@"key frame num: %d %d\n", _frameIdx, _keyFrameVec.size()]];
        
        int keyFrameIdxEachFrag = 0;
        if (_skipFrameIdx < _frameNumToSkip)
        {
            ++_skipFrameIdx;
            keyFrameIdxEachFrag = -1;
        }
        else
        {
            _frameNumToSkip = 0;
            _skipFrameIdx = 0;
        }
        
        if (_frameIdx % FRAG_SIZE == (FRAG_SIZE - 1))
        {
            _frameNumToSkip = 2;
        }
        if (_hasBeenPaused)
        {
            _frameNumToSkip = 30;
            _measurements.clear(); // clear the imu measure, let it be a signal of pause
            _hasBeenPaused = false;
        }
        if (keyFrameIdxEachFrag != -1)
        {
             keyFrameIdxEachFrag = [self saveKeyFrame: colorFrame.sampleBuffer AccordingTo: blurScore];
        }
        
        //[self compressAndSendDepthFrame:depthFrame ColorFrame:colorFrame.sampleBuffer];
        //[self compressAndSendDepthFrame:depthFrame ColorFrame:colorFrame.sampleBuffer CurGravity: _curGravity Measurements: _measurements KeyFrameIdxEachFrag:0];
        [self compressAndSendDepthFrame:depthFrame ColorFrame648x484:_resizedColorImageBufferYCbCr CurGravity: _curGravity Measurements: _measurements KeyFrameIdxEachFrag:keyFrameIdxEachFrag];
        //[self compressAndSendColorFrame:colorFrame.sampleBuffer];
        //[self compressAndSendDepthFrame:depthFrame];
        if (keyFrameIdxEachFrag != -1)
            ++_frameIdx;
    }
    
    //[self setMsgToTextView: [NSString stringWithFormat:@"%d\n",_frameIdx]];
}

- (void)sensorDidOutputSynchronizedInfraredFrame:(STInfraredFrame *)irFrame
                                      colorFrame:(STColorFrame *)colorFrame
{
    [self renderIrFrame:irFrame];
    [self renderColorFrame:colorFrame.sampleBuffer];

    if (_isStarted)
    {
        [self compressAndSendIrFrame:irFrame ColorFrame:colorFrame.sampleBuffer];
    }
    
    //[self setMsgToTextView: [NSString stringWithFormat:@"%d\n",_frameIdx]];
    ++_frameIdx;
}


- (void)sensorDidOutputDepthFrame:(STDepthFrame *)depthFrame
{
   
}

@end
