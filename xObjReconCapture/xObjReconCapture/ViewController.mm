//
//  ViewController.m
//  xObjReconCapture
//
//  Created by xujiamin on 2018/3/1.
//  Copyright © 2018年 xujiamin. All rights reserved.
//

#import "ViewController.h"
#import "ViewController+Camera+Sensor.h"

#import <Accelerate/Accelerate.h>
#import <algorithm>
#import <stdio.h>

#import "ImageTrans.h"

namespace {
    bool convertYCbCrToBGRA (size_t width,
                             size_t height,
                             const uint8_t* yData,
                             const uint8_t* cbcrData,
                             uint8_t* rgbaData,
                             uint8_t alpha,
                             size_t yBytesPerRow,
                             size_t cbCrBytesPerRow,
                             size_t rgbaBytesPerRow) {
        assert(width <= rgbaBytesPerRow);
        
        // Input RGBA buffer:
        vImage_Buffer rgbaBuffer
        {
            .data = (void*)rgbaData,
            .width = (size_t)width,
            .height = (size_t)height,
            .rowBytes = rgbaBytesPerRow
        };
        
        // Destination Y, CbCr buffers:
        vImage_Buffer cbCrBuffer
        {
            .data = (void*)cbcrData,
            .width = (size_t)width/2,
            .height = (size_t)height/2,
            .rowBytes = (size_t)cbCrBytesPerRow // 2 bytes per pixel (Cb+Cr)
        };
        
        vImage_Buffer yBuffer
        {
            .data = (void*)yData,
            .width = (size_t)width,
            .height = (size_t)height,
            .rowBytes = (size_t)yBytesPerRow
        };
        
        vImage_Error error = kvImageNoError;
        
        // Conversion information:
        static vImage_YpCbCrToARGB info;
        {
            static bool infoGenerated = false;
            if(!infoGenerated)
            {
                vImage_Flags flags = kvImageNoFlags;
                
                vImage_YpCbCrPixelRange pixelRange
                {
                    .Yp_bias =      0,
                    .CbCr_bias =    128,
                    .YpRangeMax =   255,
                    .CbCrRangeMax = 255,
                    .YpMax =        255,
                    .YpMin =        0,
                    .CbCrMax=       255,
                    .CbCrMin =      1
                };
                
                error = vImageConvert_YpCbCrToARGB_GenerateConversion(kvImage_YpCbCrToARGBMatrix_ITU_R_601_4,
                                                                      &pixelRange,
                                                                      &info,
                                                                      kvImage420Yp8_CbCr8, kvImageARGB8888,
                                                                      flags);
                
                if (kvImageNoError != error)
                    return false;
                
                infoGenerated = true;
            }
        }
        
        static const uint8_t permuteMapBGRA [4] { 3, 2, 1, 0 };
        error = vImageConvert_420Yp8_CbCr8ToARGB8888(&yBuffer,
                                                     &cbCrBuffer,
                                                     &rgbaBuffer,
                                                     &info,
                                                     permuteMapBGRA,
                                                     255,
                                                     kvImageNoFlags | kvImageDoNotTile); // Disable multithreading.
        
        return kvImageNoError == error;
    }
    
} // namespace

@interface ViewController () {
    
}
@end

@implementation ViewController

- (void)dealloc
{
    if (_linearizeBuffer != nil)
        free(_linearizeBuffer);
    if (_coloredDepthBuffer != nil)
        free(_coloredDepthBuffer);
    if (_colorImageBuffer != nil)
        free(_colorImageBuffer);
}

- (void) setMsgToTextView:(NSString *)info {
     self._infoTextView.text = [self._infoTextView.text stringByAppendingString:info];
}

- (void) clearAndSetMsgToTextView:(NSString *)info {
    self._infoTextView.text = info;
}

// 控件事件处理
- (IBAction)connectToServerSwitchValueChangedAction:(id)sender {
    UISwitch *switchButton = (UISwitch*)sender;
    BOOL isButtonOn = [switchButton isOn];
    if (isButtonOn) {
        [_socketClient setupConnection];
        [self setMsgToTextView: @"Try to connect to server\n"];
        [NSThread sleepForTimeInterval:1.0f];
        if (![_socketClient isConnected])
        {
            [switchButton setOn:false];
        }
    }
#if 0
    isButtonOn = [switchButton isOn];
    if (isButtonOn) {
        [self setMsgToTextView: @"Connect to server\n"];
    }
    else {
        [self setMsgToTextView: @"Disconnect to server\n"];
    }
#endif
}

- (IBAction)triggerButtonPressedAction:(id)sender {
    UIButton *button = (UIButton*)sender;
    if (_isStarted == true)
    {
        _hasBeenPaused = true;
    }
    _isStarted = !_isStarted;
#if 0
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",_isStarted]];
#endif
#if 0
    _isStarted = false;
    [_socketClient sendHello];
    [self setMsgToTextView: @"Send Hello\n"];
#endif
    
    if (_isStarted == true)
    {
        [button setAttributedTitle:[[NSAttributedString alloc]initWithString:@"PAUSE"] forState:UIControlStateNormal];
    }
    else
    {
        [button setAttributedTitle:[[NSAttributedString alloc]initWithString:@"START"] forState:UIControlStateNormal];
    }
}

- (IBAction)finishScanButtonPressedAction:(id)sender {
    UIButton *button = (UIButton*)sender;

    [self sendAllKeyFramesFrame];
    
    [button setAttributedTitle:[[NSAttributedString alloc]initWithString:@"finish..."] forState:UIControlStateNormal];
}

- (void)convertIrToRGBA:(const uint16_t*)values irValuesCount:(size_t)irValuesCount
{
    for (size_t i = 0; i < irValuesCount; i++)
    {
#if 1
        uint16_t value = values[i];
        
        // Use the upper byte of the linearized shift value to choose a base color
        // Base colors range from: (closest) White, Red, Orange, Yellow, Green, Cyan, Blue, Black (farthest)
        int lowerByte = (value & 0xff);
        
        // Use the lower byte to scale between the base colors
        int upperByte = (value >> 8);
        
        switch (upperByte)
        {
            case 0:
                _coloredDepthBuffer[4 * i + 0] = 255;
                _coloredDepthBuffer[4 * i + 1] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 2] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 3] = 255;
                break;
            case 1:
                _coloredDepthBuffer[4 * i + 0] = 255;
                _coloredDepthBuffer[4 * i + 1] = lowerByte;
                _coloredDepthBuffer[4 * i + 2] = 0;
                break;
            case 2:
                _coloredDepthBuffer[4 * i + 0] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 1] = 255;
                _coloredDepthBuffer[4 * i + 2] = 0;
                break;
            case 3:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 255;
                _coloredDepthBuffer[4 * i + 2] = lowerByte;
                break;
            case 4:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 2] = 255;
                break;
            case 5:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 0;
                _coloredDepthBuffer[4 * i + 2] = 255 - lowerByte;
                break;
            default:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 0;
                _coloredDepthBuffer[4 * i + 2] = 0;
                break;
        }
#endif
    }
}

- (void)renderIrFrame:(STInfraredFrame *)irFrame
{
    size_t cols = irFrame.width;
    size_t rows = irFrame.height;
    
    if (_coloredDepthBuffer == nil)
    {
        _coloredDepthBuffer = (uint8_t*)malloc(cols * rows * 4);
    }
    
    [self convertIrToRGBA:irFrame.data irValuesCount:cols * rows];
    
    [self clearAndSetMsgToTextView: @"Ir Image\n"];
    
#if 1
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
    bitmapInfo |= kCGBitmapByteOrder32Big;
    
    NSData *data = [NSData dataWithBytes:_coloredDepthBuffer length:cols * rows * 4];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data); //toll-free ARC bridging
    
    CGImageRef imageRef = CGImageCreate(cols,                       //width
                                        rows,                        //height
                                        8,                           //bits per component
                                        8 * 4,                       //bits per pixel
                                        cols * 4,                    //bytes per row
                                        colorSpace,                  //Quartz color space
                                        bitmapInfo,                  //Bitmap info (alpha channel?, order, etc)
                                        provider,                    //Source of data for bitmap
                                        NULL,                        //decode
                                        false,                       //pixel interpolation
                                        kCGRenderingIntentDefault);  //rendering intent
    
    // Assign CGImage to UIImage
    _depthImageView.image = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
#endif
}

// Conversion of 16-bit non-linear shift depth values to 32-bit RGBA
// Adapted from: https://github.com/OpenKinect/libfreenect/blob/master/examples/glview.c
// This function is equivalent to calling [STDepthAsRgba convertDepthFrameToRgba] with the STDepthToRgbaStrategyRedToBlueGradient strategy.
// Not using the SDK here for didactic purposes.
- (void)convertShiftToRGBA:(const uint16_t*)shiftValues depthValuesCount:(size_t)depthValuesCount
{
    for (size_t i = 0; i < depthValuesCount; i++)
    {
        // We should not get higher values than maxShiftValue, but let's stay on the safe side.
        uint16_t boundedShift = std::min (shiftValues[i], maxShiftValue);
        
        // Use a lookup table to make the non-linear input values vary more linearly with metric depth
        int linearizedDepth = _linearizeBuffer[boundedShift];
        
        // Use the upper byte of the linearized shift value to choose a base color
        // Base colors range from: (closest) White, Red, Orange, Yellow, Green, Cyan, Blue, Black (farthest)
        int lowerByte = (linearizedDepth & 0xff);
        
        // Use the lower byte to scale between the base colors
        int upperByte = (linearizedDepth >> 8);
        
        switch (upperByte)
        {
            case 0:
                _coloredDepthBuffer[4 * i + 0] = 255;
                _coloredDepthBuffer[4 * i + 1] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 2] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 3] = 255;
                break;
            case 1:
                _coloredDepthBuffer[4 * i + 0] = 255;
                _coloredDepthBuffer[4 * i + 1] = lowerByte;
                _coloredDepthBuffer[4 * i + 2] = 0;
                break;
            case 2:
                _coloredDepthBuffer[4 * i + 0] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 1] = 255;
                _coloredDepthBuffer[4 * i + 2] = 0;
                break;
            case 3:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 255;
                _coloredDepthBuffer[4 * i + 2] = lowerByte;
                break;
            case 4:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 255 - lowerByte;
                _coloredDepthBuffer[4 * i + 2] = 255;
                break;
            case 5:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 0;
                _coloredDepthBuffer[4 * i + 2] = 255 - lowerByte;
                break;
            default:
                _coloredDepthBuffer[4 * i + 0] = 0;
                _coloredDepthBuffer[4 * i + 1] = 0;
                _coloredDepthBuffer[4 * i + 2] = 0;
                break;
        }
    }
}

const uint16_t maxShiftValue = 2048;
- (void)renderDepthFrame:(STDepthFrame *)depthFrame
{
    size_t cols = depthFrame.width;
    size_t rows = depthFrame.height;
    
    if (_linearizeBuffer == nil)
    {
        _linearizeBuffer = (uint16_t*)malloc((maxShiftValue + 1) * sizeof(uint16_t));
        for (int i=0; i <= maxShiftValue; i++)
        {
            float v = i / (float)maxShiftValue;
            v = powf(v, 3)* 6;
            _linearizeBuffer[i] = v*6*256;
        }
        _coloredDepthBuffer = (uint8_t*)malloc(cols * rows * 4);
    }
    
    [self convertShiftToRGBA:depthFrame.shiftData depthValuesCount:cols * rows];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
    bitmapInfo |= kCGBitmapByteOrder32Big;
    
    NSData *data = [NSData dataWithBytes:_coloredDepthBuffer length:cols * rows * 4];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data); //toll-free ARC bridging
    
    CGImageRef imageRef = CGImageCreate(cols,                       //width
                                        rows,                        //height
                                        8,                           //bits per component
                                        8 * 4,                       //bits per pixel
                                        cols * 4,                    //bytes per row
                                        colorSpace,                  //Quartz color space
                                        bitmapInfo,                  //Bitmap info (alpha channel?, order, etc)
                                        provider,                    //Source of data for bitmap
                                        NULL,                        //decode
                                        false,                       //pixel interpolation
                                        kCGRenderingIntentDefault);  //rendering intent
    
    // Assign CGImage to UIImage
    _depthImageView.image = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
}

- (void)compressAndSendColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer
{
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(yCbCrSampleBuffer);
    
    // get image size
    int colorCols = CVPixelBufferGetWidth(pixelBuffer);
    int colorRows = CVPixelBufferGetHeight(pixelBuffer);
    int colorPixelNum = colorRows * colorCols;
    
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // get y plane
    const uint8_t* yData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    
    // get cbCr plane
    const uint8_t* cbCrData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    
    int yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    int cbcrBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    assert( yBytesPerRow==cbcrBytesPerRow );
    
    int cbCrLength = (int)(colorRows * cbcrBytesPerRow / 2);
    if (_yCbCrCompressedBuffer == nil)
    {
        _yCbCrCompressedBuffer = (uint8_t*)malloc(colorPixelNum + cbCrLength);
    }
    if (_cbCrSplittedBuffer == nil)
    {
        _cbCrSplittedBuffer = (uint8_t*)malloc(cbCrLength);
    }
    
    memcpy(_yCbCrCompressedBuffer, yData, colorPixelNum);
    int cbCompressedLength, crCompressedLength;
    ImageTrans::CompressCbCr((char*)_yCbCrCompressedBuffer + colorPixelNum, cbCompressedLength, crCompressedLength,
                             (const char*)cbCrData, (char*)_cbCrSplittedBuffer, cbCrLength);
#if 1
    [_socketClient sendCompressedColor:(const uint8_t*)_yCbCrCompressedBuffer withRows:colorRows andCols:colorCols andCbCompressedLength:cbCompressedLength andCrCompressedLength:crCompressedLength andFrameIdx: _frameIdx];
#endif
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame
{
    size_t depthCols = depthFrame.width;
    size_t depthRows = depthFrame.height;
    int depthPixelNum = depthRows * depthCols;
    uint8_t* pDepthData = (uint8_t*)depthFrame.shiftData;
    //uint16_t* depthData = depthFrame.shiftData;
    
    if (_depthMsbLsbCompressedBuffer == nil)
    {
        _depthMsbLsbCompressedBuffer = (uint8_t*)malloc(depthPixelNum * 2);
    }
    if (_depthMsbLsbSplittedBuffer == nil)
    {
        _depthMsbLsbSplittedBuffer = (uint8_t*)malloc(depthPixelNum * 2);
    }
    
    uint8_t *msbPtr = _depthMsbLsbSplittedBuffer;
    uint8_t *lsbPtr = _depthMsbLsbSplittedBuffer + depthPixelNum;
    for (int i = 0; i < depthPixelNum; ++i)
    {
        *(lsbPtr++) = *(pDepthData++);
        *(msbPtr++) = *(pDepthData++);
    }
    int msbCompressedLength, lsbCompressedLength;
#if 1
    ImageTrans::CompressDepth((char *)_depthMsbLsbCompressedBuffer, msbCompressedLength, lsbCompressedLength,
                              (char *)_depthMsbLsbSplittedBuffer, depthPixelNum * 2);
#endif
#if 0
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthRows]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthCols]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",msbCompressedLength]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",lsbCompressedLength]];
#endif
#if 0
    memset(_depthMsbLsbSplittedBuffer, 0, depthPixelNum * 2);
    char *tmp = (char *)malloc(depthPixelNum * 2);
    ImageTrans::DecompressDepth((char *)_depthMsbLsbSplittedBuffer,  depthPixelNum * 2, (char *)tmp,
                                (char *)_depthMsbLsbCompressedBuffer, msbCompressedLength, lsbCompressedLength);
    free(tmp);
#endif
#if 1
    [_socketClient sendCompressedDepth:(const uint8_t*)_depthMsbLsbCompressedBuffer withRows:depthRows andCols:depthCols andMsbCompressedLength:msbCompressedLength andLsbCompressedLength:lsbCompressedLength andFrameIdx: _frameIdx];
#endif
#if 0
    [_socketClient sendCompressedDepth:(const uint8_t*)_depthMsbLsbSplittedBuffer withRows:depthRows andCols:depthCols andMsbCompressedLength:depthRows*depthCols andLsbCompressedLength:depthRows*depthCols andFrameIdx: _frameIdx];
#endif
}

- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame ColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer
{
#if 1
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(yCbCrSampleBuffer);
    
    // get image size
    int colorCols = CVPixelBufferGetWidth(pixelBuffer);
    int colorRows = CVPixelBufferGetHeight(pixelBuffer);
    int colorPixelNum = colorRows * colorCols;
    
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // get y plane
    const uint8_t* yData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    
    // get cbCr plane
    const uint8_t* cbCrData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    
    int yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    int cbcrBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    assert( yBytesPerRow==cbcrBytesPerRow );
    
    int cbCrLength = (int)(colorRows * cbcrBytesPerRow / 2);
    if (_colorDepthCompressedBuffer == nil)
    {
        _colorDepthCompressedBuffer = (uint8_t*)malloc(colorPixelNum * 5);
    }
    if (_cbCrSplittedBuffer == nil)
    {
        _cbCrSplittedBuffer = (uint8_t*)malloc(cbCrLength);
    }
    
    memcpy(_colorDepthCompressedBuffer, yData, colorPixelNum);
    int cbCompressedLength, crCompressedLength;
    ImageTrans::CompressCbCr((char*)_colorDepthCompressedBuffer + colorPixelNum, cbCompressedLength, crCompressedLength,
                             (const char*)cbCrData, (char*)_cbCrSplittedBuffer, cbCrLength);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
#endif
    
#if 1
    size_t depthCols = depthFrame.width;
    size_t depthRows = depthFrame.height;
    int depthPixelNum = depthRows * depthCols;
    uint8_t* pDepthData = (uint8_t*)depthFrame.shiftData;
    //uint16_t* depthData = depthFrame.shiftData;
    
    if (_depthMsbLsbSplittedBuffer == nil)
    {
        _depthMsbLsbSplittedBuffer = (uint8_t*)malloc(depthPixelNum * 2);
    }
    
    uint8_t *msbPtr = _depthMsbLsbSplittedBuffer;
    uint8_t *lsbPtr = _depthMsbLsbSplittedBuffer + depthPixelNum;
    for (int i = 0; i < depthPixelNum; ++i)
    {
        *(lsbPtr++) = *(pDepthData++);
        *(msbPtr++) = *(pDepthData++);
    }
    int msbCompressedLength, lsbCompressedLength;
#if 1
    ImageTrans::CompressDepth((char *)_colorDepthCompressedBuffer + colorPixelNum + cbCompressedLength + crCompressedLength, msbCompressedLength, lsbCompressedLength,
                              (char *)_depthMsbLsbSplittedBuffer, depthPixelNum * 2);
#endif
#if 0
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthRows]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthCols]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",msbCompressedLength]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",lsbCompressedLength]];
#endif
#if 1
    [_socketClient sendCompressedColorAndDepth:(const uint8_t*)_colorDepthCompressedBuffer withRows:depthRows andCols:depthCols andCbCompressedLength:cbCompressedLength andCrCompressedLength:crCompressedLength andMsbCompressedLength:msbCompressedLength andLsbCompressedLength:lsbCompressedLength andFrameIdx: _frameIdx];
#endif
#endif
}

- (void)compressAndSendIrFrame:(STInfraredFrame *)irFrame ColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer
{
#if 1
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(yCbCrSampleBuffer);
    
    // get image size
    int colorCols = CVPixelBufferGetWidth(pixelBuffer);
    int colorRows = CVPixelBufferGetHeight(pixelBuffer);
    int colorPixelNum = colorRows * colorCols;
    
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // get y plane
    const uint8_t* yData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    
    // get cbCr plane
    const uint8_t* cbCrData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    
    int yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    int cbcrBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    assert( yBytesPerRow==cbcrBytesPerRow );
    
    int cbCrLength = (int)(colorRows * cbcrBytesPerRow / 2);
    if (_colorDepthCompressedBuffer == nil)
    {
        _colorDepthCompressedBuffer = (uint8_t*)malloc(colorPixelNum * 5);
    }
    if (_cbCrSplittedBuffer == nil)
    {
        _cbCrSplittedBuffer = (uint8_t*)malloc(cbCrLength);
    }
    
    memcpy(_colorDepthCompressedBuffer, yData, colorPixelNum);
    int cbCompressedLength, crCompressedLength;
    ImageTrans::CompressCbCr((char*)_colorDepthCompressedBuffer + colorPixelNum, cbCompressedLength, crCompressedLength,
                             (const char*)cbCrData, (char*)_cbCrSplittedBuffer, cbCrLength);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
#endif
    
#if 1
    size_t irCols = irFrame.width;
    size_t irRows = irFrame.height;
    int irPixelNum = irRows * irCols;
    uint8_t* pIrData = (uint8_t*)irFrame.data;
    
    if (_depthMsbLsbSplittedBuffer == nil)
    {
        _depthMsbLsbSplittedBuffer = (uint8_t*)malloc(irPixelNum * 2);
    }
    
    uint8_t *msbPtr = _depthMsbLsbSplittedBuffer;
    uint8_t *lsbPtr = _depthMsbLsbSplittedBuffer + irPixelNum;
    for (int i = 0; i < irPixelNum; ++i)
    {
        *(lsbPtr++) = *(pIrData++);
        *(msbPtr++) = *(pIrData++);
    }
    int msbCompressedLength, lsbCompressedLength;
#if 1
    ImageTrans::CompressDepth((char *)_colorDepthCompressedBuffer + colorPixelNum + cbCompressedLength + crCompressedLength, msbCompressedLength, lsbCompressedLength,
                              (char *)_depthMsbLsbSplittedBuffer, irPixelNum * 2);
#endif
#if 0
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthRows]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthCols]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",msbCompressedLength]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",lsbCompressedLength]];
#endif
#if 1
    [_socketClient sendCompressedColorAndIr:(const uint8_t*)_colorDepthCompressedBuffer withRows:irRows andCols:irCols andCbCompressedLength:cbCompressedLength andCrCompressedLength:crCompressedLength andMsbCompressedLength:msbCompressedLength andLsbCompressedLength:lsbCompressedLength andFrameIdx: _frameIdx];
#endif
#endif
}

- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame ColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer CurGravity: (Eigen::Vector3d&) curGravity Measurements: (std::vector<ImuMsg>&) measurements KeyFrameIdxEachFrag: (int)keyFrameIdxEachFrag
{
#if 1
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(yCbCrSampleBuffer);
    
    // get image size
    int colorCols = CVPixelBufferGetWidth(pixelBuffer);
    int colorRows = CVPixelBufferGetHeight(pixelBuffer);
    int colorPixelNum = colorRows * colorCols;
    
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // get y plane
    const uint8_t* yData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    
    // get cbCr plane
    const uint8_t* cbCrData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    
    int yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    int cbcrBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    assert( yBytesPerRow==cbcrBytesPerRow );
    
    int cbCrLength = (int)(colorRows * cbcrBytesPerRow / 2);
    if (_colorDepthCompressedBuffer == nil)
    {
        _colorDepthCompressedBuffer = (uint8_t*)malloc(colorPixelNum * 5);
    }
    if (_cbCrSplittedBuffer == nil)
    {
        _cbCrSplittedBuffer = (uint8_t*)malloc(cbCrLength);
    }

    memcpy(_colorDepthCompressedBuffer, yData, colorPixelNum);
    int cbCompressedLength, crCompressedLength;
    ImageTrans::CompressCbCr((char*)_colorDepthCompressedBuffer + colorPixelNum, cbCompressedLength, crCompressedLength,
                             (const char*)cbCrData, (char*)_cbCrSplittedBuffer, cbCrLength);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
#endif
    
#if 1
    size_t depthCols = depthFrame.width;
    size_t depthRows = depthFrame.height;
    int depthPixelNum = depthRows * depthCols;
    uint8_t* pDepthData = (uint8_t*)depthFrame.shiftData;
    //uint16_t* depthData = depthFrame.shiftData;

    if (_depthMsbLsbSplittedBuffer == nil)
    {
        _depthMsbLsbSplittedBuffer = (uint8_t*)malloc(depthPixelNum * 2);
    }
    
    uint8_t *msbPtr = _depthMsbLsbSplittedBuffer;
    uint8_t *lsbPtr = _depthMsbLsbSplittedBuffer + depthPixelNum;
    for (int i = 0; i < depthPixelNum; ++i)
    {
        *(lsbPtr++) = *(pDepthData++);
        *(msbPtr++) = *(pDepthData++);
    }
    int msbCompressedLength, lsbCompressedLength;
#if 1
    ImageTrans::CompressDepth((char *)_colorDepthCompressedBuffer + colorPixelNum + cbCompressedLength + crCompressedLength, msbCompressedLength, lsbCompressedLength,
                                 (char *)_depthMsbLsbSplittedBuffer, depthPixelNum * 2);
#endif
#if 0
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthRows]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthCols]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",msbCompressedLength]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",lsbCompressedLength]];
#endif
#if 1
    [_socketClient sendCompressedColorAndDepth:(const uint8_t*)_colorDepthCompressedBuffer withColorRows:colorRows andColorCols:colorCols andDepthRows:depthRows andDepthCols:depthCols andCbCompressedLength:cbCompressedLength andCrCompressedLength:crCompressedLength andMsbCompressedLength:msbCompressedLength andLsbCompressedLength:lsbCompressedLength andFrameIdx: _frameIdx andCurGravity: (char *)curGravity.data() andGravityElemSize: sizeof(curGravity) andMeasurements: (char *)measurements.data() andMeasurementElemSize: sizeof(measurements[0]) andMeasurementSize: measurements.size() andKeyFrameIdxEachFrag: keyFrameIdxEachFrag];
#endif
#endif
}

- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame ColorFrame648x484:(uint8_t*)yCbCrBuffer CurGravity: (Eigen::Vector3d&) curGravity Measurements: (std::vector<ImuMsg>&) measurements KeyFrameIdxEachFrag: (int)keyFrameIdxEachFrag
{
#if 1
    // get image size
    int colorCols = 648;
    int colorRows = 484;
    int colorPixelNum = colorRows * colorCols;
    
    // get y plane
    uint8_t* yData = yCbCrBuffer;
    
    // get cbCr plane
    uint8_t* cbCrData = yCbCrBuffer + colorRows * colorCols;
    
    int cbCrLength = (int)(colorRows * colorCols / 2);
    if (_colorDepthCompressedBuffer == nil)
    {
        _colorDepthCompressedBuffer = (uint8_t*)malloc(colorPixelNum * 5);
    }
    if (_cbCrSplittedBuffer == nil)
    {
        _cbCrSplittedBuffer = (uint8_t*)malloc(cbCrLength);
    }
    
    memcpy(_colorDepthCompressedBuffer, yData, colorPixelNum);
    int cbCompressedLength, crCompressedLength;
    ImageTrans::CompressCbCr((char*)_colorDepthCompressedBuffer + colorPixelNum, cbCompressedLength, crCompressedLength,
                             (const char*)cbCrData, (char*)_cbCrSplittedBuffer, cbCrLength);
#endif
    
#if 1
    size_t depthCols = depthFrame.width;
    size_t depthRows = depthFrame.height;
    int depthPixelNum = depthRows * depthCols;
    uint8_t* pDepthData = (uint8_t*)depthFrame.shiftData;
    //uint16_t* depthData = depthFrame.shiftData;
    
    if (_depthMsbLsbSplittedBuffer == nil)
    {
        _depthMsbLsbSplittedBuffer = (uint8_t*)malloc(depthPixelNum * 2);
    }
    
    uint8_t *msbPtr = _depthMsbLsbSplittedBuffer;
    uint8_t *lsbPtr = _depthMsbLsbSplittedBuffer + depthPixelNum;
    for (int i = 0; i < depthPixelNum; ++i)
    {
        *(lsbPtr++) = *(pDepthData++);
        *(msbPtr++) = *(pDepthData++);
    }
    int msbCompressedLength, lsbCompressedLength;
#if 1
    ImageTrans::CompressDepth((char *)_colorDepthCompressedBuffer + colorPixelNum + cbCompressedLength + crCompressedLength, msbCompressedLength, lsbCompressedLength,
                              (char *)_depthMsbLsbSplittedBuffer, depthPixelNum * 2);
#endif
#if 0
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthRows]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",depthCols]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",msbCompressedLength]];
    [self setMsgToTextView: [NSString stringWithFormat:@"%d\n",lsbCompressedLength]];
#endif
#if 1
    [_socketClient sendCompressedColorAndDepth:(const uint8_t*)_colorDepthCompressedBuffer withColorRows:colorRows andColorCols:colorCols andDepthRows:depthRows andDepthCols:depthCols andCbCompressedLength:cbCompressedLength andCrCompressedLength:crCompressedLength andMsbCompressedLength:msbCompressedLength andLsbCompressedLength:lsbCompressedLength andFrameIdx: _frameIdx andCurGravity: (char *)curGravity.data() andGravityElemSize: sizeof(curGravity) andMeasurements: (char *)measurements.data() andMeasurementElemSize: sizeof(measurements[0]) andMeasurementSize: measurements.size() andKeyFrameIdxEachFrag: keyFrameIdxEachFrag];
#endif
#endif
}

- (void)sendAllKeyFramesFrame
{
    for (int i = 0; i < _keyFrameVec.size(); ++i)
    {
        [_socketClient sendFullColor:(const uint8_t*)_keyFrameVec[i] withKeyFrameNum: _keyFrameVec.size() andFrameIdx: _keyFrameIdxVec[i] andColorRows:_keyFrameRows andColorCols:_keyFrameCols andColorBytesPerRow:_keyFrameBytePerRow];
        
        NSLog([NSString stringWithFormat:@"key frame blur score: %f", _keyFrameBlurScoreVec[i]]);
    }
}

- (void)renderColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer
{
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(yCbCrSampleBuffer);
    
    // get image size
    size_t cols = CVPixelBufferGetWidth(pixelBuffer);
    size_t rows = CVPixelBufferGetHeight(pixelBuffer);
    //NSLog([NSString stringWithFormat:@"row: %d col: %d\n",rows, cols]);
    
    // allocate memory for RGBA image for the first time
    if(_colorImageBuffer == nil)
        _colorImageBuffer = (uint8_t*)malloc(cols * rows * 4);
    
    // color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // get y plane
    const uint8_t* yData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    
    // get cbCr plane
    const uint8_t* cbCrData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    
    size_t yBytePerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    size_t cbcrBytePerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    assert( yBytePerRow==cbcrBytePerRow );
    
    uint8_t* bgra = _colorImageBuffer;
    
    bool ok = convertYCbCrToBGRA(cols, rows, yData, cbCrData, bgra, 0xff, yBytePerRow, cbcrBytePerRow, 4 * cols);
    
    if (!ok)
    {
        NSLog(@"YCbCr to BGRA conversion failed.");
        return;
    }
    
    NSData *data = [[NSData alloc] initWithBytes:_colorImageBuffer length:rows*cols*4];
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipFirst;
    bitmapInfo |= kCGBitmapByteOrder32Little;
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(
                                        cols,
                                        rows,
                                        8,
                                        8 * 4,
                                        cols*4,
                                        colorSpace,
                                        bitmapInfo,
                                        provider,
                                        NULL,
                                        false,
                                        kCGRenderingIntentDefault
                                        );
    
    _colorImageView.image = [[UIImage alloc] initWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
}

- (void)renderColorFrame648x484:(uint8_t*)yCbCrBuffer
{
    // get image size
    size_t cols = 648;
    size_t rows = 484;
    //NSLog([NSString stringWithFormat:@"row: %d col: %d\n",rows, cols]);
    
    // allocate memory for RGBA image for the first time
    if(_colorImageBuffer == nil)
        _colorImageBuffer = (uint8_t*)malloc(cols * rows * 4);
    
    // color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // get y plane
    const uint8_t* yData = yCbCrBuffer;
    
    // get cbCr plane
    const uint8_t* cbCrData = yCbCrBuffer + rows * cols;
    
    size_t yBytePerRow = cols;
    size_t cbcrBytePerRow = cols;
    assert( yBytePerRow==cbcrBytePerRow );
    
    uint8_t* bgra = _colorImageBuffer;
    
    bool ok = convertYCbCrToBGRA(cols, rows, yData, cbCrData, bgra, 0xff, yBytePerRow, cbcrBytePerRow, 4 * cols);
    
    if (!ok)
    {
        NSLog(@"YCbCr to BGRA conversion failed.");
        return;
    }
    
    NSData *data = [[NSData alloc] initWithBytes:_colorImageBuffer length:rows*cols*4];
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipFirst;
    bitmapInfo |= kCGBitmapByteOrder32Little;
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(
                                        cols,
                                        rows,
                                        8,
                                        8 * 4,
                                        cols*4,
                                        colorSpace,
                                        bitmapInfo,
                                        provider,
                                        NULL,
                                        false,
                                        kCGRenderingIntentDefault
                                        );
    
    _colorImageView.image = [[UIImage alloc] initWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
}

- (float)calcBlurScore:(uint8_t*)yBuffer
{
    size_t bytePerRow = 648;//_keyFrameBytePerRow;
    size_t rows = 484;//_keyFrameRows;
    size_t cols = 648;//_keyFrameCols;
    
    if(_bVer == nil)
    {
        _bVer = (float*)malloc(cols * rows * sizeof(float));
        _bHor = (float*)malloc(cols * rows * sizeof(float));
        _fVer = (float*)malloc(cols * rows * sizeof(float));
        _fHor = (float*)malloc(cols * rows * sizeof(float));
        memset(_bVer, 0, cols * rows * sizeof(float));
        memset(_bHor, 0, cols * rows * sizeof(float));
        memset(_fVer, 0, cols * rows * sizeof(float));
        memset(_fHor, 0, cols * rows * sizeof(float));
    }
   
    float scale = 1.0f / 5.0f;
    int left = 4, right = cols - 4;
    int top = 4, bottom = rows - 4;
    int currPos;
    int doubleCols = 2 * cols;
    for (int r = top; r < bottom; ++r)
    {
        for (int c = left; c < right; ++c)
        {
            currPos = r * cols + c;
            _bVer[currPos] =
            + yBuffer[currPos - 2]  * scale
            + yBuffer[currPos - 1]  * scale
            + yBuffer[currPos]  * scale
            + yBuffer[currPos + 1]  * scale
            + yBuffer[currPos + 2]  * scale;
            
            _bHor[currPos] =
            + yBuffer[currPos - doubleCols] * scale
            + yBuffer[currPos - cols] * scale
            + yBuffer[currPos] * scale
            + yBuffer[currPos + cols] * scale
            + yBuffer[currPos + doubleCols] * scale;
            
            _fVer[currPos] = abs((float)yBuffer[currPos] - (float)yBuffer[currPos - cols]);
            _fHor[currPos] = abs((float)yBuffer[currPos] - (float)yBuffer[currPos - 1]);
        }
    }
    
    double sumVVer = 0.0, sumVHor = 0.0, sumFVer = 0.0, sumFHor = 0.0;
    for (int r = top; r < bottom; ++r)
    {
        for (int c = left; c < right; ++c)
        {
            currPos = r * cols + c;
            sumFVer += _fVer[currPos];
            sumFHor += _fHor[currPos];
            sumVVer += fmaxf(0, _fVer[currPos]
                             - abs(_bVer[currPos] - _bVer[currPos - cols]));
            sumVHor += fmaxf(0, _fHor[currPos]
                             - abs(_bHor[currPos] - _bHor[currPos - 1]));
        }
    }
    
    return fmax(1.0 - sumVVer / sumFVer, 1.0 - sumVHor / sumFHor);
}

- (int)saveKeyFrame:(CMSampleBufferRef)yCbCrSampleBuffer AccordingTo:(float)blurScore
{
#if 0
    if (_keyFrameVec.size() == _frameIdx / FRAG_SIZE && _frameIdx % FRAG_SIZE > 5)
    {
        if (_frameIdx % FRAG_SIZE >= (FRAG_SIZE - 5))
        {
            acc = 0.0; // let it be key frame
        }
        if (acc > 0.5)
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
#endif
    
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(yCbCrSampleBuffer);
    
    // get image size
    size_t cols = CVPixelBufferGetWidth(pixelBuffer);
    size_t rows = CVPixelBufferGetHeight(pixelBuffer);
    //NSLog([NSString stringWithFormat:@"row: %d col: %d\n",rows, cols]);
    
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // get y plane
    const uint8_t* yData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    
    // get cbCr plane
    const uint8_t* cbCrData = reinterpret_cast<uint8_t*> (CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    
    size_t yBytePerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    size_t cbcrBytePerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    assert( yBytePerRow==cbcrBytePerRow );
    _keyFrameBytePerRow = yBytePerRow;
    _keyFrameRows = rows;
    _keyFrameCols = cols;
    
    uint8_t* buffer = (uint8_t*)malloc(yBytePerRow * rows + cbcrBytePerRow * rows / 2);
    memcpy(buffer, yData, yBytePerRow * rows);
    memcpy(buffer + yBytePerRow * rows, cbCrData, cbcrBytePerRow * rows / 2);
    
    _candidateKeyFrameVec.push_back(buffer);
    _candidateKeyFrameIdxVec.push_back(_frameIdx);
    _blurScoreVec.push_back(blurScore);
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    
    if (_frameIdx % FRAG_SIZE == (FRAG_SIZE - 1))
    {
        float minBlurScore = 1.0e24f;
        int minBlurIdx;
        for (int i = 0; i < _blurScoreVec.size(); ++i)
        {
            if (_blurScoreVec[i] < minBlurScore)
            {
                minBlurScore = _blurScoreVec[i];
                minBlurIdx = i;
                //minBlurIdx = 20;
            }
        }
        
        _keyFrameVec.push_back(_candidateKeyFrameVec[minBlurIdx]);
        _keyFrameIdxVec.push_back(_candidateKeyFrameIdxVec[minBlurIdx]);
        _keyFrameBlurScoreVec.push_back(_blurScoreVec[minBlurIdx]);
        
        for (int i = 0; i < _blurScoreVec.size(); ++i)
        {
            if (i != minBlurIdx)
            {
                free(_candidateKeyFrameVec[i]);
            }
        }
        _candidateKeyFrameVec.clear();
        _candidateKeyFrameIdxVec.clear();
        _blurScoreVec.clear();
        
        return minBlurIdx;
    }
    
    return 0;
}

#pragma mark - IMU
#if 1
ImuMsg curAcc;
std::queue<ImuMsg> imuMsgBuf;
int imuPrepare = 0;
std::vector<ImuMsg> gyroBuf;  // for Interpolation
- (void)imuStartUpdate
{
    CMMotionManager *motionManager = [[CMMotionManager alloc] init];
    if (!motionManager.accelerometerAvailable) {
        NSLog(@"没有加速计");
    }
#ifdef DATA_EXPORT
    motionManager.accelerometerUpdateInterval = 0.1;
    motionManager.gyroUpdateInterval = 0.1;
#else
    motionManager.accelerometerUpdateInterval = 0.01;
    motionManager.gyroUpdateInterval = 0.01;
#endif
    
    [motionManager startDeviceMotionUpdates];
#if 0
    [motionManager startDeviceMotionUpdatesToQueue:[NSOperationQueue currentQueue] withHandler:^(CMDeviceMotion *motion, NSError *error)
    {
        _curGravity << motion.gravity.x,
        motion.gravity.y,
        motion.gravity.z;
        
        if(imuPrepare < 10)
        {
            imuPrepare++;
        }
        
        curAcc.header = motion.timestamp;
        curAcc.acc << motion.userAcceleration.x * GRAVITY,
        motion.userAcceleration.y * GRAVITY,
        motion.userAcceleration.z * GRAVITY;
        
        ImuMsg gyroMsg;
        gyroMsg.header = curAcc.header;
        gyroMsg.gyr << motion.rotationRate.x,
        motion.rotationRate.y,
        motion.rotationRate.z;
        
        ImuMsg imuMsg;
        imuMsg.header = curAcc.header;
        imuMsg.acc = curAcc.acc;
        imuMsg.gyr = gyroMsg.gyr;
        
        _bufMtx.lock();
        imuMsgBuf.push(imuMsg);
        if (imuMsgBuf.size() > 20)
            imuMsgBuf.pop();
        //NSLog(@"IMU_buf timestamp %lf, acc_x = %lf",imu_msg_buf.front()->header,imu_msg_buf.front()->acc.x());
        _bufMtx.unlock();
        //_bufCon.notify_one();
    }];
#endif
    
#if 1
    [motionManager startAccelerometerUpdatesToQueue:[NSOperationQueue currentQueue]
                                        withHandler:^(CMAccelerometerData *latestAcc, NSError *error)
     {
         _curGravity << motionManager.deviceMotion.gravity.x,
         motionManager.deviceMotion.gravity.y,
         motionManager.deviceMotion.gravity.z;
         
         if(imuPrepare < 10)
         {
             imuPrepare++;
         }
         
         curAcc.header = latestAcc.timestamp;
#if 0
         curAcc.acc << latestAcc.acceleration.x * GRAVITY,
         latestAcc.acceleration.y * GRAVITY,
         latestAcc.acceleration.z * GRAVITY;
#endif
         curAcc.acc << motionManager.deviceMotion.userAcceleration.x * GRAVITY,
         motionManager.deviceMotion.userAcceleration.y * GRAVITY,
         motionManager.deviceMotion.userAcceleration.z * GRAVITY;
         //printf("imu acc update %lf %lf %lf %lf\n", acc_msg->header, acc_msg->acc.x(), acc_msg->acc.y(), acc_msg->acc.z());
         
     }];
    [motionManager startGyroUpdatesToQueue:[NSOperationQueue currentQueue] withHandler:^(CMGyroData *latestGyro, NSError *error)
     {
         //The time stamp is the amount of time in seconds since the device booted.
         NSTimeInterval header = latestGyro.timestamp;
         if(header <= 0)
             return;
         if(imuPrepare < 10)
             return;
         
         ImuMsg gyroMsg;
         gyroMsg.header = header;
#if 0
         gyroMsg.gyr << latestGyro.rotationRate.x,
         latestGyro.rotationRate.y,
         latestGyro.rotationRate.z;
#endif
         gyroMsg.gyr << motionManager.deviceMotion.rotationRate.x,
         motionManager.deviceMotion.rotationRate.y,
         motionManager.deviceMotion.rotationRate.z;
         
         if(gyroBuf.size() == 0)
         {
             gyroBuf.push_back(gyroMsg);
             gyroBuf.push_back(gyroMsg);
             return;
         }
         else
         {
             gyroBuf[0] = gyroBuf[1];
             gyroBuf[1] = gyroMsg;
         }
         //interpolation
         ImuMsg imuMsg;
         if(curAcc.header >= gyroBuf[0].header && curAcc.header < gyroBuf[1].header)
         {
             imuMsg.header = curAcc.header;
             imuMsg.acc = curAcc.acc;
             imuMsg.gyr = gyroBuf[0].gyr + (curAcc.header - gyroBuf[0].header)*(gyroBuf[1].gyr - gyroBuf[0].gyr)/(gyroBuf[1].header - gyroBuf[0].header);
             //printf("imu gyro update %lf %lf %lf\n", gyro_buf[0].header, imu_msg->header, gyro_buf[1].header);
             //printf("imu inte update %lf %lf %lf %lf\n", imu_msg->header, gyro_buf[0].gyr.x(), imu_msg->gyr.x(), gyro_buf[1].gyr.x());
         }
         else
         {
             //printf("imu error %lf %lf %lf\n", gyroBuf[0].header, curAcc.header, gyroBuf[1].header);
             return;
         }
         
         _bufMtx.lock();
         imuMsgBuf.push(imuMsg);
         if (imuMsgBuf.size() > 20)
             imuMsgBuf.pop();
         //NSLog(@"IMU_buf timestamp %lf, acc_x = %lf",imu_msg_buf.front()->header,imu_msg_buf.front()->acc.x());
         _bufMtx.unlock();
         //_bufCon.notify_one();
     }];
#endif
}

void getMeasurements(std::vector<ImuMsg> &measurements)
{
    measurements.clear();
    while (true)
    {
        if (imuMsgBuf.empty())
            return;
        
        measurements.push_back(imuMsgBuf.front());
        imuMsgBuf.pop();
    }
}
#endif

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    [self setMsgToTextView: @"Load The View\n"];
    
    _isStarted = false;
    _hasBeenPaused = false;
    _linearizeBuffer = nil;
    _coloredDepthBuffer = nil;
    _resizedColorImageBufferYCbCr = nil;
    _colorImageBuffer = nil;
    _yCbCrCompressedBuffer = nil;
    _cbCrSplittedBuffer = nil;
    _depthMsbLsbCompressedBuffer = nil;
    _depthMsbLsbSplittedBuffer = nil;
    _colorDepthCompressedBuffer = nil;
    _depthImageView.contentMode = UIViewContentModeScaleAspectFit;
    _colorImageView.contentMode = UIViewContentModeScaleAspectFit;
    _bVer = nil;
    _bHor = nil;

    _socketClient = [[SocketClient alloc]init];
    _socketClient._view = _modelView;
    _frameIdx = 0;
    _frameNumToSkip = 0;
    _skipFrameIdx = 0;
    
    [self imuStartUpdate];
    [self setupStructureSensor];
}

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
    
    [self connectToStructureSensorAndStartStreaming];
    
    // From now on, make sure we get notified when the app becomes active to restore the sensor state if necessary.
    [[NSNotificationCenter defaultCenter]
     addObserver:self
     selector:@selector(appDidBecomeActive)
     name:UIApplicationDidBecomeActiveNotification
     object:nil
     ];
}

- (void)appDidBecomeActive {
    [self setMsgToTextView: @"App Become Active\n"];
    
    [self connectToStructureSensorAndStartStreaming];
}

@end
