//
//  SocketClient.m
//  xObjReconCapture
//
//  Created by xujiamin on 2018/3/1.
//  Copyright © 2018年 xujiamin. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "SocketClient.h"

#define HEAD_SIZE 1024

enum PKG_HEAD_TYPE
{
    TEST_TYPE = 0,
    COLOR_FRAME_TYPE = 1,
    DEPTH_FRAME_TYPE = 2,
    DEPTH_COLOR_FRAME_TYPE = 3,
    IR_COLOR_FRAME_TYPE = 4,
    FULL_COLOR_FRAME_TYPE = 5,
};

@implementation SocketClient

-(id) init{
    _socket = nil;
    self._pkgHead = malloc(HEAD_SIZE);
    self._pkgContent = malloc(8000*800*4);
    self._receiveData = nil;
    
    return self;
}

- (void)dealloc
{
    _socket = nil;
    free(self._pkgHead);
    free(self._pkgContent);
}

-(NSError *)connectToClient {
    NSError *err = nil;
    if (nil == _socket)
        _socket = [[GCDAsyncSocket alloc] initWithDelegate:self delegateQueue:dispatch_get_main_queue()];
    if (![_socket connectToHost:@"192.168.31.8" onPort:(uint16_t)6000 error:&err]) {
        NSLog(@"Connection error : %@",err);
    } else {
        err = nil;
    }
    //[self._socket readDataWithTimeout:-1 tag:0];
    return err;
}

//判断是否是连接的状态
-(BOOL)isConnected {
    return _socket.isConnected;
}

//断开连接
-(void)disconnectClient {
    [_socket disconnect];
    _socket = nil;
}

-(void)setupConnection {
    if (![_socket isConnected]) {
        [self disconnectClient];
        [self connectToClient];
    }
}

- (void)socket:(GCDAsyncSocket *)sender didConnectToHost:(NSString *)host port:(UInt16)port
{
    NSLog(@"Cool, I'm connected! That was easy.");
    //[_socket readDataToData:[GCDAsyncSocket CRLFData] withTimeout:-1 tag:0];
    [_socket readDataToLength:648*484 withTimeout:-1 tag:0];
}

- (void)socket:(GCDAsyncSocket *)sender didReadData:(NSData *)data withTag:(long)tag
{
    NSLog(@"receiving data");
    //NSString* aStr = [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding];
    //NSLog(@"===%@",aStr);
    [self renderModelFrame:(uint8_t *)data.bytes withRows:484 andCols:648];
#if 0
    if (self._receiveData == nil) {
        //如果不存在，则创建一个新的
        self._receiveData = [[NSMutableData alloc] init];
    }
    NSString* aStr = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    NSLog(@"===%@",aStr);
    [self._receiveData appendData:data];//把接收到的数据添加上
    NSRange endRange = [aStr rangeOfString:@"\r\n"];
    if (endRange.location != NSNotFound) {
        //这里表示数据已经读取完成了
        //在if代码块中处理receiveData
        //(void)renderModelFrame:(uint8_t *)modelFrame withRows: (int)rows andCols: (int) cols
        self._receiveData = nil;//用于接受数据的对象置空
    }
#endif
    //[_socket readDataToData:[GCDAsyncSocket CRLFData] withTimeout:-1 tag:0];
    [_socket readDataToLength:648*484 withTimeout:-1 tag:0];
}

- (void)convertGrayToRGBA:(const uint8_t*)values grayValuesCount:(size_t)grayValuesCount
{
    for (size_t i = 0; i < grayValuesCount; i++)
    {
#if 1
        uint16_t value = values[i];
        self._coloredModelBuffer[4 * i + 0] = value;
        self._coloredModelBuffer[4 * i + 1] = value;
        self._coloredModelBuffer[4 * i + 2] = value;
        self._coloredModelBuffer[4 * i + 3] = 255;
#endif
    }
}

- (void)renderModelFrame:(uint8_t *)modelFrame withRows: (int)rows andCols: (int) cols
{
    if (self._coloredModelBuffer == nil)
    {
        self._coloredModelBuffer = (uint8_t*)malloc(cols * rows * 4);
    }
    
    [self convertGrayToRGBA:modelFrame grayValuesCount:cols * rows];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
    bitmapInfo |= kCGBitmapByteOrder32Big;
    
    NSData *data = [NSData dataWithBytes:self._coloredModelBuffer length:cols * rows * 4];
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
    self._view.image = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
}

- (BOOL) sendHello{
    uint8_t pkgHeadType = TEST_TYPE;
    //uint64_t time_Stamp = [[NSDate date] timeIntervalSince1970]*1000;
    //NSLog(@"%d", time_Stamp);
    memset(self._pkgHead, 0, HEAD_SIZE);
    memcpy(self._pkgHead, &pkgHeadType, sizeof(pkgHeadType));
    sprintf(self._pkgHead + sizeof(pkgHeadType), "%s", "Hello_this_is_iPad_client");
    //NSData *colorData = [[NSData alloc] initWithByteww    es:_pkgHead length:HEAD_SIZE];
    [_socket writeData:[NSData dataWithBytes:self._pkgHead length:HEAD_SIZE] withTimeout:-1 tag:TEST_TYPE];
    //[_socket writeData:colorData withTimeout:-1 tag:TEST_TYPE];
    return true;
}

inline const char* writeData(const char* dst, const char* src, int size)
{
    memcpy(dst, src, size);
    
    return dst + size;
}
inline const char* readData(const char* dst, const char* src, int size)
{
    memcpy(dst, src, size);
    
    return src + size;
}

-(BOOL) sendCompressedColor:(const uint8_t*)yCbCrCompressData withRows:(int)rows andCols:(int)cols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andFrameIdx: (int)frameIdx
{
    uint8_t pkgHeadType = COLOR_FRAME_TYPE;
    memset(self._pkgHead, 0, HEAD_SIZE);
    //memcpy(_pkgHead, &pkgHeadType, sizeof(pkgHeadType));
    //sprintf(_pkgHead + sizeof(pkgHeadType), "%d %d %d %d %d", rows, cols, cbCompressedLength, crCompressedLength, frameIdx);
    char* pCur = self._pkgHead;
    pCur = writeData(pCur, (char *)&pkgHeadType, sizeof(pkgHeadType));
    pCur = writeData(pCur, (char *)&rows, sizeof(rows));
    pCur = writeData(pCur, (char *)&cols, sizeof(cols));
    pCur = writeData(pCur, (char *)&cbCompressedLength, sizeof(cbCompressedLength));
    pCur = writeData(pCur, (char *)&crCompressedLength, sizeof(crCompressedLength));
    pCur = writeData(pCur, (char *)&frameIdx, sizeof(frameIdx));
    [_socket writeData:[NSData dataWithBytes:self._pkgHead length:HEAD_SIZE] withTimeout:-1 tag:COLOR_FRAME_TYPE];
    int yCbCrCompressLength = rows * cols + cbCompressedLength + crCompressedLength;
    [_socket writeData:[NSData dataWithBytes:yCbCrCompressData length:yCbCrCompressLength] withTimeout:-1 tag:COLOR_FRAME_TYPE];
    
    return true;
}

-(BOOL) sendCompressedDepth:(const uint8_t*)msbLsbCompressData withRows:(int)rows andCols:(int)cols andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx
{
    uint8_t pkgHeadType = DEPTH_FRAME_TYPE;
    memset(self._pkgHead, 0, HEAD_SIZE);
    //memcpy(_pkgHead, &pkgHeadType, sizeof(pkgHeadType));
    //sprintf(_pkgHead + sizeof(pkgHeadType), "%d %d %d %d %d", rows, cols, msbCompressedLength, lsbCompressedLength, frameIdx);
    char* pCur = self._pkgHead;
    pCur = writeData(pCur, (char *)&pkgHeadType, sizeof(pkgHeadType));
    pCur = writeData(pCur, (char *)&rows, sizeof(rows));
    pCur = writeData(pCur, (char *)&cols, sizeof(cols));
    pCur = writeData(pCur, (char *)&msbCompressedLength, sizeof(msbCompressedLength));
    pCur = writeData(pCur, (char *)&lsbCompressedLength, sizeof(lsbCompressedLength));
    pCur = writeData(pCur, (char *)&frameIdx, sizeof(frameIdx));
    [_socket writeData:[NSData dataWithBytes:self._pkgHead length:HEAD_SIZE] withTimeout:-1 tag:DEPTH_FRAME_TYPE];
    int msbLsbCompressLength = msbCompressedLength + lsbCompressedLength;
    [_socket writeData:[NSData dataWithBytes:msbLsbCompressData length:msbLsbCompressLength] withTimeout:-1 tag:DEPTH_FRAME_TYPE];
    
    return true;
}

-(BOOL) sendCompressedColorAndDepth:(const uint8_t*)colorDepthCompressData withRows:(int)rows andCols:(int)cols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx
{
    uint8_t pkgHeadType = DEPTH_COLOR_FRAME_TYPE;
    memset(self._pkgHead, 0, HEAD_SIZE);
#if 0
    memcpy(_pkgHead, &pkgHeadType, sizeof(pkgHeadType));
    sprintf(_pkgHead + sizeof(pkgHeadType), "%d %d %d %d %d %d %d", rows, cols, cbCompressedLength, crCompressedLength, msbCompressedLength, lsbCompressedLength, frameIdx);
#endif
#if 1
    char* pCur = self._pkgHead;
    pCur = writeData(pCur, (char *)&pkgHeadType, sizeof(pkgHeadType));
    pCur = writeData(pCur, (char *)&rows, sizeof(rows));
    pCur = writeData(pCur, (char *)&cols, sizeof(cols));
    pCur = writeData(pCur, (char *)&cbCompressedLength, sizeof(cbCompressedLength));
    pCur = writeData(pCur, (char *)&crCompressedLength, sizeof(crCompressedLength));
    pCur = writeData(pCur, (char *)&msbCompressedLength, sizeof(msbCompressedLength));
    pCur = writeData(pCur, (char *)&lsbCompressedLength, sizeof(lsbCompressedLength));
    pCur = writeData(pCur, (char *)&frameIdx, sizeof(frameIdx));
#endif
    [_socket writeData:[NSData dataWithBytes:self._pkgHead length:HEAD_SIZE] withTimeout:-1 tag:DEPTH_COLOR_FRAME_TYPE];
    int colorDepthCompressLength = rows * cols + cbCompressedLength + crCompressedLength + msbCompressedLength + lsbCompressedLength;
    [_socket writeData:[NSData dataWithBytes:colorDepthCompressData length:colorDepthCompressLength] withTimeout:-1 tag:DEPTH_COLOR_FRAME_TYPE];
    
    return true;
}

-(BOOL) sendCompressedColorAndIr:(const uint8_t*)colorDepthCompressData withRows:(int)rows andCols:(int)cols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx
{
    uint8_t pkgHeadType = IR_COLOR_FRAME_TYPE;
    memset(self._pkgHead, 0, HEAD_SIZE);
#if 0
    memcpy(_pkgHead, &pkgHeadType, sizeof(pkgHeadType));
    sprintf(_pkgHead + sizeof(pkgHeadType), "%d %d %d %d %d %d %d", rows, cols, cbCompressedLength, crCompressedLength, msbCompressedLength, lsbCompressedLength, frameIdx);
#endif
#if 1
    char* pCur = self._pkgHead;
    pCur = writeData(pCur, (char *)&pkgHeadType, sizeof(pkgHeadType));
    pCur = writeData(pCur, (char *)&rows, sizeof(rows));
    pCur = writeData(pCur, (char *)&cols, sizeof(cols));
    pCur = writeData(pCur, (char *)&cbCompressedLength, sizeof(cbCompressedLength));
    pCur = writeData(pCur, (char *)&crCompressedLength, sizeof(crCompressedLength));
    pCur = writeData(pCur, (char *)&msbCompressedLength, sizeof(msbCompressedLength));
    pCur = writeData(pCur, (char *)&lsbCompressedLength, sizeof(lsbCompressedLength));
    pCur = writeData(pCur, (char *)&frameIdx, sizeof(frameIdx));
#endif
    [_socket writeData:[NSData dataWithBytes:self._pkgHead length:HEAD_SIZE] withTimeout:-1 tag:IR_COLOR_FRAME_TYPE];
    int colorDepthCompressLength = rows * cols + cbCompressedLength + crCompressedLength + msbCompressedLength + lsbCompressedLength;
    [_socket writeData:[NSData dataWithBytes:colorDepthCompressData length:colorDepthCompressLength] withTimeout:-1 tag:IR_COLOR_FRAME_TYPE];
    
    return true;
}

-(BOOL) sendCompressedColorAndDepth:(const uint8_t*)colorDepthCompressData withColorRows:(int)colorRows andColorCols:(int)colorCols andDepthRows:(int)depthRows andDepthCols:(int)depthCols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx andCurGravity: (char *)curGravity andGravityElemSize: (int)gravityElemSize andMeasurements: (char *)measurements andMeasurementElemSize: (int)measurementElemSize andMeasurementSize: (int)measurementSize andKeyFrameIdxEachFrag: (int)keyFrameIdxEachFrag;
{
    uint8_t pkgHeadType = DEPTH_COLOR_FRAME_TYPE;
    memset(self._pkgHead, 0, HEAD_SIZE);
#if 0
    memcpy(_pkgHead, &pkgHeadType, sizeof(pkgHeadType));
    sprintf(_pkgHead + sizeof(pkgHeadType), "%d %d %d %d %d %d %d", rows, cols, cbCompressedLength, crCompressedLength, msbCompressedLength, lsbCompressedLength, frameIdx);
#endif
#if 1
    char* pCur = self._pkgHead;
    pCur = writeData(pCur, (char *)&pkgHeadType, sizeof(pkgHeadType));
    pCur = writeData(pCur, (char *)&keyFrameIdxEachFrag, sizeof(keyFrameIdxEachFrag));
    pCur = writeData(pCur, (char *)&colorRows, sizeof(colorRows));
    pCur = writeData(pCur, (char *)&colorCols, sizeof(colorCols));
    pCur = writeData(pCur, (char *)&depthRows, sizeof(depthRows));
    pCur = writeData(pCur, (char *)&depthCols, sizeof(depthCols));
    pCur = writeData(pCur, (char *)&cbCompressedLength, sizeof(cbCompressedLength));
    pCur = writeData(pCur, (char *)&crCompressedLength, sizeof(crCompressedLength));
    pCur = writeData(pCur, (char *)&msbCompressedLength, sizeof(msbCompressedLength));
    pCur = writeData(pCur, (char *)&lsbCompressedLength, sizeof(lsbCompressedLength));
    pCur = writeData(pCur, (char *)&frameIdx, sizeof(frameIdx));
    
    pCur = writeData(pCur, (char *)&gravityElemSize, sizeof(gravityElemSize));
    pCur = writeData(pCur, (char *)curGravity, gravityElemSize);
    pCur = writeData(pCur, (char *)&measurementElemSize, sizeof(measurementElemSize));
    pCur = writeData(pCur, (char *)&measurementSize, sizeof(measurementSize));
    pCur = writeData(pCur, (char *)measurements, measurementElemSize * measurementSize);
    
#endif
    [_socket writeData:[NSData dataWithBytes:self._pkgHead length:HEAD_SIZE] withTimeout:-1 tag:DEPTH_COLOR_FRAME_TYPE];
    int colorDepthCompressLength = colorRows * colorCols + cbCompressedLength + crCompressedLength + msbCompressedLength + lsbCompressedLength;
    [_socket writeData:[NSData dataWithBytes:colorDepthCompressData length:colorDepthCompressLength] withTimeout:-1 tag:DEPTH_COLOR_FRAME_TYPE];
    
    return true;
}

-(BOOL) sendFullColor:(const uint8_t*)fullColorData withKeyFrameNum:(int)keyFrameNum andFrameIdx:(int)frameIdx andColorRows:(int)colorRows andColorCols:(int)colorCols andColorBytesPerRow:(int)colorBytesPerRow
{
    uint8_t pkgHeadType = FULL_COLOR_FRAME_TYPE;
    memset(self._pkgHead, 0, HEAD_SIZE);

    char* pCur = self._pkgHead;
    pCur = writeData(pCur, (char *)&pkgHeadType, sizeof(pkgHeadType));
    pCur = writeData(pCur, (char *)&keyFrameNum, sizeof(keyFrameNum));
    pCur = writeData(pCur, (char *)&frameIdx, sizeof(frameIdx));
    pCur = writeData(pCur, (char *)&colorRows, sizeof(colorRows));
    pCur = writeData(pCur, (char *)&colorCols, sizeof(colorCols));
    pCur = writeData(pCur, (char *)&colorBytesPerRow, sizeof(colorBytesPerRow));
    
    [_socket writeData:[NSData dataWithBytes:self._pkgHead length:HEAD_SIZE] withTimeout:-1 tag:FULL_COLOR_FRAME_TYPE];
    int fullColorLength = 1.5 * colorRows * colorBytesPerRow;
    [_socket writeData:[NSData dataWithBytes:fullColorData length:fullColorLength] withTimeout:-1 tag:FULL_COLOR_FRAME_TYPE];
    
    return true;
}

- (void)socket:(GCDAsyncSocket *)sock didWriteDataWithTag:(long)tag
{
#if 0
    if(tag == COLOR_TAG){
        //        NSLog(@"Did Wirte%.0f %d", [[NSDate date] timeIntervalSince1970]*1000, sended_cnt);
        sended_cnt--;
    }
    //    NSLog(@"did wirte %ld", tag);
#endif
}

@end
