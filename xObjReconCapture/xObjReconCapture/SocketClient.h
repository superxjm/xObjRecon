//
//  SocketClient.h
//  xObjReconCapture
//
//  Created by xujiamin on 2018/3/1.
//  Copyright © 2018年 xujiamin. All rights reserved.
//

#ifndef SocketClient_h
#define SocketClient_h

#import <CocoaAsyncSocket/GCDAsyncSocket.h>
#import <UIKit/UIKit.h>

@interface SocketClient : NSObject//<GCDAsyncSocketDelegate>:GCDAsyncSocket
{
    GCDAsyncSocket* _socket;
}
@property NSMutableData *_receiveData;
@property char* _pkgHead;
@property char* _pkgContent;
@property __weak UIImageView *_view;
@property uint8_t *_coloredModelBuffer;

//@property (nonatomic, retain) NSMutableData *_receiveData;

-(NSError *)connectToClient;
-(BOOL)isConnected;
-(void)disconnectClient;
-(void)setupConnection;

-(id) init;
-(void) socket:(GCDAsyncSocket *)sender didConnectToHost:(NSString *)host port:(UInt16)port;
-(BOOL) sendHello;
-(BOOL) sendCompressedColor:(const uint8_t*)yCbCrCompressData withRows:(int)rows andCols:(int)cols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andFrameIdx: (int)frameIdx;
-(BOOL) sendCompressedDepth:(const uint8_t*)msbLsbCompressData withRows:(int)rows andCols:(int)cols andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx;
-(BOOL) sendCompressedColorAndDepth:(const uint8_t*)colorDepthCompressData withRows:(int)rows andCols:(int)cols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx;
-(BOOL) sendCompressedColorAndIr:(const uint8_t*)colorDepthCompressData withRows:(int)rows andCols:(int)cols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx;
-(BOOL) sendCompressedColorAndDepth:(const uint8_t*)colorDepthCompressData withColorRows:(int)colorRows andColorCols:(int)colorCols andDepthRows:(int)depthRows andDepthCols:(int)depthCols andCbCompressedLength:(int)cbCompressedLength andCrCompressedLength:(int)crCompressedLength andMsbCompressedLength:(int)msbCompressedLength andLsbCompressedLength:(int)lsbCompressedLength andFrameIdx: (int)frameIdx andCurGravity: (char *)curGravity andGravityElemSize: (int)gravityElemSize andMeasurements: (char *)measurements andMeasurementElemSize: (int)measurementElemSize andMeasurementSize: (int)measurementSize andKeyFrameIdxEachFrag: (int)keyFrameIdxEachFrag;
-(BOOL) sendFullColor:(const uint8_t*)fullColorData withKeyFrameNum:(int)keyFrameNum andFrameIdx:(int)frameIdx andColorRows:(int)colorRows andColorCols:(int)colorCols andColorBytesPerRow:(int)colorBytesPerRow;

- (void)socket:(GCDAsyncSocket *)sock didReadData:(NSData *)data withTag:(long)tag;
- (void)renderModelFrame:(uint8_t *)modelFrame withRows: (int)rows andCols: (int) cols;

@end

#endif /* SocketClient_h */
