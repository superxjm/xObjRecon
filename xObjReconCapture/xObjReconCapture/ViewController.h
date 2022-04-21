//
//  ViewController.h
//  xObjReconCapture
//
//  Created by xujiamin on 2018/3/1.
//  Copyright © 2018年 xujiamin. All rights reserved.
//

#import <AVFoundation/AVFoundation.h>
#define HAS_LIBCXX
#import <Structure/Structure.h>
#import <UIKit/UIKit.h>
#import <CoreMotion/CoreMotion.h>
#import <GLKit/GLKit.h>
#import <queue>
#import <Eigen/Eigen>

#import "SocketClient.h"

#define GRAVITY ((double)9.805)

struct Options
{
    // Whether we should use depth aligned to the color viewpoint when Structure Sensor was calibrated.
    // This setting may get overwritten to false if no color camera can be used.
    bool useHardwareRegisteredDepth = false;
    
    // Whether to enable an expensive per-frame depth accuracy refinement.
    // Note: this option requires useHardwareRegisteredDepth to be set to false.
    const bool applyExpensiveCorrectionToDepth = true;
    
    // Focus position for the color camera (between 0 and 1). Must remain fixed one depth streaming
    // has started when using hardware registered depth.
    const float lensPosition = 0.75f;
    
    bool highResColoring;
};

const int FRAG_SIZE = 50;

struct ImuMsg {
    NSTimeInterval header;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
};

@interface ViewController : UIViewController {
    volatile BOOL _isStarted;
    volatile BOOL _hasBeenPaused;
    SocketClient* _socketClient;
    
    __weak IBOutlet UIImageView *_colorImageView;
    __weak IBOutlet UIImageView *_depthImageView;
    __weak IBOutlet UIImageView *_modelView;
    uint16_t *_linearizeBuffer;
    uint8_t *_coloredDepthBuffer;
    uint8_t *_colorImageBuffer;
    uint8_t *_resizedColorImageBufferYCbCr;
    uint8_t *_colorImageBufferOnlyY;
    
    uint8_t *_yCbCrCompressedBuffer;
    uint8_t *_cbCrSplittedBuffer;
    uint8_t *_depthMsbLsbCompressedBuffer;
    uint8_t *_depthMsbLsbSplittedBuffer;
    uint8_t *_colorDepthCompressedBuffer;
    
    STSensorController* _sensorController;
    STStreamConfig _structureStreamConfig;
    
    // IMU handling.
    CMMotionManager* _motionManager;
    std::mutex _bufMtx;
    std::condition_variable _bufCon;
    std::vector<ImuMsg> _measurements;
    Eigen::Vector3d _curGravity;
    
    Options _options;
    
    int _frameIdx;
    int _skipFrameIdx, _frameNumToSkip;
    std::vector<uint8_t*> _keyFrameVec;
    std::vector<int> _keyFrameIdxVec;
    std::vector<uint8_t*> _candidateKeyFrameVec;
    std::vector<int> _candidateKeyFrameIdxVec;
    std::vector<float> _blurScoreVec;
    std::vector<float> _keyFrameBlurScoreVec;
    int _keyFrameBytePerRow;
    int _keyFrameRows;
    int _keyFrameCols;
    
    float *_bVer, *_bHor, *_fVer, *_fHor;
}

- (void) setMsgToTextView:(NSString *)info;
- (void) clearAndSetMsgToTextView:(NSString *)info;

@property (nonatomic, retain) AVCaptureSession* _avCaptureSession;
@property (nonatomic, retain) AVCaptureDevice* _videoDevice;

@property (weak, nonatomic) IBOutlet UITextView* _infoTextView;

- (void)renderDepthFrame:(STDepthFrame*)depthFrame;
- (void)renderIrFrame:(STInfraredFrame *)irFrame;
- (void)renderColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer;
- (void)renderColorFrame648x484:(uint8_t*)yCbCrBuffer;
- (void)compressAndSendColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer;
- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame;
- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame ColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer;
- (void)compressAndSendIrFrame:(STInfraredFrame *)irFrame ColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer;
- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame ColorFrame:(CMSampleBufferRef)yCbCrSampleBuffer CurGravity: (Eigen::Vector3d&) curGravity Measurements: (std::vector<ImuMsg>&) measurements KeyFrameIdxEachFrag: (int)keyFrameIdxEachFrag;
- (void)compressAndSendDepthFrame:(STDepthFrame *)depthFrame ColorFrame648x484:(uint8_t*)yCbCrBuffer CurGravity: (Eigen::Vector3d&) curGravity Measurements: (std::vector<ImuMsg>&) measurements KeyFrameIdxEachFrag: (int)keyFrameIdxEachFrag;
- (void)sendAllKeyFramesFrame;

- (float)calcBlurScore:(uint8_t*)yCbCrBuffer;
- (int)saveKeyFrame:(CMSampleBufferRef)yCbCrSampleBuffer AccordingTo:(float)acc;

void getMeasurements(std::vector<ImuMsg> &measurements);

@end

