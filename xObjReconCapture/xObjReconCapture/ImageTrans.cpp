//
//  ImageTrans.cpp
//  cs
//
//  Created by Wen Jiang on 2/4/18.
//

#include "ImageTrans.h"
#include "DepthCompress.hpp"
#include "FastLZ.h"

#if 0
void ImageTrans::split_YCbCr(char* origin_data, const int cols, const int rows) {
    int image_size = cols * rows * 3;
    char* temp = (char*)malloc(image_size);
    char* p_Y = temp;
    char* p_Cb = temp + cols*rows;
    char *p_Cr = temp + 2 * cols*rows;
    char *p_origin = origin_data;
    while (p_origin < origin_data + image_size) {
        *(p_Y++) = *(p_origin++);
        *(p_Cb++) = *(p_origin++);
        *(p_Cr++) = *(p_origin++);
    }
    memcpy(origin_data, temp, image_size);
    free(temp);
}
void ImageTrans::join_YCbCr(char* origin_data, int cols, int rows) {
    int image_size = cols * rows * 3;
    char* temp = (char*)malloc(image_size);
    char* p_Y = origin_data;
    char* p_Cb = origin_data + cols*rows;
    char *p_Cr = origin_data + 2 * cols*rows;
    char *p_origin = temp;
    while (p_origin < temp + image_size) {
        *(p_origin++) = *(p_Y++);
        *(p_origin++) = *(p_Cb++);
        *(p_origin++) = *(p_Cr++);
    }
    memcpy(origin_data, temp, image_size);
    free(temp);
}
#endif

#if 0
void ImageTrans::split_Depth(char* origin_data, const int cols, const int rows) {
    int image_size = cols * rows * 2;
    char* temp = (char*)malloc(image_size);
    char* p_L = temp;
    char* p_M = temp + cols*rows;
    char *p_origin = origin_data;
    while (p_origin < origin_data + image_size) {
        *(p_L++) = *(p_origin++);
        *(p_M++) = *(p_origin++);
    }
    memcpy(origin_data, temp, image_size);
    free(temp);
}
#endif

void ImageTrans::JoinDepth(char* msbLsbBuffer, char* msbLsbSplittedBuffer, int msbLsbLenth)
{
    char* pM = msbLsbSplittedBuffer;
    char* pL = msbLsbSplittedBuffer + msbLsbLenth / 2;
    char *ptr = msbLsbBuffer;
    while (ptr < msbLsbBuffer + msbLsbLenth)
    {
        *(ptr++) = *(pL++);
        *(ptr++) = *(pM++);
    }
}
void ImageTrans::CompressDepth(char* msbLsbCompressedBuffer, int& msbCompressedLength, int& lsbCompressedLength,
                                 char* msbLsbSplittedBuffer, int msbLsbLenth)
{
    msbCompressedLength = fastlz_compress(msbLsbSplittedBuffer, msbLsbLenth / 2, msbLsbCompressedBuffer);
#if 1
    lsbCompressedLength = fastlz_compress(msbLsbSplittedBuffer + msbLsbLenth / 2, msbLsbLenth / 2, msbLsbCompressedBuffer + msbCompressedLength);
#endif
#if 0
    lsbCompressedLength = encode((const uint8_t *)msbLsbSplittedBuffer + msbLsbLenth / 2, msbLsbLenth / 2, (uint8_t *)msbLsbCompressedBuffer + msbCompressedLength, uint32_t(msbLsbLenth / 2));
#endif
}
void ImageTrans::DecompressDepth(char* msbLsbBuffer,  int msbLsbLenth, char* msbLsbSplittedBuffer,
                                    char* msbLsbCompressedBuffer, const int msbCompressedLength, const int lsbCompressedLength)
{
    int size1 = fastlz_decompress(msbLsbCompressedBuffer, msbCompressedLength, msbLsbSplittedBuffer, msbLsbLenth / 2);
#if 0
    int size2 = fastlz_decompress(msbLsbCompressedBuffer + msbCompressedLength, lsbCompressedLength, msbLsbSplittedBuffer + msbLsbLenth / 2, msbLsbLenth / 2);
    assert(size1 == msbLsbLenth / 2);
    assert(size2 == msbLsbLenth / 2);
#endif
#if 1
	decode((unsigned char*)msbLsbCompressedBuffer + msbCompressedLength, lsbCompressedLength, msbLsbLenth / 2, (unsigned char*)msbLsbSplittedBuffer + msbLsbLenth / 2);
#endif
#if 1
	JoinDepth(msbLsbBuffer, msbLsbSplittedBuffer, msbLsbLenth);
#endif
}

void ImageTrans::SplitCbCr(char* cbCrSplittedBuffer, const char* cbCrBuffer, int cbCrLenth)
{
    char* pCb = cbCrSplittedBuffer;
    char* pCr = cbCrSplittedBuffer + cbCrLenth / 2;
    const char *ptr = cbCrBuffer;
    while (ptr < cbCrBuffer + cbCrLenth)
    {
        *(pCb++) = *(ptr++);
        *(pCr++) = *(ptr++);
    }
}
void ImageTrans::JoinCbCr(char* cbCrBuffer, char* cbCrSplittedBuffer, int cbCrLenth)
{
    char* pL = cbCrSplittedBuffer;
    char* pM = cbCrSplittedBuffer + cbCrLenth / 2;
    char *ptr = cbCrBuffer;
    while (ptr < cbCrBuffer + cbCrLenth)
    {
        *(ptr++) = *(pL++);
        *(ptr++) = *(pM++);
    }
}
void ImageTrans::CompressCbCr(char* cbCrCompressedBuffer, int& cbCompressedLength, int& crCompressedLength,
                                const char* cbCrBuffer, char* cbCrSplittedBuffer, int cbCrLenth)
{
    ImageTrans::SplitCbCr(cbCrSplittedBuffer, cbCrBuffer, cbCrLenth);
    cbCompressedLength = fastlz_compress(cbCrSplittedBuffer, cbCrLenth / 2, cbCrCompressedBuffer);
    crCompressedLength = fastlz_compress(cbCrSplittedBuffer + cbCrLenth / 2, cbCrLenth / 2, cbCrCompressedBuffer + cbCompressedLength);
}
void ImageTrans::DecompressCbCr(char* cbCrBuffer,  int cbCrLenth, char* cbCrSplittedBuffer,
                                  char* cbCrCompressedBuffer, const int cbCompressedLength, const int crCompressedLength)
{
    fastlz_decompress(cbCrCompressedBuffer, cbCompressedLength, cbCrSplittedBuffer, cbCrLenth / 2);
    fastlz_decompress(cbCrCompressedBuffer + cbCompressedLength, crCompressedLength , cbCrSplittedBuffer + cbCrLenth / 2, cbCrLenth / 2) ;
    ImageTrans::JoinCbCr(cbCrBuffer, cbCrSplittedBuffer, cbCrLenth);
}


