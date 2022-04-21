//
//  ImageTrans.h
//  cs
//
//  Created by Wen Jiang on 2/4/18.
//

#ifndef ImageTrans_h
#define ImageTrans_h
#include <cstring>
#include <iostream>

class ImageTrans
{
public:
#if 0
    static void split_YCbCr(char* origin_data, const int cols, const int rows);
    static void join_YCbCr(char* origin_data, int cols, int rows);
    static void split_Depth(char* origin_data, const int cols, const int rows);
    static void join_Depth(char* origin_data, int cols, int rows);
    static size_t compress_depth(char* depth_image, int cols, int rows, char* compressed_image, int& msb_compressed_size, int& lsb_encoded_lenth);
    static size_t decompress_depth(char* compressed_image, int cols, int rows, char *image_decompressd,int msb_compressed_lenth, int lsb_encoded_lenth);
#endif
	static void JoinDepth(char* msbLsbBuffer, char* msbLsbSplittedBuffer, int msbLsbLenth);
	static void CompressDepth(char* msbLsbCompressedBuffer, int& msbCompressedLength, int& lsbCompressedLength,
	                            char* msbLsbSplittedBuffer, int msbLsbLenth);
	static void DecompressDepth(char* msbLsbBuffer, int msbLsbLenth, char* msbLsbSplittedBuffer,
	                              char* msbLsbCompressedBuffer, const int msbCompressedLength,
	                              const int lsbCompressedLength);

	static void SplitCbCr(char* cbCrSplittedBuffer, const char* cbCrBuffer, int cbCrLenth);
	static void JoinCbCr(char* cbCrBuffer, char* cbCrSplittedBuffer, int cbCrLenth);
	static void CompressCbCr(char* cbCrCompressedBuffer, int& cbCompressedLength, int& crCompressedLength,
	                           const char* cbCrBuffer, char* cbCrSplittedBuffer, int cbCrLenth);
	static void DecompressCbCr(char* cbCrBuffer, int cbCrLenth, char* cbCrSplittedBuffer,
	                             char* cbCrCompressedBuffer, const int cbCompressedLength, const int crCompressedLength);
};

#endif /* ImageTrans_h */
