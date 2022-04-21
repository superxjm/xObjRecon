# pragma once
#include "./BitStream.hpp"
#include <functional>
typedef uint16_t uint16;
typedef uint8_t uint8;
// TYPICAL COMPRESSION RATIO ON 640x480 test image:  0.17
// DECODE:
// Step 0. Last value is initialized to 0.  Frame size is known in advance.
// Step 1. Proceed by decoding following bitstream until all pixels are decoded.
// 00 - Next value is same as last value.
// 11 - Next value is last value + 1.
// 10 - Next value is last value - 1.
// 010 - bbbbb - Next N values are same as last value.  (N encoded w/ 5 bits)
// 0111 - bbbbbbbbbbb - Next value is X.  (X encoded w/ 11 bits)
// 01101 - Next value is last value + 2.
// 01100 - Next value is last value - 2.
const int continue_threshold = 5;
inline uint8 * decode(const uint8 * bitstream_data, unsigned int bitstream_length_bytes, int numelements, uint8* output) {

	uint8_t lastVal = 0;
	uint8_t curVal = 0;

	bitstream_t bs;
	bs_init(&bs);
	bs_attach(&bs, const_cast<uint8_t*>(bitstream_data), bitstream_length_bytes);

	uint8_t * depthimage = 0 == output
		? (uint8_t*)malloc(numelements * sizeof(uint8_t))
		: output
		;

	uint8_t * depth_ptr = depthimage;

	while (numelements > 0) {
		uint8_t bit0 = bs_get(&bs, 1);
		uint8_t bit1 = bs_get(&bs, 1);

		if (bit0 == 0 && bit1 == 0) // 00
		{
			curVal = lastVal;

			*(depth_ptr++) = curVal; lastVal = curVal;

			numelements -= 1;
		} else if (bit0 == 1) // 1 prefix
		{

			if (bit1 == 0) {
				curVal = lastVal - 1;
			} else {
				curVal = lastVal + 1;
			}

			*(depth_ptr++) = curVal; lastVal = curVal;

			numelements -= 1;

		} else  // must be 01 prefix
		{
			uint8_t bit2 = bs_get(&bs, 1);

			if (bit2 == 0) // 010 --> multiple zeros!
			{
				uint16_t numZeros = bs_get(&bs, continue_threshold);

				numZeros += continue_threshold; // We never encode less than continue_threshold.

				for (int i = 0; i < numZeros; i++) {
					*(depth_ptr++) = curVal;
				}

				numelements -= numZeros;
			} else {
				uint8_t bit3 = bs_get(&bs, 1);

				if (bit3 == 0) // 0110 -- DELTA!
				{
					uint8_t delta_bit = bs_get(&bs, 1);

					if (delta_bit == 0) {
						curVal = lastVal - 2;
					} else {
						curVal = lastVal + 2;
					}

					*(depth_ptr++) = curVal; lastVal = curVal;

					numelements -= 1;

				} else // 0111 -- RESET!
				{
					uint8_t value = bs_get(&bs, 8); // 8 bits total.

					curVal = value;

					*(depth_ptr++) = curVal; lastVal = curVal;
					numelements -= 1;
				}

			}

		}

	}

	return depthimage;

}

//------------------------------------------------------------------------------

inline uint32_t encode(const uint8_t * data_in, int numelements,
	uint8_t* out_buffer, uint32_t out_buffer_size) {
	int numZeros = 0;
	int lastVal = 0;

	bitstream_t bs;
	bs_init(&bs);
	bs_attach(&bs, out_buffer, out_buffer_size);

	// Loop over pixels.
	while (numelements > 0) {

		int curVal = *(data_in++);
		int delta = curVal - lastVal;

		if (delta == 0) {
			numZeros++;
		} else {
			if (numZeros > 0) {
				// MUST BURN ZEROS!
				while (numZeros > 0) {
					if (numZeros <= 4) {
						// Ternary is fastest way of deciding how many zeros to encode (2 * numZeros)
						bs_put(&bs, 0x0000, numZeros == 1 ? 2 : numZeros == 2 ? 4 : numZeros == 3 ? 6 : 8);
						numZeros = 0;
					} else {
						bs_put(&bs, 0x2, 3); // 010bbbbb

											 // We never encode less than 5 because in that case
											 //  we'll just use multiple 2-bit single zeros.
						unsigned int numberToEncode = numZeros - continue_threshold;

						// We're only using 5 bits, so we can't encode greater than 31.
						int bar = 2;
						for (int i = 0; i < continue_threshold - 1; i++)
							bar *= 2;
						bar -= 1;
						if (numberToEncode > bar) numberToEncode = bar;

						bs_put(&bs, numberToEncode, continue_threshold); // 0b 010

						numZeros -= (numberToEncode + continue_threshold);
					}
				}

				// numZeros is now zero.
			}

			if (delta == 1 || delta == -1) {
				bs_put(&bs, delta == 1 ? 0x3 : 0x2, 2); // 0b 11
			} else if (delta >= -2 && delta <= 2) {
				bs_put(&bs, delta == 2 ? 0xD : 0xC, 5);
			} else // Reset == 1111 bbbbbbbbbbb
			{
				bs_put(&bs, 0x7, 4); // 0111
				bs_put(&bs, curVal, 8);
			}

		} // end else block of if (delta == 0)

		lastVal = curVal;

		numelements--;
	}

	// FINISH Up -- repeat zeros check.

	if (numZeros > 0) {
		// MUST BURN ZEROS!
		while (numZeros > 0) {
			if (numZeros <= 4) {
				// Ternary is fastest way of deciding how many zeros to encode (2 * numZeros)
				bs_put(&bs, 0x0000, numZeros == 1 ? 2 : numZeros == 2 ? 4 : numZeros == 3 ? 6 : 8);
				numZeros = 0;
			} else {
				bs_put(&bs, 0x2, 3); // 010bbbbb

									 // We never encode less than 5 because in that case
									 //  we'll just use multiple 2-bit single zeros.
				unsigned int numberToEncode = numZeros - continue_threshold;

				// We're only using 5 bits, so we can't encode greater than 31.
				int bar = 2;
				for (int i = 0; i < continue_threshold - 1; i++)
					bar *= 2;
				bar -= 1;
				if (numberToEncode > bar) numberToEncode = bar;

				bs_put(&bs, numberToEncode, continue_threshold); // 0b 010

				numZeros -= (numberToEncode + continue_threshold);
			}
		}
	}
	// END FINISH UP


	bs_flush(&bs);
	return bs_bytes_used(&bs);

}
