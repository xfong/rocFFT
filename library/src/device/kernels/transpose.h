// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "array_format.h"
#include "common.h"

#define MAX_LAUNCH_BOUNDS_TRANSPOSE_KERNEL 1024

#define TRANSPOSE_TWIDDLE_MUL()                                                                   \
    if(WITH_TWL)                                                                                  \
    {                                                                                             \
        if(TWL == 2)                                                                              \
        {                                                                                         \
            if(DIR == -1)                                                                         \
            {                                                                                     \
                TWIDDLE_STEP_MUL_FWD(TWLstep2, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                TWIDDLE_STEP_MUL_INV(TWLstep2, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
        }                                                                                         \
        else if(TWL == 3)                                                                         \
        {                                                                                         \
            if(DIR == -1)                                                                         \
            {                                                                                     \
                TWIDDLE_STEP_MUL_FWD(TWLstep3, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                TWIDDLE_STEP_MUL_INV(TWLstep3, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
        }                                                                                         \
        else if(TWL == 4)                                                                         \
        {                                                                                         \
            if(DIR == -1)                                                                         \
            {                                                                                     \
                TWIDDLE_STEP_MUL_FWD(TWLstep4, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                TWIDDLE_STEP_MUL_INV(TWLstep4, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    shared[tx1][ty1 + i] = tmp; // the transpose taking place here

// - transpose input of size m * n (up to DIM_X * DIM_X) to output of size n * m
//   input, output are in device memory
//   shared memory of size DIM_X*DIM_X is allocated size_ternally as working space
// - Assume DIM_X by DIM_Y threads are reading & wrting a tile size DIM_X * DIM_X
//   DIM_X is divisible by DIM_Y
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   WITH_TWL,
          int    TWL,
          int    DIR,
          bool   ALL,
          bool   UNIT_STRIDE_0>
DEVICE_MARKER inline void transpose_tile_device(in_acc_t<T_I, 1> &input,
                                      gen_acc_t<T_O, 1>          &output,
                                      size_t                     in_offset,
                                      size_t                     out_offset,
                                      const size_t               m,
                                      const size_t               n,
                                      size_t                     gx,
                                      size_t                     gy,
                                      size_t                     ld_in,
                                      size_t                     ld_out,
                                      size_t                     stride_0_in,
                                      size_t                     stride_0_out,
                                      gen_acc_t<T, 1>            &twiddles_large,
									  local_acc_t<T, 2>          &shared,
									  cl::sycl::nd_item<3>       &item_id)
{
    size_t tid = item_id.get_local_id(0) + item_id.get_local_id(1) * item_id.get_local_range(0);
    size_t tx1 = tid % DIM_X;
    size_t ty1 = tid / DIM_X;

    if(ALL)
    {
#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            T tmp;
            if(UNIT_STRIDE_0)
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
            }
            else
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
            }
            TRANSPOSE_TWIDDLE_MUL();
        }

        item_id.barrier();

#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            // reconfigure the threads
            if(UNIT_STRIDE_0)
            {
                Handler<T_O>::write(
                    output, out_offset + tx1 + (i + ty1) * ld_out, shared[ty1 + i][tx1]);
            }
            else
            {
                Handler<T_O>::write(output,
                                    out_offset + tx1 * stride_0_out + (i + ty1) * ld_out,
                                    shared[ty1 + i][tx1]);
            }
        }
    }
    else
    {
        for(size_t i = 0; i < m; i += DIM_Y)
        {
            if(tx1 < n && (ty1 + i) < m)
            {
                T tmp;
                if(UNIT_STRIDE_0)
                {
                    tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
                }
                else
                {
                    tmp = Handler<T_I>::read(input,
                                             in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
                }
                TRANSPOSE_TWIDDLE_MUL();
            }
        }

        item_id.barrier();

        for(size_t i = 0; i < n; i += DIM_Y)
        {
            // reconfigure the threads
            if(tx1 < m && (ty1 + i) < n)
            {
                if(UNIT_STRIDE_0)
                {
                    Handler<T_O>::write(
                        output, out_offset + tx1 + (i + ty1) * ld_out, shared[ty1 + i][tx1]);
                }
                else
                {
                    Handler<T_O>::write(output,
                                        out_offset + tx1 * stride_0_out + (i + ty1) * ld_out,
                                        shared[ty1 + i][tx1]);
                }
            }
        }
    }
}

// - transpose input of size m * n to output of size n * m
//   input, output are in device memory
// - 2D grid and 2D thread block (DIM_X, DIM_Y)
// - Assume DIM_X by DIM_Y threads are transposing a tile DIM_X * DIM_X
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   WITH_TWL,
          int    TWL,
          int    DIR,
          bool   ALL,
          bool   UNIT_STRIDE_0,
          bool   DIAGONAL>
GLOBAL_MARKER void
    transpose_kernel2(in_acc_t<T_I, 1>     &input,
                      gen_acc_t<T_O, 1>    &output,
                      gen_acc_t<T, 1>      &twiddles_large,
                      len_acc              lengths,
                      len_acc              stride_in,
                      len_acc              stride_out,
					  local_acc_t<T, 2>    &shared,
					  cl::sycl::nd_item<3> &item_id)
{
    size_t ld_in  = stride_in[1];
    size_t ld_out = stride_out[1];

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = item_id.get_group(2);

    iOffset += counter_mod * stride_in[2];
    oOffset += counter_mod * stride_out[2];

    size_t tileBlockIdx_x, tileBlockIdx_y;
    if(DIAGONAL) // diagonal reordering
    {
        //TODO: template and simplify index calc for square case if necessary
        size_t bid     = item_id.get_group(0) + item_id.get_num_group(0) * item_id.get_group(1);
        tileBlockIdx_y = bid % item_id.get_num_group(1);
        tileBlockIdx_x = ((bid / item_id.get_num_group(1)) + tileBlockIdx_y) % item_id.get_num_group(0);
    }
    else
    {
        tileBlockIdx_x = item_id.get_group(0);
        tileBlockIdx_y = item_id.get_group(1);
    }

    iOffset += tileBlockIdx_x * DIM_X * stride_in[0] + tileBlockIdx_y * DIM_X * ld_in;
    oOffset += tileBlockIdx_x * DIM_X * ld_out + tileBlockIdx_y * DIM_X * stride_out[0];

    if(ALL)
    {
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, WITH_TWL, TWL, DIR, ALL, UNIT_STRIDE_0>(
            input,
            output,
            iOffset,
            oOffset,
            DIM_X,
            DIM_X,
            tileBlockIdx_x * DIM_X,
            tileBlockIdx_y * DIM_X,
            ld_in,
            ld_out,
            stride_in,
            stride_out,
            twiddles_large,
			shared,
			item_id);
    }
    else
    {
        size_t m  = lengths[1];
        size_t n  = lengths[0];
        size_t mm = min(m - tileBlockIdx_y * DIM_X, DIM_X); // the corner case along m
        size_t nn = min(n - tileBlockIdx_x * DIM_X, DIM_X); // the corner case along n
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, WITH_TWL, TWL, DIR, ALL, UNIT_STRIDE_0>(
            input,
            output,
            iOffset,
            oOffset,
            mm,
            nn,
            tileBlockIdx_x * DIM_X,
            tileBlockIdx_y * DIM_X,
            ld_in,
            ld_out,
            stride_in,
            stride_out,
            twiddles_large,
			shared,
			item_id);
    }
}

template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   WITH_TWL,
          int    TWL,
          int    DIR,
          bool   ALL,
          bool   UNIT_STRIDE_0,
          bool   DIAGONAL>
class transpose_kernel2_t {
	private:
		const in_acc_t<T_I, 1> input_acc;
		gen_acc_t<T_O, 1>      output_acc;
		gen_acc_t<T, 1>        twiddles_large_acc;
		len_acc                lengths_arr;
		len_acc                stride_in_arr;
		len_acc                stride_out_arr;
		local_acc_t<T, 2>      shared_acc;
		
	public:
		transpose_kernel2_t(const in_acc_t<T_I, 1> in_acc_,
							gen_acc_t<T_O, 1>      out_acc_,
							gen_acc_t<T, 1>        twiddles_large_acc_,
							len_acc                lengths_arr_,
							len_acc                stride_in_arr_,
							len_acc                stride_out_arr_,
							local_acc_t<T, 2>      shared_acc_)
						: input_acc(in_acc_),
						  output_acc(out_acc_),
						  twiddles_large_acc(twiddles_large_acc_),
						  lengths_arr(lengths_arr_),
						  stride_in_arr(stride_in_arr_),
						  stride_out_arr(stride_out_arr_),
						  shared_acc(shared_acc_) {}

		void operator()(cl::sycl::nd_item<3> item_id) {
			transpose_kernel2<T, T_I, T_O, DIM_X, DIM_Y, WITH_TWL, TWL, DIR, ALL, UNIT_STRIDE_0, DIAGONAL>(input_acc,
																										   output_acc,
																										   twiddles_large_acc,
																										   lengths_arr,
																										   stride_in_arr,
																										   stride_out_arr,
																										   shared_acc,
																										   item_id);
		}
};

// tiled transpose device function for transpose_scheme
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   ALL,
          bool   UNIT_STRIDE_0>
DEVICE_MARKER void transpose_tile_device_scheme(in_acc_t<T_I, 1>  &input,
                                             gen_acc_t<T_O, 1>    &output,
                                             size_t               in_offset,
                                             size_t               out_offset,
                                             const size_t         m,
                                             const size_t         n,
                                             size_t               ld_in,
                                             size_t               ld_out,
                                             size_t               stride_0_in,
                                             size_t               stride_0_out,
											 local_acc_t<T, 2>    &shared,
											 cl::sycl::nd_item<3> &item_id)
{
    size_t tid = item_id.get_local_id(0) + item_id.get_local_id(1) * item_id.get_local_range(0);
    size_t tx1 = tid % DIM_X;
    size_t ty1 = tid / DIM_X;

    if(ALL)
    {
#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            T tmp;
            if(UNIT_STRIDE_0)
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
            }
            else
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
            }
            shared[tx1][ty1 + i] = tmp; // the transpose taking place here
        }

        item_id.barrier();

#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            // reconfigure the threads
            if(UNIT_STRIDE_0)
            {
                Handler<T_O>::write(
                    output, out_offset + tx1 + (i + ty1) * ld_out, shared[ty1 + i][tx1]);
            }
            else
            {
                Handler<T_O>::write(output,
                                    out_offset + tx1 * stride_0_out + (i + ty1) * ld_out,
                                    shared[ty1 + i][tx1]);
            }
        }
    }
    else
    {
        for(size_t i = 0; i < m; i += DIM_Y)
        {
            if(tx1 < n && (ty1 + i) < m)
            {
                T tmp;
                if(UNIT_STRIDE_0)
                {
                    tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
                }
                else
                {
                    tmp = Handler<T_I>::read(input,
                                             in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
                }
                shared[tx1][ty1 + i] = tmp; // the transpose taking place here
            }
        }

        item_id.barrier();

        for(size_t i = 0; i < n; i += DIM_Y)
        {
            // reconfigure the threads
            if(tx1 < m && (ty1 + i) < n)
            {
                if(UNIT_STRIDE_0)
                {
                    Handler<T_O>::write(
                        output, out_offset + tx1 + (i + ty1) * ld_out, shared[ty1 + i][tx1]);
                }
                else
                {
                    Handler<T_O>::write(output,
                                        out_offset + tx1 * stride_0_out + (i + ty1) * ld_out,
                                        shared[ty1 + i][tx1]);
                }
            }
        }
    }
}

// global function for transpose scheme
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   ALL,
          bool   UNIT_STRIDE_0,
          bool   DIAGONAL>
GLOBAL_MARKER void
    transpose_kernel2_scheme(in_acc_t<T_I, 1>     &input,
                             gen_acc_t<T_O, 1>    &output,
                             gen_acc_t<T, 1>      &twiddles_large,
                             len_acc              lengths,
                             len_acc              stride_in,
                             len_acc              stride_out,
                             size_t               ld_in,
                             size_t               ld_out,
                             size_t               m,
                             size_t               n,
							 local_acc_t<T, 2>    &shared,
							 cl::sycl::nd_item<3> &item_id)
{
    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = item_id.get_group(2);

    iOffset += counter_mod * stride_in[3];
    oOffset += counter_mod * stride_out[3];

    size_t tileBlockIdx_x, tileBlockIdx_y;
    if(DIAGONAL) // diagonal reordering
    {
        //TODO: template and simplify index calc for square case if necessary
        size_t bid     = item_id.get_group(0) + item_id.get_num_group(0) * item_id.get_group(1);
        tileBlockIdx_y = bid % item_id.get_num_group(1);
        tileBlockIdx_x = ((bid / item_id.get_num_group(1)) + tileBlockIdx_y) % item_id.get_num_group(0);
    }
    else
    {
        tileBlockIdx_x = item_id.get_group(0);
        tileBlockIdx_y = item_id.get_group(1);
    }

    iOffset += tileBlockIdx_x * DIM_X * stride_in[0] + tileBlockIdx_y * DIM_X * ld_in;
    oOffset += tileBlockIdx_x * DIM_X * ld_out + tileBlockIdx_y * DIM_X * stride_out[0];

    if(ALL)
    {
        transpose_tile_device_scheme<T, T_I, T_O, DIM_X, DIM_Y, ALL, UNIT_STRIDE_0>(input,
                                                                                    output,
                                                                                    iOffset,
                                                                                    oOffset,
                                                                                    DIM_X,
                                                                                    DIM_X,
                                                                                    ld_in,
                                                                                    ld_out,
                                                                                    stride_in,
                                                                                    stride_out,
																					shared,
																					item_id);
    }
    else
    {
        size_t mm = min(m - tileBlockIdx_y * DIM_X, DIM_X); // the partial case along m
        size_t nn = min(n - tileBlockIdx_x * DIM_X, DIM_X); // the partial case along n
        transpose_tile_device_scheme<T, T_I, T_O, DIM_X, DIM_Y, ALL, UNIT_STRIDE_0>(input,
                                                                                    output,
																					iOffset,
																					oOffset,
																					mm,
																					nn,
																					ld_in,
																					ld_out,
																					stride_in,
																					stride_out,
																					shared,
																					item_id);
    }
}

template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   ALL,
          bool   UNIT_STRIDE_0,
          bool   DIAGONAL>
class transpose_kernel2_scheme_t{
	private:
		in_acc_t<T_I, 1>     input_acc;
        gen_acc_t<T_O, 1>    output_acc;
        gen_acc_t<T, 1>      twiddles_large_acc;
        len_acc              lengths_acc,
        len_acc              stride_in_acc,
        len_acc              stride_out_acc,
        size_t               ld_in,
        size_t               ld_out,
        size_t               m,
        size_t               n,
		local_acc_t<T, 2>    shared_acc,

	public:
		transpose_kernel2_scheme_t(in_acc_t<T_I, 1> input_acc_,
								gen_acc_t<T_O, 1>   output_acc_,
								gen_acc_t<T, 1>     twiddles_large_acc_,
								len_acc             lengths_acc_,
								len_acc             stride_in_acc_,
								len_acc             stride_out_acc_,
								size_t              ld_in_,
								size_t              ld_out_,
								size_t              m_,
								size_t              n_,
								local_acc_t<T, 2>   shared_acc_)
				: input_acc(input_acc_),
				  output_acc(output_acc_),
				  twiddles_large_acc(twiddles_large_acc_),
				  lengths_acc(lengths_acc_),
				  stride_in_acc(stride_in_acc_),
				  stride_out_acc(stride_out_acc_),
				  ld_in(ld_in_),
				  ld_out(ld_out_),
				  m(m_),
				  n(n_),
				  shared_acc(shared_acc_) {}
		void operator()(cl::sycl::nd_item<3> item_id) {
			transpose_kernel2_scheme<T, T_I, T_O, DIM_X, DIM_Y, ALL, UNIT_STRIDE_0, DIAGONAL>(input_acc,
																							  output_acc,
																							  twiddles_large_acc,
																							  lengths_acc,
																							  stride_in_acc,
																							  stride_out_acc,
																							  ld_in,
																							  ld_out,
																							  m,
																							  n,
																							  shared_acc,
																							  item_id);
		}
};
#endif // TRANSPOSE_H
