/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef COMMON_H
#define COMMON_H
#include "CL/sycl.hpp"
#include "rocfft.h"
#include <iostream>
#include <string>

// Convenience aliases
#define DEVICE_MARKER inline

#define GLOBAL_MARKER inline

#define SYCL_KERNEL_NAME(...) __VA_ARGS__

template<typename T, int dim>
using local_acc_t = cl::sycl::accessor<T, dim, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;
template<typename T, int dim>
using gen_acc_t = cl::sycl::accessor<T, dim, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>;
template<typename T, int dim>
using in_acc_t = cl::sycl::accessor<T, dim, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>;
using len_acc = cl::sycl::accessor<size_t, 1, cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>;

// NB:
//   All kernels were compiled based on the assumption that the default max
//   work group size is 256. This default value in compiler might change in
//   future. Each kernel has to explicitly set proper sizes through
//   __launch_bounds__ or __attribute__.
//   Further performance tuning might be done later.
#define MAX_LAUNCH_BOUNDS_2D_SINGLE_KERNEL 256
#define MAX_LAUNCH_BOUNDS_R2C_C2R_KERNEL 256

// To extract file name out of full path
static inline constexpr const char* KernelFileName(const char* fullname)
{
    const char* f = fullname;
    while(*fullname)
    {
        if(*fullname++ == '/') // TODO: check it for WIN
        {
            f = fullname;
        }
    }
    return f;
}

// To generate back reference line number in generated kernel code at the end
// of the line.
//     usage: str += GEN_REF_LINE()
#define GEN_REF_LINE()               \
    " // ";                          \
    const char* fullname = __FILE__; \
    str += KernelFileName(fullname); \
    str += ":";                      \
    str += std::to_string(__LINE__); \
    str += "\n";

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

template <class T>
struct real_type;

template <>
struct real_type<float4>
{
    typedef float type;
};

template <>
struct real_type<double4>
{
    typedef double type;
};

template <>
struct real_type<float2>
{
    typedef float type;
};

template <>
struct real_type<double2>
{
    typedef double type;
};

template <class T>
using real_type_t = typename real_type<T>::type;

/* example of using real_type_t */
// real_type_t<float2> float_scalar;
// real_type_t<double2> double_scalar;

template <class T>
struct complex_type;

template <>
struct complex_type<float>
{
    typedef float2 type;
};

template <>
struct complex_type<double>
{
    typedef double2 type;
};

template <class T>
using complex_type_t = typename complex_type<T>::type;

/// example of using complex_type_t:
// complex_type_t<float> float_complex_val;
// complex_type_t<double> double_complex_val;

template <class T>
struct vector4_type;

template <>
struct vector4_type<float2>
{
    typedef float4 type;
};

template <>
struct vector4_type<double2>
{
    typedef double4 type;
};

template <class T>
using vector4_type_t = typename vector4_type<T>::type;

/* example of using vector4_type_t */
// vector4_type_t<float2> float4_scalar;
// vector4_type_t<double2> double4_scalar;

template <rocfft_precision T>
struct vector2_type;

template <>
struct vector2_type<rocfft_precision_single>
{
    typedef float2 type;
};

template <>
struct vector2_type<rocfft_precision_double>
{
    typedef double2 type;
};

template <rocfft_precision T>
using vector2_type_t = typename vector2_type<T>::type;

/* example of using vector2_type_t */
// vector2_type_t<rocfft_precision_single> float2_scalar;
// vector2_type_t<rocfft_precision_double> double2_scalar;

template <typename T>
inline T lib_make_vector2(real_type_t<T> v0, real_type_t<T> v1);

template <>
inline float2 lib_make_vector2(float v0, float v1)
{
    return float2(v0, v1);
}

template <>
inline double2 lib_make_vector2(double v0, double v1)
{
    return double2(v0, v1);
}

template <typename T>
inline T
    lib_make_vector4(real_type_t<T> v0, real_type_t<T> v1, real_type_t<T> v2, real_type_t<T> v3);

template <>
inline float4 lib_make_vector4(float v0, float v1, float v2, float v3)
{
    return float4(v0, v1, v2, v3);
}

template <>
inline double4 lib_make_vector4(double v0, double v1, double v2, double v3)
{
    return double4(v0, v1, v2, v3);
}

template <typename T>
DEVICE_MARKER T TWLstep1(gen_acc_t<T, 1> &twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    return result;
}

template <typename T>
DEVICE_MARKER T TWLstep2(gen_acc_t<T, 1> &twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    return result;
}

template <typename T>
DEVICE_MARKER T TWLstep3(gen_acc_t<T, 1> &twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    return result;
}

template <typename T>
DEVICE_MARKER T TWLstep4(gen_acc_t<T, 1> &twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[768 + j].x - result.y * twiddles[768 + j].y),
                                 (result.y * twiddles[768 + j].x + result.x * twiddles[768 + j].y));
    return result;
}

#define TWIDDLE_STEP_MUL_FWD(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T              W = TWFUNC(TWIDDLES, INDEX);        \
        real_type_t<T> TR, TI;                             \
        TR    = (W.x * REG.x) - (W.y * REG.y);             \
        TI    = (W.y * REG.x) + (W.x * REG.y);             \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#define TWIDDLE_STEP_MUL_INV(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T              W = TWFUNC(TWIDDLES, INDEX);        \
        real_type_t<T> TR, TI;                             \
        TR    = (W.x * REG.x) + (W.y * REG.y);             \
        TI    = -(W.y * REG.x) + (W.x * REG.y);            \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#endif // COMMON_H
