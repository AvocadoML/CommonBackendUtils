/*
 * backend_defs.h
 *
 *  Created on: Dec 5, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_BACKEND_DEFS_H_
#define AVOCADO_BACKEND_BACKEND_DEFS_H_

namespace avocado
{
	namespace backend
	{
#ifdef _WIN32
#  ifdef BUILDING_DLL
#    define DLL_PUBLIC __declspec(dllexport)
#  else
#    define DLL_PUBLIC __declspec(dllimport)
#  endif
#else
#  define DLL_PUBLIC
#endif

#ifdef __cplusplus
		extern "C"
		{
#endif

			typedef unsigned char av_uint8;
			typedef char av_int8;
			typedef unsigned short av_uint16;
			typedef short av_int16;
			typedef unsigned int av_uint32;
			typedef int av_int32;
			typedef unsigned long long int av_uint64;
			typedef long long int av_int64;
			typedef struct
			{
					av_uint16 data;
			} bfloat16;
			typedef struct
			{
					av_uint16 data;
			} float16;
			typedef float av_float32;
			typedef double av_float64;
			typedef struct
			{
					av_float32 re, im;
			} av_complex32;
			typedef struct
			{
					av_float64 re, im;
			} av_complex64;

			const int AVOCADO_MAX_TENSOR_DIMENSIONS = 8;
			const av_int64 AVOCADO_INVALID_DESCRIPTOR = -1;
			const av_int64 AVOCADO_NULL_DESCRIPTOR = 0;

			typedef enum
			{
				AVOCADO_DEVICE_CPU,
				AVOCADO_DEVICE_CUDA,
				AVOCADO_DEVICE_OPENCL
			} avDeviceType_t;
			typedef int avDeviceIndex_t;
			const avDeviceIndex_t AVOCADO_INVALID_DEVICE_INDEX = -1;

			/**
			 * Enumeration type used to query device properties.
			 */
			typedef enum
			{
				AVOCADO_DEVICE_NAME, /**< char[256] - Name of the device */
				AVOCADO_DEVICE_PROCESSOR_COUNT, /**< int32 - Number of processors in the device (logical cores of CPU devices or streaming multiprocessors of CUDA devices) */
				AVOCADO_DEVICE_MEMORY, /**< int64 - Number of bytes of memory (RAM memory of CPU devices, or global memory of CUDA devices) */
				AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, /**< bool - Whether the device supports half floats */
				AVOCADO_DEVICE_SUPPORTS_BFLOAT16, /**< bool - Whether the device supports bfloat16 format */
				AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION, /**< bool - Whether the device supports single precision floating point numbers */
				AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, /**< bool - Whether the device supports double precision floating point numbers */
				AVOCADO_DEVICE_SUPPORTS_SSE, /**< bool - Whether the device supports SSE instruction set */
				AVOCADO_DEVICE_SUPPORTS_SSE2, /**< bool - Whether the device supports SSE2 instruction set */
				AVOCADO_DEVICE_SUPPORTS_SSE3, /**< bool - Whether the device supports SSE3 instruction set */
				AVOCADO_DEVICE_SUPPORTS_SSSE3, /**< bool - Whether the device supports SSSE3 instruction set */
				AVOCADO_DEVICE_SUPPORTS_SSE41, /**< bool - Whether the device supports SSE4.1 instruction set */
				AVOCADO_DEVICE_SUPPORTS_SSE42, /**< bool - Whether the device supports SSE4.2 instruction set */
				AVOCADO_DEVICE_SUPPORTS_AVX, /**< bool - Whether the device supports AVX instruction set */
				AVOCADO_DEVICE_SUPPORTS_AVX2, /**< bool - Whether the device supports AVX2 instruction set */
				AVOCADO_DEVICE_SUPPORTS_AVX512_F, /**< bool - Whether the device supports AVX512F instruction set */
				AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ, /**< bool - Whether the device supports AVX512VL, AVX512BW and AVX512DQ instruction sets */
				AVOCADO_DEVICE_SUPPORTS_DP4A, /**< bool - Whether the device supports dp4a instruction (only for CUDA devices) */
				AVOCADO_DEVICE_ARCH_MAJOR, /**< int32 - Major version number of the CUDA or OpenCl architecture */
				AVOCADO_DEVICE_ARCH_MINOR, /**< int32 - Minor version number of the CUDA or OpenCl architecture */
				AVOCADO_DEVICE_SUPPORTS_TENSOR_CORES, /**< bool - Whether the device supports tensor core operations (only for CUDA devices) */
			} avDeviceProperty_t;

			/**
			 *  Enumeration type used for function status returns, which can be one of the following values.
			 */
			typedef enum
			{
				AVOCADO_STATUS_SUCCESS, /**< The operation was completed successfully. */
				AVOCADO_STATUS_ALLOC_FAILED, /**< Resource allocation failed inside the library. */
				AVOCADO_STATUS_FREE_FAILED, /**< Resource deallocation failed inside the library. This is an irrecoverable error. */
				AVOCADO_STATUS_BAD_PARAM, /**< An incorrect value or parameter was passed to the function. */
				AVOCADO_STATUS_ARCH_MISMATCH, /**< The function requires a feature that is not supported on a device. */
				AVOCADO_STATUS_INTERNAL_ERROR, /**< Some internal operation failed. */
				AVOCADO_STATUS_NOT_SUPPORTED, /**< The functionality requested is not presently supported. */
				AVOCADO_STATUS_UNSUPPORTED_DATATYPE, /**< The data type is not presently supported. */
				AVOCADO_STATUS_EXECUTION_FAILED, /**< The function failed to execute.*/
				AVOCADO_STATUS_INSUFFICIENT_DRIVER, /**< The function failed to execute.*/
				AVOCADO_STATUS_DEVICE_TYPE_MISMATCH /**< Passed descriptors are allocated on different devices */
			} avStatus_t;

			/**
			 * Enumeration type indicating the data type tensors and scalars use.
			 */
			typedef enum
			{
				AVOCADO_DTYPE_UNKNOWN, /**< */
				AVOCADO_DTYPE_UINT8, /**< The data is an 8-bit unsigned integer. */
				AVOCADO_DTYPE_INT8, /**< The data is an 8-bit signed integer. */
				AVOCADO_DTYPE_INT16, /**< The data is an 16-bit signed integer. */
				AVOCADO_DTYPE_INT32, /**< The data is an 32-bit signed integer. */
				AVOCADO_DTYPE_INT64, /**< The data is an 64-bit signed integer. */
				AVOCADO_DTYPE_FLOAT16, /**< The data is a 16-bit floating-point. */
				AVOCADO_DTYPE_BFLOAT16, /**< The data is a 16-bit quantity, with 7 mantissa bits, 8 exponent bits, and 1 sign bit. */
				AVOCADO_DTYPE_FLOAT32, /**< The data is a 32-bit single-precision floating-point (float). */
				AVOCADO_DTYPE_FLOAT64, /**< The data is a 64-bit double-precision floating-point (double). */
				AVOCADO_DTYPE_COMPLEX32, /**< The data is a complex number 32-bit single-precision floating-point. */
				AVOCADO_DTYPE_COMPLEX64 /**< The data is a complex number 64-bit double-precision floating-point. */
			} avDataType_t;

			/**
			 * Enumeration type used to specify which activation function will be used.
			 */
			typedef enum
			{
				AVOCADO_ACTIVATION_LINEAR, /**< Selects identity function. */
				AVOCADO_ACTIVATION_SIGMOID, /**< Selects the sigmoid function. */
				AVOCADO_ACTIVATION_TANH, /**< Selects the hyperbolic tangent function. */
				AVOCADO_ACTIVATION_RELU, /**< Selects the clipped rectified linear function. */
				AVOCADO_ACTIVATION_SELU, /**< Selects the scaled exponential linear function. */
				AVOCADO_ACTIVATION_ELU, /**< Selects the exponential linear function. */
				AVOCADO_ACTIVATION_EXPONENTIAL, /**< Selects the exponential function. */
				AVOCADO_ACTIVATION_SOFTPLUS, /**< Selects the softplus function. */
				AVOCADO_ACTIVATION_SOFTSIGN, /**< Selects the softsign function. */
				AVOCADO_ACTIVATION_SOFTMAX /**< Selects the softmax function. */
			} avActivationType_t;

			/**
			 * Enumeration type used to indicate the operation to be used by the ReduceTensor() routine.
			 */
			typedef enum
			{
				AVOCADO_REDUCE_ADD, /**< The operation to be performed is addition. */
				AVOCADO_REDUCE_MUL, /**< The operation to be performed is multiplication. */
				AVOCADO_REDUCE_MIN, /**< The operation to be performed is a minimum comparison. */
				AVOCADO_REDUCE_MAX, /**< The operation to be performed is a maximum comparison. */
				AVOCADO_REDUCE_AMAX, /**< The operation to be performed is a maximum comparison of absolute values. */
				AVOCADO_REDUCE_AVG, /**< The operation to be performed is averaging. */
				AVOCADO_REDUCE_NORM1, /**< The operation to be performed is addition of absolute values. */
				AVOCADO_REDUCE_NORM2, /**< The operation to be performed is a square root of the sum of squares. */
				AVOCADO_REDUCE_MUL_NO_ZEROS, /**< The operation to be performed is multiplication, not including elements of value zero. */
				AVOCADO_REDUCE_LOGICAL_OR, /**< The operation to be performed is logical OR */
				AVOCADO_REDUCE_LOGICAL_AND /**< The operation to be performed is logical AND */
			} avReduceOp_t;

			/**
			 * Enumeration type used to indicate the operation to be used by the BinaryOp() routine.
			 */
			typedef enum
			{
				AVOCADO_BINARY_OP_ADD, /**< The operation to be performed is addition. */
				AVOCADO_BINARY_OP_ADD_SQUARE, /**< The operation to be performed is addition between the first tensor and the square of the second tensor. */
				AVOCADO_BINARY_OP_SUB, /**< The operation to be performed is subtraction. */
				AVOCADO_BINARY_OP_MUL, /**< The operation to be performed is multiplication. */
				AVOCADO_BINARY_OP_DIV, /**< The operation to be performed is division. */
				AVOCADO_BINARY_OP_MOD, /**< The operation to be performed is floating-point remainder of the first tensor's division by the second tensor. */
				AVOCADO_BINARY_OP_POW, /**< The operation to be performed is value from the first tensor to the power of the second tensor. */
				AVOCADO_BINARY_OP_MIN, /**< The operation to be performed is a minimum comparison. */
				AVOCADO_BINARY_OP_MAX, /**< The operation to be performed is a maximum comparison. */
				AVOCADO_BINARY_OP_COMPARE_EQ, /**< The operation to be performed is truth value of the first tensor equal to the second tensor. */
				AVOCADO_BINARY_OP_COMPARE_NEQ, /**< The operation to be performed is truth value of the first tensor not equal to the second tensor. */
				AVOCADO_BINARY_OP_COMPARE_GT, /**< The operation to be performed is truth value of the first tensor greater than to the second tensor. */
				AVOCADO_BINARY_OP_COMPARE_GE, /**< The operation to be performed is truth value of the first tensor greater than equal to the second tensor. */
				AVOCADO_BINARY_OP_COMPARE_LT, /**< The operation to be performed is truth value of the first tensor less than to the second tensor. */
				AVOCADO_BINARY_OP_COMPARE_LE, /**< The operation to be performed is truth value of the first tensor less than equal to the second tensor. */
				AVOCADO_BINARY_OP_LOGICAL_AND, /**< The operation to be performed is truth value of the first tensor logical AND to the second tensor. */
				AVOCADO_BINARY_OP_LOGICAL_OR, /**< The operation to be performed is truth value of the first tensor logical OR to the second tensor. */
				AVOCADO_BINARY_OP_LOGICAL_XOR /**< The operation to be performed is truth value of the first tensor logical XOR to the second tensor. */
			} avBinaryOp_t;

			/**
			 * Enumeration type used to indicate the operation to be used by the UnaryOp() routine.
			 */
			typedef enum
			{
				AVOCADO_UNARY_OP_ABS, /**< The operation to be performed is absolute value. */
				AVOCADO_UNARY_OP_CEIL, /**< The operation to be performed is ceiling value. */
				AVOCADO_UNARY_OP_COS, /**< The operation to be performed is trigonometric cosine. */
				AVOCADO_UNARY_OP_EXP, /**< The operation to be performed is exponential of a tensor. */
				AVOCADO_UNARY_OP_FLOOR, /**< The operation to be performed is floor value. */
				AVOCADO_UNARY_OP_LN, /**< The operation to be performed is natural logarithm. */
				AVOCADO_UNARY_OP_NEG, /**< The operation to be performed is negation. */
				AVOCADO_UNARY_OP_RCP, /**< The operation to be performed is reciprocal value. */
				AVOCADO_UNARY_OP_RSQRT, /**< The operation to be performed is reciprocal of the square root. */
				AVOCADO_UNARY_OP_SIN, /**< The operation to be performed is trigonometric sine. */
				AVOCADO_UNARY_OP_SQUARE, /**< The operation to be performed is squaring. */
				AVOCADO_UNARY_OP_SQRT, /**< The operation to be performed is square root. */
				AVOCADO_UNARY_OP_TAN, /**< The operation to be performed is trigonometric tangent. */
				AVOCADO_UNARY_OP_LOGICAL_NOT /**< The operation to be performed is logical negation. */
			} avUnaryOp_t;

			/**
			 * Enumeration type used to select the pooling method in PoolingForward() and PoolingBackward().
			 */
			typedef enum
			{
				AVOCADO_POOLING_MAX, /**< The maximum value inside the pooling window is used. */
				AVOCADO_POOLING_AVERAGE_INCLUDE_PADDING, /**< Values inside the pooling window are averaged including values from the padding region. */
				AVOCADO_POOLING_AVERAGE_EXCLUDE_PADDING /**< Values inside the pooling window are averaged excluding values from the padding region. */
			} avPoolingMode_t;

			/**
			 * Enumeration type indicating over which data the softmax function is calculated.
			 */
			typedef enum
			{
				AVOCADO_SOFTMAX_MODE_INSTANCE, /**< The softmax operation is computed per image (N) across the dimensions H,W,C. */
				AVOCADO_SOFTMAX_MODE_CHANNEL /**< The softmax operation is computed per image (N) and spatial location (H,W) across dimension C. */
			} avSoftmaxMode_t;

			typedef enum
			{
				AVOCADO_GEMM_OPERATION_N, /**< No operation is performed. */
				AVOCADO_GEMM_OPERATION_T, /**< The matrix is transposed. */
				AVOCADO_GEMM_OPERATION_C /**<  */
			} avGemmOperation_t;

			typedef enum
			{
				AVOCADO_ACCURACY_METRIC /**<  */
			} avMetricType_t;

			typedef enum
			{
				AVOCADO_MEAN_SQUARE_LOSS, /**<  */
				AVOCADO_CROSS_ENTROPY_LOSS, /**<  */
				AVOCADO_KL_DIVERGENCE_LOSS /**<  */
			} avLossType_t;

			typedef enum
			{
				AVOCADO_OPTIMIZER_SGD, /**< */
				AVOCADO_OPTIMIZER_ADAM /**< */
			} avOptimizerType_t;

			typedef enum
			{
				AVOCADO_CONVOLUTION_MODE, /**<  */
				AVOCADO_CROSS_CORRELATION_MODE /**<  */
			} avConvolutionMode_t;

			typedef enum
			{
				AVOCADO_CONVOLUTION_ALGORITHM_AUTO, /**<  */
				AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM, /**<  */
				AVOCADO_CONVOLUTION_ALGORITHM_IMPLICIT_GEMM, /**<  */
				AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED, /**<  */
				AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_FUSED /**<  */
			} avConvolutionAlgorithm_t;

			/* Opaque type for backend memory block descriptor */
			typedef av_int64 avMemoryDescriptor_t;

			/* Opaque type for backend context descriptor */
			typedef av_int64 avContextDescriptor_t;

			/* Opaque type for backend tensor descriptor */
			typedef av_int64 avTensorDescriptor_t;

			/* Opaque type for backend convolution descriptor */
			typedef av_int64 avConvolutionDescriptor_t;

			/* Opaque type for backend pooling descriptor */
			typedef av_int64 avPoolingDescriptor_t;

			/* Opaque type for backend optimizer descriptor */
			typedef av_int64 avOptimizerDescriptor_t;

			/* Opaque type for backend dropout descriptor */
			typedef av_int64 avDropoutDescriptor_t;

#ifdef __cplusplus
		}
#endif
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_BACKEND_DEFS_H_ */
