/*
 * backend_utils.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/backend_utils.hpp>
#include <Avocado/backend_descriptors.hpp>

#include <algorithm>

#if defined(CPU_BACKEND)
#elif defined(CUDA_BACKEND)
#  include <cuda_runtime_api.h>
#  include <cuda_fp16.h>
#  include <cublas_v2.h>
#elif defined(OPENCL_BACKEND)
#  include <CL/cl2.hpp>
#else
#endif

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{
			std::string descriptorTypeToString(av_int64 desc)
			{
				switch (getDescriptorType(desc))
				{
					case MemoryDescriptor::descriptor_type:
						return "MemoryDescriptor";
					case ContextDescriptor::descriptor_type:
						return "ContextDescriptor";
					case TensorDescriptor::descriptor_type:
						return "TensorDescriptor";
					case ConvolutionDescriptor::descriptor_type:
						return "ConvolutionDescriptor";
					case PoolingDescriptor::descriptor_type:
						return "PoolingDescriptor";
					case OptimizerDescriptor::descriptor_type:
						return "OptimizerDescriptor";
					case DropoutDescriptor::descriptor_type:
						return "DropoutDescriptor";
					default:
						return "incorrect type (" + std::to_string(getDescriptorType(desc)) + ")";
				}
			}
			std::string descriptorToString(av_int64 desc)
			{
				std::string result = std::to_string(desc) + " : ";
				if (desc < 0)
					return result + "invalid (negative)";
				result += descriptorTypeToString(desc) + " on ";
				switch (getDeviceType(desc))
				{
					case AVOCADO_DEVICE_CPU:
						result += "CPU";
						break;
					case AVOCADO_DEVICE_CUDA:
						result += "CUDA:" + std::to_string(getCurrentDeviceIndex());
						break;
					case AVOCADO_DEVICE_OPENCL:
						result += "OPENCL:" + std::to_string(getCurrentDeviceIndex());
						break;
					default:
						return result + " unknown device type";
				}
				return result + ", ID = " + std::to_string(getDescriptorIndex(desc));
			}
			std::string statusToString(avStatus_t status)
			{
				switch (status)
				{
					case AVOCADO_STATUS_SUCCESS:
						return "SUCCESS";
					case AVOCADO_STATUS_ALLOC_FAILED:
						return "ALLOC_FAILED";
					case AVOCADO_STATUS_FREE_FAILED:
						return "FREE_FAILED";
					case AVOCADO_STATUS_BAD_PARAM:
						return "BAD_PARAM";
					case AVOCADO_STATUS_ARCH_MISMATCH:
						return "ARCH_MISMATCH";
					case AVOCADO_STATUS_INTERNAL_ERROR:
						return "INTERNAL_ERROR";
					case AVOCADO_STATUS_NOT_SUPPORTED:
						return "NOT_SUPPORTED";
					case AVOCADO_STATUS_UNSUPPORTED_DATATYPE:
						return "UNSUPPORTED_DATATYPE";
					case AVOCADO_STATUS_EXECUTION_FAILED:
						return "EXECUTION_FAILED";
					case AVOCADO_STATUS_INSUFFICIENT_DRIVER:
						return "INSUFFICIENT_DRIVER";
					case AVOCADO_STATUS_DEVICE_TYPE_MISMATCH:
						return "DEVICE_TYPE_MISMATCH";
					default:
						return "UNKNOWN_STATUS";
				}
			}
			std::string dtypeToString(avDataType_t dtype)
			{
				switch (dtype)
				{
					default:
					case AVOCADO_DTYPE_UNKNOWN:
						return "UNKNOWN";
					case AVOCADO_DTYPE_UINT8:
						return "UINT8";
					case AVOCADO_DTYPE_INT8:
						return "INT8";
					case AVOCADO_DTYPE_INT16:
						return "INT16";
					case AVOCADO_DTYPE_INT32:
						return "INT32";
					case AVOCADO_DTYPE_INT64:
						return "INT64";
					case AVOCADO_DTYPE_FLOAT16:
						return "FLOAT16";
					case AVOCADO_DTYPE_BFLOAT16:
						return "BFLOAT16";
					case AVOCADO_DTYPE_FLOAT32:
						return "FLOAT32";
					case AVOCADO_DTYPE_FLOAT64:
						return "FLOAT64";
					case AVOCADO_DTYPE_COMPLEX32:
						return "COMPLEX32";
					case AVOCADO_DTYPE_COMPLEX64:
						return "COMPLEX64";
				}
			}
			std::string deviceTypeToString(avDeviceType_t type)
			{
				switch (type)
				{
					case AVOCADO_DEVICE_CPU:
						return "CPU";
					case AVOCADO_DEVICE_CUDA:
						return "CUDA";
					case AVOCADO_DEVICE_OPENCL:
						return "OPENCL";
					default:
						return "reference";
				}
			}

			int getNumberOfDevices() noexcept
			{
#if defined(CUDA_BACKEND)
				static const int result = []()
				{
					int tmp = 0;
					cudaError_t status = cudaGetDeviceCount(&tmp);
					return (status == cudaSuccess) ? tmp : 0;
				}();
#elif defined(OPENCL_BACKEND)
				static const int result = 0;
#else
				static const int result = 1;
#endif
				return result;
			}

			avDeviceType_t getDeviceType(av_int64 descriptor) noexcept
			{
				const av_int64 device_type_mask = 0xFF00000000000000ull;
				return static_cast<avDeviceType_t>((descriptor & device_type_mask) >> 56ull);
			}
			int getDescriptorType(av_int64 descriptor) noexcept
			{
				const av_int64 descriptor_type_mask = 0x00FF000000000000ull;
				return static_cast<int>((descriptor & descriptor_type_mask) >> 48ull);
			}
			avDeviceIndex_t getDeviceIndex(av_int64 descriptor) noexcept
			{
				const av_int64 device_index_mask = 0x0000FFFF00000000ull;
				return static_cast<avDeviceIndex_t>((descriptor & device_index_mask) >> 32ull);
			}
			int getDescriptorIndex(av_int64 descriptor) noexcept
			{
				const av_int64 descriptor_index_mask = 0x00000000FFFFFFFFull;
				return static_cast<int>(descriptor & descriptor_index_mask);
			}

			avDeviceType_t getCurrentDeviceType() noexcept
			{
#if defined(CUDA_BACKEND)
				return AVOCADO_DEVICE_CUDA;
#elif defined(OPENCL_BACKEND)
				return AVOCADO_DEVICE_OPENCL;
#else
				return AVOCADO_DEVICE_CPU;
#endif
			}
			av_int64 getCurrentDeviceIndex() noexcept
			{
#if defined(CUDA_BACKEND)
				int tmp = 0;
				cudaError_t status = cudaGetDevice(&tmp);
				return (status == cudaSuccess) ? tmp : 0;
#elif defined(OPENCL_BACKEND)
				return 0;
#else
				return 0;
#endif
			}

			av_int64 createDescriptor(int index, av_int64 type) noexcept
			{
				return (static_cast<av_int64>(getCurrentDeviceType()) << 56ull) | (type << 48ull) | (static_cast<av_int64>(getCurrentDeviceIndex()) << 32ull)
						| static_cast<av_int64>(index);
			}

			int dataTypeSize(avDataType_t dtype) noexcept
			{
				switch (dtype)
				{
					default:
					case AVOCADO_DTYPE_UNKNOWN:
						return 0;
					case AVOCADO_DTYPE_UINT8:
						return 1;
					case AVOCADO_DTYPE_INT8:
						return 1;
					case AVOCADO_DTYPE_INT16:
						return 2;
					case AVOCADO_DTYPE_INT32:
						return 4;
					case AVOCADO_DTYPE_INT64:
						return 8;
					case AVOCADO_DTYPE_FLOAT16:
						return 2;
					case AVOCADO_DTYPE_BFLOAT16:
						return 2;
					case AVOCADO_DTYPE_FLOAT32:
						return 4;
					case AVOCADO_DTYPE_FLOAT64:
						return 8;
					case AVOCADO_DTYPE_COMPLEX32:
						return 8;
					case AVOCADO_DTYPE_COMPLEX64:
						return 16;
				}
			}

			template<>
			av_complex32 getAlphaValue<av_complex32>(const void *alpha) noexcept
			{
				if (alpha == nullptr)
					return av_complex32 { 1.0f, 0.0f };
				else
					return reinterpret_cast<const av_complex32*>(alpha)[0];
			}
			template<>
			av_complex32 getBetaValue<av_complex32>(const void *beta) noexcept
			{
				if (beta == nullptr)
					return av_complex32 { 0.0f, 0.0f };
				else
					return reinterpret_cast<const av_complex32*>(beta)[0];
			}
			template<>
			av_complex64 getAlphaValue<av_complex64>(const void *alpha) noexcept
			{
				if (alpha == nullptr)
					return av_complex64 { 1.0, 0.0 };
				else
					return reinterpret_cast<const av_complex64*>(alpha)[0];
			}
			template<>
			av_complex64 getBetaValue<av_complex64>(const void *beta) noexcept
			{
				if (beta == nullptr)
					return av_complex64 { 0.0, 0.0 };
				else
					return reinterpret_cast<const av_complex64*>(beta)[0];
			}

#ifdef CUDA_BACKEND
			template<>
			cuComplex getAlphaValue<cuComplex>(const void *alpha) noexcept
			{
				if (alpha == nullptr)
					return cuComplex { 1.0f, 0.0f };
				else
					return reinterpret_cast<const cuComplex*>(alpha)[0];
			}
			template<>
			cuComplex getBetaValue<cuComplex>(const void *beta) noexcept
			{
				if (beta == nullptr)
					return cuComplex { 0.0f, 0.0f };
				else
					return reinterpret_cast<const cuComplex*>(beta)[0];
			}
			template<>
			cuDoubleComplex getAlphaValue<cuDoubleComplex>(const void *alpha) noexcept
			{
				if (alpha == nullptr)
					return double2 { 1.0, 0.0 };
				else
					return reinterpret_cast<const cuDoubleComplex*>(alpha)[0];
			}
			template<>
			cuDoubleComplex getBetaValue<cuDoubleComplex>(const void *beta) noexcept
			{
				if (beta == nullptr)
					return cuDoubleComplex { 0.0, 0.0 };
				else
					return reinterpret_cast<const cuDoubleComplex*>(beta)[0];
			}
#endif

			bool is_transpose(avGemmOperation_t op) noexcept
			{
				return op == AVOCADO_GEMM_OPERATION_T;
			}
			bool is_logical(avBinaryOp_t op) noexcept
			{
				return (op == AVOCADO_BINARY_OP_LOGICAL_AND) or (op == AVOCADO_BINARY_OP_LOGICAL_OR) or (op == AVOCADO_BINARY_OP_LOGICAL_OR);
			}
			bool is_logical(avUnaryOp_t op) noexcept
			{
				return op == AVOCADO_UNARY_OP_LOGICAL_NOT;
			}
			bool is_logical(avReduceOp_t op) noexcept
			{
				return (op == AVOCADO_REDUCE_LOGICAL_AND) or (op == AVOCADO_REDUCE_LOGICAL_OR);
			}

			thread_local ErrorDescription last_error;

			std::string ErrorDescription::toString() const
			{
				if (status == AVOCADO_STATUS_SUCCESS)
					return statusToString(status);
				else
					return statusToString(status) + " : in function '" + method_name + "' : '" + message + "'";
			}
			ErrorDescription getLastError()
			{
				ErrorDescription result;
				std::swap(result, last_error);
				return result;
			}
			ErrorDescription peekLastError()
			{
				return last_error;
			}
			avStatus_t reportError(avStatus_t status, const char *method, const std::string &msg)
			{
				last_error = ErrorDescription { status, std::string(method), msg };
				return status;
			}

#ifdef CUDA_BACKEND
			static const char* decode_cublas_status(cublasStatus_t status)
			{
				switch (status)
				{
					case CUBLAS_STATUS_SUCCESS:
						return "CUBLAS_STATUS_SUCCESS";
					case CUBLAS_STATUS_NOT_INITIALIZED:
						return "CUBLAS_STATUS_NOT_INITIALIZED";
					case CUBLAS_STATUS_ALLOC_FAILED:
						return "CUBLAS_STATUS_ALLOC_FAILED";
					case CUBLAS_STATUS_INVALID_VALUE:
						return "CUBLAS_STATUS_INVALID_VALUE";
					case CUBLAS_STATUS_ARCH_MISMATCH:
						return "CUBLAS_STATUS_ARCH_MISMATCH";
					case CUBLAS_STATUS_MAPPING_ERROR:
						return "CUBLAS_STATUS_MAPPING_ERROR";
					case CUBLAS_STATUS_EXECUTION_FAILED:
						return "CUBLAS_STATUS_EXECUTION_FAILED";
					case CUBLAS_STATUS_INTERNAL_ERROR:
						return "CUBLAS_STATUS_INTERNAL_ERROR";
					case CUBLAS_STATUS_NOT_SUPPORTED:
						return "CUBLAS_STATUS_NOT_SUPPORTED";
					case CUBLAS_STATUS_LICENSE_ERROR:
						return "CUBLAS_STATUS_LICENSE_ERROR";
					default:
						return "unknown status";
				}
			}
			std::runtime_error cuda_runtime_error_creator(const char* methodName, cudaError_t status)
			{
				std::string msg(methodName);
				msg += std::string(" : ") + cudaGetErrorName(status) + " (" + cudaGetErrorString(status);
				return std::runtime_error(msg);
			}
			std::runtime_error cublas_runtime_error_creator(const char* methodName, cublasStatus_t status)
			{
				std::string msg(methodName);
				msg += std::string(" : ") + decode_cublas_status(status);
				return std::runtime_error(msg);
			}
#endif
		}
	} /* namespace backend */
} /* namespace avocado */

