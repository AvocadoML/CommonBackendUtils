/*
 * backend_utils.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/backend_utils.hpp>
#include <Avocado/descriptors/ContextDescriptor.hpp>
#include <Avocado/descriptors/ConvolutionDescriptor.hpp>
#include <Avocado/descriptors/DropoutDescriptor.hpp>
#include <Avocado/descriptors/MemoryDescriptor.hpp>
#include <Avocado/descriptors/OptimizerDescriptor.hpp>
#include <Avocado/descriptors/PoolingDescriptor.hpp>
#include <Avocado/descriptors/TensorDescriptor.hpp>

#include <algorithm>

#if defined(CPU_BACKEND)
#elif defined(CUDA_BACKEND)
#  include <cuda_runtime_api.h>
#  include <cuda_fp16.h>
#  include <cublas_v2.h>
#elif defined(OPENCL_BACKEND)
#  include <CL/cl.hpp>
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
					if (status != cudaSuccess)
						return 0;
					else
						return tmp;
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
				cudaError_t status = cudaGetCurrentDevice(&tmp);
				if (status != cudaSuccess)
					return 0;
				else
					return tmp;
#elif defined(OPENCL_BACKEND)
				return 0;
#else
				return 0;
#endif
			}

			av_int64 createDescriptor(int index, av_int64 type) noexcept
			{
				return (static_cast<av_int64>(getCurrentDeviceType()) << 56ull) | (type << 48ull)
						| (static_cast<av_int64>(getCurrentDeviceIndex()) << 32ull) | static_cast<av_int64>(index);
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
		}
	} /* namespace backend */
} /* namespace avocado */

