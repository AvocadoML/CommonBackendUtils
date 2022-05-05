/*
 * backend_utils.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/backend_utils.hpp>

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
			std::string deviceTypeToString()
			{
#if defined(CPU_BACKEND)
				return "CPU";
#elif defined(CUDA_BACKEND)
				return "CUDA";
#elif defined(OPENCL_BACKEND)
				return "OPENCL";
#else
				return "reference";
#endif
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

			av_int64 getCurrentDeviceType() noexcept
			{
#if defined(CUDA_BACKEND)
				return static_cast<av_int64>(AVOCADO_DEVICE_CUDA);
#elif defined(OPENCL_BACKEND)
				return static_cast<av_int64>(AVOCADO_DEVICE_OPENCL);
#else
				return static_cast<av_int64>(AVOCADO_DEVICE_CPU);
#endif
			}
			av_int64 getCurrentDeviceIndex() noexcept
			{
				return 0;
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
		}
	} /* namespace backend */
} /* namespace avocado */

