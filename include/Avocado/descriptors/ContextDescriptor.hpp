/*
 * ContextDescriptor.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_DESCRIPTORS_CONTEXTDESCRIPTOR_HPP_
#define AVOCADO_DESCRIPTORS_CONTEXTDESCRIPTOR_HPP_

#include <Avocado/backend_defs.h>
#include <Avocado/backend_utils.hpp>
#include <Avocado/descriptors/MemoryDescriptor.hpp>

#include <string>

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

			class ContextDescriptor
			{
#if defined(CPU_BACKEND)
#elif defined(CUDA_BACKEND)
					cudaStream_t m_stream = nullptr;
					cublasHandle_t m_handle = nullptr;
#elif defined(OPENCL_BACKEND)
#else
#endif
					avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
					mutable MemoryDescriptor m_workspace;
					mutable av_int64 m_workspace_size = 0;
				public:
					static constexpr av_int64 descriptor_type = 2;
					static constexpr bool must_check_device_index = true;

					ContextDescriptor() = default;
#if defined(CUDA_BACKEND)
					ContextDescriptor(avDeviceIndex_t deviceIndex, bool useDefaultStream = false);
#elif defined(OPENCL_BACKEND)
					ContextDescriptor(avDeviceIndex_t deviceIndex, bool useDefaultCommandQueue);
#endif
					ContextDescriptor(const ContextDescriptor &other) = delete;
					ContextDescriptor(ContextDescriptor &&other);
					ContextDescriptor& operator=(const ContextDescriptor &other) = delete;
					ContextDescriptor& operator=(ContextDescriptor &&other);
					~ContextDescriptor();
					static std::string className();
#if defined(CUDA_BACKEND)
					static avContextDescriptor_t create(avDeviceIndex_t deviceIndex, bool useDefaultStream = false);
#elif defined(OPENCL_BACKEND)
					static avContextDescriptor_t create(avDeviceIndex_t deviceIndex, bool useDefaultCommandQueue);
#else
					static avContextDescriptor_t create();
#endif
					static void destroy(avContextDescriptor_t desc);
					static ContextDescriptor& getObject(avContextDescriptor_t desc);
					static bool isValid(avContextDescriptor_t desc);

					MemoryDescriptor& getWorkspace() const;
#ifdef CUDA_BACKEND
					void setDevice() const;
					avDeviceIndex_t getDevice() const noexcept;
					cudaStream_t getStream() const noexcept;
					cublasHandle_t getHandle() const noexcept;
#endif
			};

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_DESCRIPTORS_CONTEXTDESCRIPTOR_HPP_ */
