/*
 * ContextDescriptor.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/descriptors/ContextDescriptor.hpp>
#include "DescriptorPool.hpp"

#include <stdexcept>
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

namespace
{
	using namespace avocado::backend::BACKEND_NAMESPACE;

	DescriptorPool<ContextDescriptor> create_default_pool()
	{
		try
		{
			DescriptorPool<ContextDescriptor> tmp;
			for (int i = 0; i < getNumberOfDevices(); i++)
#if USE_CUDA or USE_OPENCL
				tmp.create(i, true);
#else
				tmp.create();
#endif
			return tmp;
		} catch (std::exception &e)
		{
			return DescriptorPool<ContextDescriptor>();
		}
	}

}

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			static DescriptorPool<ContextDescriptor> context_descriptor_pool;
			thread_local DescriptorPool<ContextDescriptor> default_context_descriptor_pool = create_default_pool();

			bool isDefault(avContextDescriptor_t desc) noexcept
			{
				const int idx = getDescriptorIndex(desc);
				return 0 <= idx and idx < getNumberOfDevices();
			}

#if defined(CUDA_BACKEND)
			ContextDescriptor::ContextDescriptor(avDeviceIndex_t deviceIndex, bool useDefaultStream):
					m_device_index(deviceIndex)
			{
				cudaError_t err = cudaSetDevice(deviceIndex);
				CHECK_CUDA_ERROR(err, "ContextDescriptor::create() : cudaSetDevice()");
				if (useDefaultStream)
					m_stream = nullptr;
				else
				{
					err = cudaStreamCreate(&m_stream);
					CHECK_CUDA_ERROR(err, "ContextDescriptor::create() : cudaStreamCreate()");
				}

				cublasStatus_t status = cublasCreate_v2(&m_handle);
				CHECK_CUBLAS_STATUS(status, "ContextDescriptor::create() : cublasCreate_v2()");
				status = cublasSetStream_v2(m_handle, m_stream);
				CHECK_CUBLAS_STATUS(status, "ContextDescriptor::create() : cublasSetStream_v2()");
			}
#elif defined(OPENCL_BACKEND)
			ContextDescriptor::ContextDescriptor(avDeviceIndex_t deviceIndex, bool useDefaultCommandQueue):
					m_device_index(deviceIndex)
			{

			}
#endif
			ContextDescriptor::ContextDescriptor(ContextDescriptor &&other) :
#if defined(CUDA_BACKEND)
					m_stream(other.m_stream), m_handle(other.m_handle),
#elif defined(OPENCL_BACKEND)
#endif
					m_device_index(other.m_device_index),
					m_workspace(std::move(other.m_workspace)),
					m_workspace_size(other.m_workspace_size)
			{
#if defined(CUDA_BACKEND)
				other.m_stream = nullptr;
				other.m_handle = nullptr;
#elif defined(OPENCL_BACKEND)
#endif
				other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				other.m_workspace_size = 0;
			}
			ContextDescriptor& ContextDescriptor::operator=(ContextDescriptor &&other)
			{
#if defined(CPU_BACKEND)
#elif defined(CUDA_BACKEND)
				std::swap(this->m_stream, other.m_stream);
				std::swap(this->m_handle, other.m_handle);
#elif defined(OPENCL_BACKEND)
#endif
				std::swap(this->m_device_index, other.m_device_index);
				std::swap(this->m_workspace, other.m_workspace);
				std::swap(this->m_workspace_size, other.m_workspace_size);
				return *this;
			}
			ContextDescriptor::~ContextDescriptor()
			{
//				std::cout << device_type() << " : destroying context " << this << '\n';
#if defined(CPU_BACKEND)
#elif defined(CUDA_BACKEND)
				cudaError_t err = cudaDeviceSynchronize();
				CHECK_CUDA_ERROR(err, "ContextDescriptor::destroy() : cudaDeviceSynchronize()");
				if (m_handle != nullptr)
				{
					cublasStatus_t status = cublasDestroy_v2(m_handle);
					CHECK_CUBLAS_STATUS(status, "ContextDescriptor::destroy() : cublasDestroy_v2()");
				}

				if (m_stream != nullptr)
				{
					err = cudaStreamDestroy(m_stream);
					CHECK_CUDA_ERROR(err, "ContextDescriptor::destroy() : cudaStreamDestroy()");
				}
#elif defined(OPENCL_BACKEND)
#else
#endif
			}
			std::string ContextDescriptor::className()
			{
				return "ContextDescriptor";
			}
#if defined(CUDA_BACKEND)
			avContextDescriptor_t ContextDescriptor::create(avDeviceIndex_t deviceIndex, bool useDefaultStream)
			{
				return context_descriptor_pool.create(deviceIndex, useDefaultStream);
			}
#elif defined(OPENCL_BACKEND)
			avContextDescriptor_t ContextDescriptor::create(avDeviceIndex_t deviceIndex, bool useDefaultCommandQueue)
			{
				return context_descriptor_pool.create(deviceIndex, useDefaultCommandQueue);
			}
#else
			avContextDescriptor_t ContextDescriptor::create()
			{
				return context_descriptor_pool.create();
			}
#endif
			void ContextDescriptor::destroy(avContextDescriptor_t desc)
			{
				context_descriptor_pool.destroy(desc);
			}
			ContextDescriptor& ContextDescriptor::getObject(avContextDescriptor_t desc)
			{
				if (isDefault(desc))
					return default_context_descriptor_pool.get(desc);
				else
					return context_descriptor_pool.get(desc);
			}
			bool ContextDescriptor::isValid(avContextDescriptor_t desc)
			{
				if (isDefault(desc))
					return default_context_descriptor_pool.isValid(desc);
				else
					return context_descriptor_pool.isValid(desc);
			}

			MemoryDescriptor& ContextDescriptor::getWorkspace() const
			{
				if (m_workspace.isNull())
				{
					m_workspace_size = 1 << 23; // lazy allocation of 8MB workspace
					m_workspace = MemoryDescriptor(m_workspace_size, m_device_index);
				}
				return m_workspace;
			}
#ifdef CUDA_BACKEND
			void ContextDescriptor::setDevice() const
			{
				cudaError_t err = cudaSetDevice(m_device_index);
				CHECK_CUDA_ERROR(err, "ContextDescriptor::setDevice() : cudaSetDevice()");
			}
			avDeviceIndex_t ContextDescriptor::getDevice() const noexcept
			{
				return m_device_index;
			}
			cudaStream_t ContextDescriptor::getStream() const noexcept
			{
				return m_stream;
			}
			cublasHandle_t ContextDescriptor::getHandle() const noexcept
			{
				return m_handle;
			}
#endif

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

