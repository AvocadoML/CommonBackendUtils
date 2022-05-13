/*
 * MemoryDescriptor.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/descriptors/MemoryDescriptor.hpp>
#include "DescriptorPool.hpp"

#include <stdexcept>
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

			static DescriptorPool<MemoryDescriptor> memory_descriptor_pool;

			MemoryDescriptor::MemoryDescriptor(av_int64 sizeInBytes, avDeviceIndex_t deviceIndex)
			{
				if (sizeInBytes > 0)
				{
#if defined(CUDA_BACKEND)
					cudaError_t err = cudaSetDevice(deviceIndex);
					CHECK_CUDA_ERROR(err);
					err = cudaMalloc(reinterpret_cast<void**>(&m_data), sizeInBytes);
					CHECK_CUDA_ERROR(err);
#elif defined(OPENCL_BACKEND)

#else
					m_data = new int8_t[sizeInBytes];
#endif
				}
				m_device_index = 0;
				m_offset = 0;
				m_size = sizeInBytes;
				m_is_owning = true;
			}
			MemoryDescriptor::MemoryDescriptor(const MemoryDescriptor &other, av_int64 sizeInBytes, av_int64 offsetInBytes)
			{
				if (other.m_is_owning == false)
					throw std::logic_error("cannot create memory view from non-owning memory descriptor");
				if (other.m_size < offsetInBytes + sizeInBytes)
					throw std::logic_error(
							"the view would extend beyond the original tensor : " + std::to_string(other.m_size) + " < "
									+ std::to_string(offsetInBytes) + "+" + std::to_string(sizeInBytes));
#ifndef OPENCL_BACKEND
				m_data = other.m_data;
#endif
				m_device_index = other.m_device_index;
				m_size = sizeInBytes;
				m_offset = other.m_offset + offsetInBytes;
				m_is_owning = false;
			}
			MemoryDescriptor::MemoryDescriptor(MemoryDescriptor &&other) :
#ifndef OPENCL_BACKEND
					m_data(other.m_data),
#endif
					m_device_index(other.m_device_index),
					m_size(other.m_size),
					m_offset(other.m_offset),
					m_is_owning(other.m_is_owning)
			{
#ifndef OPENCL_BACKEND
				other.m_data = nullptr;
#endif
				other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				other.m_offset = 0;
				other.m_size = 0;
				other.m_is_owning = false;
			}
			MemoryDescriptor& MemoryDescriptor::operator=(MemoryDescriptor &&other)
			{
#ifndef OPENCL_BACKEND
				std::swap(this->m_data, other.m_data);
#endif
				std::swap(this->m_device_index, other.m_device_index);
				std::swap(this->m_size, other.m_size);
				std::swap(this->m_offset, other.m_offset);
				std::swap(this->m_is_owning, other.m_is_owning);
				return *this;
			}
			MemoryDescriptor::~MemoryDescriptor()
			{
#ifndef OPENCL_BACKEND
				if (m_data != nullptr)
				{
					if (m_is_owning)
					{
#  if defined(CUDA_BACKEND)
						cudaError_t err = cudaFree(m_data);
						CHECK_CUDA_ERROR(err);
						delete[] m_data;
#  endif
					}
					m_data = nullptr;
				}
#endif
				m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				m_size = 0;
				m_offset = 0;
				m_is_owning = false;
			}
			bool MemoryDescriptor::isNull() const noexcept
			{
#if defined(OPENCL_BACKEND)
				return true;
#else
				return m_data == nullptr;
#endif
			}
			av_int64 MemoryDescriptor::sizeInBytes() const noexcept
			{
				return m_size;
			}
			avDeviceIndex_t MemoryDescriptor::device() const noexcept
			{
				return m_device_index;
			}
			std::string MemoryDescriptor::className()
			{
				return "MemoryDescriptor";
			}

			avMemoryDescriptor_t MemoryDescriptor::create(av_int64 sizeInBytes, avDeviceIndex_t deviceIndex)
			{
				return memory_descriptor_pool.create(sizeInBytes, deviceIndex);
			}
			avMemoryDescriptor_t MemoryDescriptor::create(const MemoryDescriptor &other, av_int64 sizeInBytes, av_int64 offsetInBytes)
			{
				return memory_descriptor_pool.create(other, sizeInBytes, offsetInBytes);
			}
			void MemoryDescriptor::destroy(avMemoryDescriptor_t desc)
			{
				memory_descriptor_pool.destroy(desc);
			}
			MemoryDescriptor& MemoryDescriptor::getObject(avMemoryDescriptor_t desc)
			{
				return memory_descriptor_pool.get(desc);
			}
			bool MemoryDescriptor::isValid(avMemoryDescriptor_t desc)
			{
				return memory_descriptor_pool.isValid(desc);
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

