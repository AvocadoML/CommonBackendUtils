/*
 * MemoryDescriptor.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_DESCRIPTORS_MEMORYDESCRIPTOR_HPP_
#define AVOCADO_DESCRIPTORS_MEMORYDESCRIPTOR_HPP_

#include <Avocado/backend_defs.h>
#include <Avocado/backend_utils.hpp>

#include <cinttypes>
#include <string>

#ifdef OPENCL_BACKEND
#  include <CL/cl.hpp>
#endif

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			class MemoryDescriptor
			{
#ifdef OPENCL_BACKEND
					// here will be cl::buffer
#else
					int8_t *m_data = nullptr;
#endif
					avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
					av_int64 m_size = 0;
					av_int64 m_offset = 0;
					bool m_is_owning = false;
				public:
					static constexpr av_int64 descriptor_type = 1;

					MemoryDescriptor() = default;
					MemoryDescriptor(av_int64 sizeInBytes, avDeviceIndex_t deviceIndex);
					MemoryDescriptor(const MemoryDescriptor &other, av_int64 sizeInBytes, av_int64 offsetInBytes);
					MemoryDescriptor(const MemoryDescriptor &other) = delete;
					MemoryDescriptor(MemoryDescriptor &&other);
					MemoryDescriptor& operator=(const MemoryDescriptor &other) = delete;
					MemoryDescriptor& operator=(MemoryDescriptor &&other);
					~MemoryDescriptor();
					bool isNull() const noexcept;
					av_int64 sizeInBytes() const noexcept;
					avDeviceIndex_t device() const noexcept;
					static std::string className();

					static avMemoryDescriptor_t create(av_int64 sizeInBytes, avDeviceIndex_t deviceIndex);
					/**
					 * \brief Creates a non-owning view of another memory block.
					 */
					static avMemoryDescriptor_t create(const MemoryDescriptor &other, av_int64 sizeInBytes, av_int64 offsetInBytes);

					/**
					 * \brief This method deallocates underlying memory and resets the descriptor.
					 * Calling this method on an already destroyed descriptor has no effect.
					 */
					static void destroy(avMemoryDescriptor_t desc);

					/**
					 * \brief Returns reference to the object behind the descriptor.
					 */
					static MemoryDescriptor& getObject(avMemoryDescriptor_t desc);

#ifdef OPENCL_BACKEND
//					cl::Buffer& data(void *ptr) noexcept;
//					const cl::Buffer& data(const void *ptr) const noexcept;

#else
					template<typename T = void>
					T* data() noexcept
					{
						return reinterpret_cast<T*>(m_data + m_offset);
					}
					template<typename T = void>
					const T* data() const noexcept
					{
						return reinterpret_cast<const T*>(m_data + m_offset);
					}
#endif
			};

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_DESCRIPTORS_MEMORYDESCRIPTOR_HPP_ */
