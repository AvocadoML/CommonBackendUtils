/*
 * DescriptorPool.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef DESCRIPTORS_DESCRIPTORPOOL_HPP_
#define DESCRIPTORS_DESCRIPTORPOOL_HPP_

#include <Avocado/backend_defs.h>
#include <Avocado/backend_utils.hpp>

#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>
#include <iostream>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{
			template<typename T>
			class DescriptorPool
			{
					T m_null_descriptor;
					std::vector<std::unique_ptr<T>> m_pool;
					std::vector<int> m_available_descriptors;
					mutable std::mutex m_pool_mutex;
				public:
					DescriptorPool(size_t initialSize = 10, int numRestricted = 0)
					{
						m_pool.reserve(initialSize + numRestricted);
						m_available_descriptors.reserve(initialSize);
					}
					DescriptorPool(const DescriptorPool<T> &other) = delete;
					DescriptorPool(DescriptorPool<T> &&other) noexcept :
							m_null_descriptor(std::move(other.m_null_descriptor)),
							m_pool(std::move(other.m_pool)),
							m_available_descriptors(std::move(other.m_available_descriptors))
					{
					}
					DescriptorPool<T>& operator=(const DescriptorPool<T> &other) = delete;
					DescriptorPool<T>& operator=(DescriptorPool<T> &&other) noexcept
					{
						std::swap(this->m_null_descriptor, other.m_null_descriptor);
						std::swap(this->m_pool, other.m_pool);
						std::swap(this->m_available_descriptors, other.m_available_descriptors);
						return *this;
					}

					/**
					 * \brief Checks if the passed descriptor is valid.
					 * The descriptor is valid if and only if its index is within the size of m_pool vector and is not in the list of available descriptors.
					 */
					bool isValid(int64_t desc) const noexcept
					{
						std::lock_guard<std::mutex> lock(m_pool_mutex);
						return check_validity(desc) == 0;
					}

					T& get(av_int64 desc)
					{
						std::lock_guard<std::mutex> lock(m_pool_mutex);
						if (desc == AVOCADO_NULL_DESCRIPTOR)
							return m_null_descriptor;
						else
						{
							validate_descriptor(desc);
							return *m_pool.at(getDescriptorIndex(desc));
						}
					}

					template<typename ... Args>
					av_int64 create(Args &&... args)
					{
						std::lock_guard<std::mutex> lock(m_pool_mutex);

						int index;
						if (m_available_descriptors.size() > 0)
						{
							index = m_available_descriptors.back();
							m_available_descriptors.pop_back();
						}
						else
						{
							m_pool.emplace_back();
							index = m_pool.size() - 1;
						}
						if (m_pool.at(index) == nullptr)
							m_pool.at(index) = std::make_unique<T>(std::forward<Args>(args)...);
						else
							*m_pool.at(index) = T(std::forward<Args>(args)...);

						av_int64 descriptor = createDescriptor(index, T::descriptor_type);
						return descriptor;
					}
					void destroy(av_int64 desc)
					{
						std::lock_guard<std::mutex> lock(m_pool_mutex);
						validate_descriptor(desc);
						int index = getDescriptorIndex(desc);
						try
						{
							*m_pool.at(index) = T();
							m_available_descriptors.push_back(index);
						} catch (std::exception &e)
						{
							throw DeallocationError(e.what());
						}
					}
				private:
					void validate_descriptor(int64_t desc)
					{
						switch (check_validity(desc))
						{
							case -1:
								throw InvalidDescriptorError("descriptor has negative value (" + std::to_string(desc) + ")");
							case -2:
								throw InvalidDescriptorError(
										"descriptor type mismatch, expected " + T::className() + " (" + std::to_string(T::descriptor_type) + "), got "
												+ descriptorTypeToString(desc));
							case -3:
								throw InvalidDescriptorError(
										"device type mismatch, expected " + deviceTypeToString(getCurrentDeviceType()) + ", got "
												+ deviceTypeToString(getDeviceType(desc)));
							case -4:
								throw InvalidDescriptorError(
										"device index mismatch, expected " + std::to_string(getCurrentDeviceIndex()) + ", got "
												+ std::to_string(getDeviceIndex(desc)));
							case -5:
								throw InvalidDescriptorError(
										"descriptor index " + std::to_string(getDescriptorIndex(desc)) + " out of range [0,"
												+ std::to_string(m_pool.size()) + ")");
							case -6:
								throw InvalidDescriptorError("descriptor is not used");
						}
					}
					int check_validity(int64_t desc) const noexcept
					{
						if (desc == AVOCADO_NULL_DESCRIPTOR)
							return 0; // null descriptor is valid
						if (desc < 0)
							return -1; // totally invalid descriptor
						if (T::descriptor_type != getDescriptorType(desc))
							return -2; // descriptor type mismatch
						if (getCurrentDeviceType() != getDeviceType(desc))
							return -3; // device type mismatch
						if (T::must_check_device_index and getCurrentDeviceIndex() != getDeviceIndex(desc))
							return -4; // device index mismatch

						const int index = getDescriptorIndex(desc);
						if (index < 0 or index > static_cast<int>(m_pool.size()))
							return -5; // descriptor index out of range
						if (std::find(m_available_descriptors.begin(), m_available_descriptors.end(), index) != m_available_descriptors.end())
							return -6; // descriptor not in use
						return 0;
					}
			};

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* DESCRIPTORS_DESCRIPTORPOOL_HPP_ */
