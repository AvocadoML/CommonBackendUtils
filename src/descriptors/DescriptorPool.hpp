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
#include <mutex>
#include <shared_mutex>
#include <algorithm>
#include <iostream>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{
			/*
			 * DescriptorPool
			 */
			template<typename T>
			class DescriptorPool
			{
					T m_null_descriptor;
					std::vector<T> m_pool;
					std::vector<int> m_available_descriptors;
					std::shared_mutex m_pool_mutex;
				public:
					DescriptorPool(size_t initialSize = 10, int numRestricted = 0)
					{
						m_pool.reserve(initialSize + numRestricted);
						m_available_descriptors.reserve(initialSize);
					}
					DescriptorPool(const DescriptorPool<T> &other) = delete;
					DescriptorPool(DescriptorPool<T> &&other) :
							m_pool(std::move(other.m_pool)),
							m_available_descriptors(std::move(other.m_available_descriptors))
					{
					}
					DescriptorPool& operator=(const DescriptorPool<T> &other) = delete;
					DescriptorPool& operator=(DescriptorPool<T> &&other)
					{
						std::swap(this->m_pool, other.m_pool);
						std::swap(this->m_available_descriptors, other.m_available_descriptors);
						return *this;
					}
					~DescriptorPool()
					{
						std::cout << "destroying DescriptorPool<" << T::className() << ">\n";
					}

					/**
					 * \brief Checks if the passed descriptor is valid.
					 * The descriptor is valid if and only if its index is within the size of m_pool vector and is not in the list of available descriptors.
					 */
					bool isValid(int64_t desc) const noexcept
					{
//						std::cout << __FUNCTION__ << "() " << __LINE__ << " : index = " << desc << '\n';
//						std::cout << __FUNCTION__ << "() " << __LINE__ << " : device type = " << get_current_device_type() << '\n';
						if (getCurrentDeviceType() != getDeviceType(desc))
						{
//							std::cout << __FUNCTION__ << "() " << __LINE__ << " : device type mismatch : " << get_current_device_type() << " vs "
//									<< get_device_type(desc) << std::endl;
							return false;
						}
						if (T::descriptor_type != getDescriptorType(desc))
						{
//							std::cout << __FUNCTION__ << "() " << __LINE__ << " : type mismatch : " << T::descriptor_type << " vs "
//									<< get_descriptor_type(desc) << std::endl;
							return false;
						}

						int index = getDescriptorIndex(desc);
//						std::cout << __FUNCTION__ << "() " << __LINE__ << " object index = " << index << '\n';
						if (index < 0 or index > static_cast<int>(m_pool.size()))
						{
//							std::cout << __FUNCTION__ << "() " << __LINE__ << " : out of bounds : " << index << " vs 0:" << m_pool.size()
//									<< std::endl;
							return false;
						}
						bool asdf = std::find(m_available_descriptors.begin(), m_available_descriptors.end(), index) == m_available_descriptors.end();
						if (asdf == false)
						{
//							std::cout << "not in available" << std::endl;
						}
						return asdf;
					}

					T& get(av_int64 desc)
					{
//						std::cout << __FUNCTION__ << "() " << __LINE__ << " : " << T::className() << " object index = " << getDescriptorIndex(desc)
//								<< '\n';
						std::shared_lock lock(m_pool_mutex);
						if (desc == AVOCADO_NULL_DESCRIPTOR)
							return m_null_descriptor;
						else
						{
							if (isValid(desc))
								return m_pool.at(getDescriptorIndex(desc));
							else
								throw std::logic_error("invalid descriptor " + std::to_string(desc) + " for pool type '" + T::className() + "'");
						}
					}

					template<typename ... Args>
					av_int64 create(Args &&... args)
					{
						std::unique_lock lock(m_pool_mutex);

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
						m_pool.at(index) = T(std::forward<Args>(args)...);

						av_int64 descriptor = createDescriptor(index, T::descriptor_type);
//						std::cout << __FUNCTION__ << "() " << __LINE__ << " : " << T::className() << " = " << tmp << ", object index = " << result
//								<< std::endl;
						return descriptor;
					}
					void destroy(av_int64 desc)
					{
						std::unique_lock lock(m_pool_mutex);
//						std::cout << __FUNCTION__ << "() " << __LINE__ << " : " << T::className() << " = " << desc << std::endl;
						if (not isValid(desc))
							throw std::logic_error("invalid descriptor " + std::to_string(desc) + " of type '" + T::className() + "'");
						int index = getDescriptorIndex(desc);
//						std::cout << __FUNCTION__ << "() " << __LINE__ << " object index = " << index << std::endl;
						m_pool.at(index) = T();
						m_available_descriptors.push_back(index);
					}
			};

		} /* namespace cpu/cuda/opencl/reference */
	} /* namespace backend */
} /* namespace avocado */

#endif /* DESCRIPTORS_DESCRIPTORPOOL_HPP_ */
