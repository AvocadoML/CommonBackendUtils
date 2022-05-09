/*
 * PoolingDescriptor.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/descriptors/PoolingDescriptor.hpp>
#include "DescriptorPool.hpp"

#include <stdexcept>
#include <algorithm>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			static DescriptorPool<PoolingDescriptor> pooling_descriptor_pool;

			std::string PoolingDescriptor::className()
			{
				return "PoolingDescriptor";
			}
			avPoolingDescriptor_t PoolingDescriptor::create()
			{
				return pooling_descriptor_pool.create();
			}
			void PoolingDescriptor::destroy(avPoolingDescriptor_t desc)
			{
				pooling_descriptor_pool.destroy(desc);
			}
			PoolingDescriptor& PoolingDescriptor::getObject(avPoolingDescriptor_t desc)
			{
				return pooling_descriptor_pool.get(desc);
			}
			bool PoolingDescriptor::isValid(avPoolingDescriptor_t desc)
			{
				return pooling_descriptor_pool.isValid(desc);
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

