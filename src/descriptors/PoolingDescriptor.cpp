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

namespace
{
	using namespace avocado::backend::BACKEND_NAMESPACE;

	DescriptorPool<PoolingDescriptor>& get_descriptor_pool()
	{
		static DescriptorPool<PoolingDescriptor> pool;
		return pool;
	}
}

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			std::string PoolingDescriptor::className()
			{
				return "PoolingDescriptor";
			}
			avPoolingDescriptor_t PoolingDescriptor::create()
			{
				return get_descriptor_pool().create();
			}
			void PoolingDescriptor::destroy(avPoolingDescriptor_t desc)
			{
				get_descriptor_pool().destroy(desc);
			}
			PoolingDescriptor& PoolingDescriptor::getObject(avPoolingDescriptor_t desc)
			{
				return get_descriptor_pool().get(desc);
			}
			bool PoolingDescriptor::isValid(avPoolingDescriptor_t desc)
			{
				return get_descriptor_pool().isValid(desc);
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

