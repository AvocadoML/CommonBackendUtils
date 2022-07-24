/*
 * DropoutDescriptor.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/descriptors/DropoutDescriptor.hpp>
#include "DescriptorPool.hpp"

#include <stdexcept>
#include <algorithm>

namespace
{
	using namespace avocado::backend::BACKEND_NAMESPACE;

	DescriptorPool<DropoutDescriptor>& get_descriptor_pool()
	{
		static DescriptorPool<DropoutDescriptor> pool;
		return pool;
	}
}

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{
			std::string DropoutDescriptor::className()
			{
				return "DropoutDescriptor";
			}
			avDropoutDescriptor_t DropoutDescriptor::create()
			{
				return get_descriptor_pool().create();
			}
			void DropoutDescriptor::destroy(avDropoutDescriptor_t desc)
			{
				get_descriptor_pool().destroy(desc);
			}
			DropoutDescriptor& DropoutDescriptor::getObject(avDropoutDescriptor_t desc)
			{
				return get_descriptor_pool().get(desc);
			}
			bool DropoutDescriptor::isValid(avDropoutDescriptor_t desc)
			{
				return get_descriptor_pool().isValid(desc);
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

