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

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			static DescriptorPool<DropoutDescriptor> dropout_descriptor_pool;

			std::string DropoutDescriptor::className()
			{
				return "DropoutDescriptor";
			}
			avDropoutDescriptor_t DropoutDescriptor::create()
			{
				return dropout_descriptor_pool.create();
			}
			void DropoutDescriptor::destroy(avDropoutDescriptor_t desc)
			{
				dropout_descriptor_pool.destroy(desc);
			}
			DropoutDescriptor& DropoutDescriptor::getObject(avDropoutDescriptor_t desc)
			{
				return dropout_descriptor_pool.get(desc);
			}
			bool DropoutDescriptor::isValid(avDropoutDescriptor_t desc)
			{
				return dropout_descriptor_pool.isValid(desc);
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

