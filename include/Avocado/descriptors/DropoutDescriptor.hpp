/*
 * DropoutDescriptor.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_DESCRIPTORS_DROPOUTDESCRIPTOR_HPP_
#define AVOCADO_DESCRIPTORS_DROPOUTDESCRIPTOR_HPP_

#include <Avocado/backend_defs.h>
#include <Avocado/backend_utils.hpp>

#include <string>
#include <array>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{
			class DropoutDescriptor
			{
				public:
					static constexpr av_int64 descriptor_type = 7;

					static std::string className();
					static avDropoutDescriptor_t create();
					static void destroy(avDropoutDescriptor_t desc);
					static DropoutDescriptor& getObject(avDropoutDescriptor_t desc);

			};
		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_DESCRIPTORS_DROPOUTDESCRIPTOR_HPP_ */
