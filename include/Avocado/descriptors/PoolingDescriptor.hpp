/*
 * PoolingDescriptor.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_DESCRIPTORS_POOLINGDESCRIPTOR_HPP_
#define AVOCADO_DESCRIPTORS_POOLINGDESCRIPTOR_HPP_

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
			class PoolingDescriptor
			{
				public:
					static constexpr av_int64 descriptor_type = 5;
					avPoolingMode_t mode = AVOCADO_POOLING_MAX;
					std::array<int, 3> filter;
					std::array<int, 3> padding;
					std::array<int, 3> stride;

					static std::string className();
					static avPoolingDescriptor_t create();
					static void destroy(avPoolingDescriptor_t desc);
					static PoolingDescriptor& getObject(avPoolingDescriptor_t desc);

			};
		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_DESCRIPTORS_POOLINGDESCRIPTOR_HPP_ */
