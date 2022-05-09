/*
 * ConvolutionDescriptor.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_DESCRIPTORS_CONVOLUTIONDESCRIPTOR_HPP_
#define AVOCADO_DESCRIPTORS_CONVOLUTIONDESCRIPTOR_HPP_

#include <Avocado/backend_defs.h>
#include <Avocado/backend_utils.hpp>
#include <Avocado/descriptors/TensorDescriptor.hpp>

#include <array>
#include <cstring>
#include <string>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			class ConvolutionDescriptor
			{
				public:
					static constexpr av_int64 descriptor_type = 4;
					static constexpr bool must_check_device_index = false;

					avConvolutionMode_t mode = AVOCADO_CONVOLUTION_MODE;
					int dimensions = 2;
					std::array<int, 3> padding;
					std::array<int, 3> stride;
					std::array<int, 3> dilation;
					std::array<uint8_t, 16> padding_value;
					int groups = 1;

					ConvolutionDescriptor() noexcept;

					static std::string className();
					static avConvolutionDescriptor_t create();
					static void destroy(avConvolutionDescriptor_t desc);
					static ConvolutionDescriptor& getObject(avConvolutionDescriptor_t desc);
					static bool isValid(avConvolutionDescriptor_t desc);

					void set(avConvolutionMode_t mode, int nbDims, const int padding[], const int strides[], const int dilation[], int groups,
							const void *paddingValue);
					void get(avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[], int dilation[], int *groups,
							void *paddingValue) const;

					template<typename T>
					T getPaddingValue() const noexcept
					{
						static_assert(sizeof(T) <= sizeof(padding_value), "");
						T result;
						std::memcpy(&result, padding_value.data(), sizeof(T));
						return result;
					}
					bool paddingWithZeros() const noexcept;
					TensorDescriptor getOutputShape(const TensorDescriptor &xDesc, const TensorDescriptor &wDesc) const;
					bool isStrided() const noexcept;
					bool isDilated() const noexcept;
					std::string toString() const;
			};

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_DESCRIPTORS_CONVOLUTIONDESCRIPTOR_HPP_ */
