/*
 * ConvolutionDescriptor.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/descriptors/ConvolutionDescriptor.hpp>
#include "DescriptorPool.hpp"

#include <stdexcept>
#include <algorithm>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			static DescriptorPool<ConvolutionDescriptor> convolution_descriptor_pool;

			ConvolutionDescriptor::ConvolutionDescriptor() noexcept
			{
				padding.fill(0);
				stride.fill(1);
				dilation.fill(1);
				padding_value.fill(0u);
				groups = 1;
			}
			std::string ConvolutionDescriptor::className()
			{
				return "ConvolutionDescriptor";
			}
			avConvolutionDescriptor_t ConvolutionDescriptor::create()
			{
				return convolution_descriptor_pool.create();
			}
			void ConvolutionDescriptor::destroy(avConvolutionDescriptor_t desc)
			{
				convolution_descriptor_pool.destroy(desc);
			}
			ConvolutionDescriptor& ConvolutionDescriptor::getObject(avConvolutionDescriptor_t desc)
			{
				return convolution_descriptor_pool.get(desc);
			}
			bool ConvolutionDescriptor::isValid(avConvolutionDescriptor_t desc)
			{
				return convolution_descriptor_pool.isValid(desc);
			}
			void ConvolutionDescriptor::set(avConvolutionMode_t mode, int nbDims, const int padding[], const int strides[], const int dilation[],
					int groups, const void *paddingValue)
			{
				if (nbDims < 0 or nbDims > 3)
					throw std::invalid_argument("");
				this->mode = mode;
				this->dimensions = nbDims;
				if (strides != nullptr)
					std::memcpy(this->stride.data(), strides, sizeof(int) * dimensions);
				if (padding != nullptr)
					std::memcpy(this->padding.data(), padding, sizeof(int) * dimensions);
				if (dilation != nullptr)
					std::memcpy(this->dilation.data(), dilation, sizeof(int) * dimensions);

				this->groups = groups;
				if (paddingValue != nullptr)
					std::memcpy(this->padding_value.data(), paddingValue, sizeof(int8_t) * padding_value.size());
			}
			void ConvolutionDescriptor::get(avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[], int dilation[], int *groups,
					void *paddingValue) const
			{
				if (mode != nullptr)
					mode[0] = this->mode;
				if (nbDims != nullptr)
					nbDims[0] = this->dimensions;
				if (strides != nullptr)
					std::memcpy(strides, this->stride.data(), sizeof(int) * dimensions);
				if (padding != nullptr)
					std::memcpy(padding, this->padding.data(), sizeof(int) * dimensions);
				if (dilation != nullptr)
					std::memcpy(dilation, this->dilation.data(), sizeof(int) * dimensions);

				if (groups != nullptr)
					groups[0] = this->groups;
				if (paddingValue != nullptr)
					std::memcpy(paddingValue, this->padding_value.data(), sizeof(int8_t) * padding_value.size());
			}
			bool ConvolutionDescriptor::paddingWithZeros() const noexcept
			{
				return std::all_of(padding_value.begin(), padding_value.end(), [](uint8_t x)
				{	return x == 0u;});
			}
			TensorDescriptor ConvolutionDescriptor::getOutputShape(const TensorDescriptor &xDesc, const TensorDescriptor &wDesc) const
			{
				std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> shape;
				shape[0] = xDesc.firstDim(); // batch size
				for (int i = 0; i < dimensions; i++)
					shape[1 + i] = 1 + (xDesc.dimension(1 + i) - 2 * padding[i] - (((wDesc.dimension(1 + i) - 1) * dilation[i]) + 1)) / stride[i];
				shape[xDesc.nbDims() - 1] = wDesc.firstDim(); // output filters

				TensorDescriptor result;
				result.set(xDesc.dtype(), xDesc.nbDims(), shape.data());
				return result;
			}
			bool ConvolutionDescriptor::isStrided() const noexcept
			{
				for (int i = 0; i < dimensions; i++)
					if (stride[i] > 1)
						return true;
				return false;
			}
			bool ConvolutionDescriptor::isDilated() const noexcept
			{
				for (int i = 0; i < dimensions; i++)
					if (dilation[i] > 1)
						return true;
				return false;
			}
			std::string ConvolutionDescriptor::toString() const
			{
				std::string result;
				if (mode == AVOCADO_CONVOLUTION_MODE)
					result += "convolution : ";
				else
					result += "cross-correlation : ";
				result += "padding = [";
				for (int i = 0; i < dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(padding[i]);
				}
				result += "], strides = [";
				for (int i = 0; i < dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(stride[i]);
				}
				result += "], dilation = [";
				for (int i = 0; i < dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(dilation[i]);
				}
				result += "], groups = " + std::to_string(groups);
				return result;
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

