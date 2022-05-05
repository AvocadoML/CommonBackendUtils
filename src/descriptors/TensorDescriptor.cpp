/*
 * TensorDescriptor.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/descriptors/TensorDescriptor.hpp>
#include "DescriptorPool.hpp"

#include <algorithm>
#include <cstring>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			static DescriptorPool<TensorDescriptor> tensor_descriptor_pool;

			TensorDescriptor::TensorDescriptor(std::initializer_list<int> dimensions, avDataType_t dtype) :
					m_number_of_dimensions(dimensions.size()),
					m_dtype(dtype)
			{
				m_dimensions.fill(0);
				m_strides.fill(0);
				std::memcpy(m_dimensions.data(), dimensions.begin(), sizeof(int) * dimensions.size());
				setup_stride();
			}
			std::string TensorDescriptor::className()
			{
				return "TensorDescriptor";
			}

			avTensorDescriptor_t TensorDescriptor::create(std::initializer_list<int> dimensions, avDataType_t dtype)
			{
				return tensor_descriptor_pool.create(dimensions, dtype);
			}
			void TensorDescriptor::destroy(avTensorDescriptor_t desc)
			{
				tensor_descriptor_pool.destroy(desc);
			}
			TensorDescriptor& TensorDescriptor::getObject(avTensorDescriptor_t desc)
			{
				return tensor_descriptor_pool.get(desc);
			}
			void TensorDescriptor::set(avDataType_t dtype, int nbDims, const int dimensions[])
			{
				if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS)
					throw std::invalid_argument("invalid number of dimensions");
				if (dimensions == nullptr and nbDims != 0)
					throw std::invalid_argument("null pointer passed as dimensions array");

				if (dimensions != nullptr)
					std::memcpy(m_dimensions.data(), dimensions, sizeof(int) * nbDims);
				m_number_of_dimensions = nbDims;
				m_dtype = dtype;
				setup_stride();
			}
			void TensorDescriptor::get(avDataType_t *dtype, int *nbDims, int dimensions[]) const
			{
				if (dtype != nullptr)
					dtype[0] = m_dtype;
				if (nbDims != nullptr)
					nbDims[0] = m_number_of_dimensions;
				if (dimensions != nullptr)
					std::memcpy(dimensions, m_dimensions.data(), sizeof(int) * m_number_of_dimensions);
			}
			int& TensorDescriptor::operator[](int index)
			{
				return m_dimensions[index];
			}
			int TensorDescriptor::operator[](int index) const
			{
				return m_dimensions[index];
			}
			int TensorDescriptor::dimension(int index) const
			{
				return m_dimensions[index];
			}
			int TensorDescriptor::nbDims() const noexcept
			{
				return m_number_of_dimensions;
			}
			av_int64 TensorDescriptor::sizeInBytes() const noexcept
			{
				return dataTypeSize(m_dtype) * this->volume();
			}
			int TensorDescriptor::getIndex(std::initializer_list<int> indices) const noexcept
			{
				assert(nbDims() == static_cast<int>(indices.size()));
				int result = 0;
				for (int i = 0; i < m_number_of_dimensions; i++)
				{
					const int idx = indices.begin()[i];
					assert(idx >= 0 && idx < m_dimensions[i]);
					result += idx * m_strides[i];
				}
				return result;
			}
			int TensorDescriptor::firstDim() const noexcept
			{
				if (m_number_of_dimensions == 0)
					return 0;
				else
					return m_dimensions[0];
			}
			int TensorDescriptor::lastDim() const noexcept
			{
				if (m_number_of_dimensions == 0)
					return 0;
				else
					return m_dimensions[m_number_of_dimensions - 1];
			}
			int TensorDescriptor::volume() const noexcept
			{
				if (m_number_of_dimensions == 0)
					return 0;
				else
				{
					int result = 1;
					for (int i = 0; i < m_number_of_dimensions; i++)
						result *= m_dimensions[i];
					return result;
				}
			}
			int TensorDescriptor::volumeWithoutFirstDim() const noexcept
			{
				if (m_number_of_dimensions == 0)
					return 0;
				else
				{
					int result = 1;
					for (int i = 1; i < m_number_of_dimensions; i++)
						result *= m_dimensions[i];
					return result;
				}
			}
			int TensorDescriptor::volumeWithoutLastDim() const noexcept
			{
				if (m_number_of_dimensions == 0)
					return 0;
				else
				{
					int result = 1;
					for (int i = 0; i < m_number_of_dimensions - 1; i++)
						result *= m_dimensions[i];
					return result;
				}
			}
			avDataType_t TensorDescriptor::dtype() const noexcept
			{
				return m_dtype;
			}
			bool TensorDescriptor::equalShape(const TensorDescriptor &other) noexcept
			{
				if (m_number_of_dimensions != other.m_number_of_dimensions)
					return false;
				for (int i = 0; i < m_number_of_dimensions; i++)
					if (m_dimensions[i] != other.m_dimensions[i])
						return false;
				return true;
			}
			std::string TensorDescriptor::toString() const
			{
				std::string result = std::string("Tensor<") + dtypeToString(m_dtype) + ">[";
				for (int i = 0; i < m_number_of_dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(m_dimensions[i]);
				}
				result += "] on " + deviceTypeToString();
				return result;
			}
			/*
			 * private
			 */
			void TensorDescriptor::setup_stride()
			{
				int tmp = 1;
				for (int i = m_number_of_dimensions - 1; i >= 0; i--)
				{
					m_strides[i] = tmp;
					tmp *= this->dimension(i);
				}
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

