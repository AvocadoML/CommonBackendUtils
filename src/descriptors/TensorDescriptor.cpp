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

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;
	;
	template<typename T>
	std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> get_shape(T begin, T end)
	{
		std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> result;
		auto iter = std::copy(begin, end, result.begin());
		std::fill(iter, result.end(), 0);
		return result;
	}
	template<typename T>
	std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> get_strides(T begin, T end)
	{
		std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> result;
		result.fill(0);
		int nb_dims = std::distance(begin, end);
		if (nb_dims > 0)
		{
			result[nb_dims - 1] = 1;
			for (int i = nb_dims - 2; i >= 0; i--)
				result[i] = result[i + 1] * begin[i + 1];
		}
		return result;
	}

	DescriptorPool<TensorDescriptor>& get_descriptor_pool()
	{
		static DescriptorPool<TensorDescriptor> pool;
		return pool;
	}
}

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			TensorDescriptor::TensorDescriptor(std::initializer_list<int> dimensions, avDataType_t dtype) :
					m_dimensions(get_shape(dimensions.begin(), dimensions.end())),
					m_strides(get_strides(dimensions.begin(), dimensions.end())),
					m_number_of_dimensions(dimensions.size()),
					m_dtype(dtype)
			{
			}
			TensorDescriptor::TensorDescriptor(std::initializer_list<int> dimensions, std::initializer_list<int> strides, avDataType_t dtype) :
					m_dimensions(get_shape(dimensions.begin(), dimensions.end())),
					m_strides(get_shape(strides.begin(), strides.end())),
					m_number_of_dimensions(dimensions.size()),
					m_dtype(dtype)
			{
			}
			std::string TensorDescriptor::className()
			{
				return "TensorDescriptor";
			}

			avTensorDescriptor_t TensorDescriptor::create(std::initializer_list<int> dimensions, avDataType_t dtype)
			{
				return get_descriptor_pool().create(dimensions, dtype);
			}
			avTensorDescriptor_t TensorDescriptor::create(std::initializer_list<int> dimensions, std::initializer_list<int> strides,
					avDataType_t dtype)
			{
				return get_descriptor_pool().create(dimensions, strides, dtype);
			}
			void TensorDescriptor::destroy(avTensorDescriptor_t desc)
			{
				get_descriptor_pool().destroy(desc);
			}
			TensorDescriptor& TensorDescriptor::getObject(avTensorDescriptor_t desc)
			{
				return get_descriptor_pool().get(desc);
			}
			bool TensorDescriptor::isValid(avTensorDescriptor_t desc)
			{
				return get_descriptor_pool().isValid(desc);
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
				m_strides = get_strides(dimensions, dimensions + nbDims);
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
				assert(index >= 0 && index < m_number_of_dimensions);
				return m_dimensions[index];
			}
			int TensorDescriptor::operator[](int index) const
			{
				assert(index >= 0 && index < m_number_of_dimensions);
				return m_dimensions[index];
			}
			int TensorDescriptor::dimension(int index) const
			{
				assert(index >= 0 && index < m_number_of_dimensions);
				return m_dimensions[index];
			}
			int TensorDescriptor::stride(int index) const
			{
				assert(index >= 0 && index < m_number_of_dimensions);
				return m_strides[index];
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
				result += "] on " + deviceTypeToString(getCurrentDeviceType());
				return result;
			}

			bool isBroadcastPossible(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept
			{
				if (lhs.nbDims() < rhs.nbDims())
					return false;
				else
				{
					for (int i = 0, k = lhs.nbDims() - rhs.nbDims(); i < rhs.nbDims(); i++, k++)
						if (lhs.dimension(k) != rhs.dimension(i) and rhs.dimension(i) != 1)
							return false;
					return true;
				}
			}
			int volume(const BroadcastedDimensions &dims) noexcept
			{
				return dims.first * dims.last;
			}
			BroadcastedDimensions getBroadcastDimensions(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept
			{
				assert(isBroadcastPossible(lhs, rhs));
				int lhs_volume = lhs.volume();
				int rhs_volume = rhs.volume();
				assert(lhs_volume > 0 && rhs_volume > 0);
				BroadcastedDimensions result { lhs_volume / rhs_volume, rhs_volume };
				return result;
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

