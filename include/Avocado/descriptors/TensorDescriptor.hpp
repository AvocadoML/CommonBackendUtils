/*
 * TensorDescriptor.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_DESCRIPTORS_TENSORDESCRIPTOR_HPP_
#define AVOCADO_DESCRIPTORS_TENSORDESCRIPTOR_HPP_

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

			class TensorDescriptor
			{
					std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> m_dimensions;
					std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> m_strides;
					int m_number_of_dimensions = 0;
					avDataType_t m_dtype = AVOCADO_DTYPE_UNKNOWN;
				public:
					static constexpr av_int64 descriptor_type = 3;
					static constexpr bool must_check_device_index = false;

					TensorDescriptor() = default;
					TensorDescriptor(std::initializer_list<int> dimensions, avDataType_t dtype);
					static std::string className();

					static avTensorDescriptor_t create(std::initializer_list<int> dimensions, avDataType_t dtype);
					static void destroy(avTensorDescriptor_t desc);
					static TensorDescriptor& getObject(avTensorDescriptor_t desc);
					static bool isValid(avTensorDescriptor_t desc);

					void set(avDataType_t dtype, int nbDims, const int dimensions[]);
					void get(avDataType_t *dtype, int *nbDims, int dimensions[]) const;

					int& operator[](int index);
					int operator[](int index) const;
					int dimension(int index) const;
					int nbDims() const noexcept;
					av_int64 sizeInBytes() const noexcept;
					int getIndex(std::initializer_list<int> indices) const noexcept;
					int firstDim() const noexcept;
					int lastDim() const noexcept;
					int volume() const noexcept;
					int volumeWithoutFirstDim() const noexcept;
					int volumeWithoutLastDim() const noexcept;
					avDataType_t dtype() const noexcept;

					bool equalShape(const TensorDescriptor &other) noexcept;
					std::string toString() const;
				private:
					void setup_stride();
			};

			/**
			 * Only the right hand side (rhs) operand can be broadcasted into the left hand side (lhs).
			 * The number of dimensions of the rhs tensor must be lower or equal to the lhs tensor.
			 * All k dimensions of the rhs must match the last k dimensions of the lhs.
			 *
			 */
			bool isBroadcastPossible(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept;
			int volume(const BroadcastedDimensions &dims) noexcept;
			BroadcastedDimensions getBroadcastDimensions(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept;
		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_DESCRIPTORS_TENSORDESCRIPTOR_HPP_ */
