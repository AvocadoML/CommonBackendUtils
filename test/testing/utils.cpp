/*
 * utils.cpp
 *
 *  Created on: May 8, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/testing/utils.hpp>
#include <Avocado/backend_defs.h>
#if defined(CPU_BACKEND)
#  include <Avocado/cpu_backend.h>
#elif defined(CUDA_BACKEND)
#  include <Avocado/cuda_backend.h>
#elif defined(OPENCL_BACKEND)
#  include <Avocado/opencl_backend.h>
#endif
#include <Avocado/reference_backend.h>

#include <complex>
#include <cmath>
#include <memory>
#include <iostream>
#include <cassert>

namespace
{
	using namespace avocado;
	using namespace avocado::backend;

	template<typename T>
	void init_for_test(void *ptr, size_t elements, T offset, T minValue, T maxValue)
	{
		for (size_t i = 0; i < elements; i++)
			reinterpret_cast<T*>(ptr)[i] = minValue + 0.5 * (1.0 + sin(i / 10.0 + offset)) * (maxValue - minValue);
	}

	template<typename T>
	float diff_for_test(const void *ptr1, const void *ptr2, size_t elements)
	{
		double result = 0.0;
		for (size_t i = 0; i < elements; i++)
			result += std::fabs(reinterpret_cast<const T*>(ptr1)[i] - reinterpret_cast<const T*>(ptr2)[i]);
		return result / elements;
	}

	template<typename T>
	float norm_for_test(const void *ptr, size_t elements)
	{
		double result = 0.0;
		for (size_t i = 0; i < elements; i++)
			result += fabs(reinterpret_cast<const T*>(ptr)[i]);
		return result;
	}

	template<typename T>
	void abs_for_test(void *ptr, size_t elements)
	{
		for (size_t i = 0; i < elements; i++)
			reinterpret_cast<T*>(ptr)[i] = fabs(reinterpret_cast<T*>(ptr)[i]);
	}
	template<typename T>
	std::unique_ptr<T[]> toArray(const TensorWrapper &t)
	{
		std::unique_ptr<T[]> result = std::make_unique<T[]>(t.volume());
		t.copyToHost(result.get());
		return result;
	}
	template<typename T>
	void fromArray(TensorWrapper &dst, const std::unique_ptr<T[]> &src)
	{
		dst.copyFromHost(src.get());
	}

	std::ostream& operator<<(std::ostream &stream, av_bfloat16 x)
	{
		float tmp;
		refChangeTypeHost(AVOCADO_NULL_DESCRIPTOR, &tmp, AVOCADO_DTYPE_FLOAT32, &x, AVOCADO_DTYPE_BFLOAT16, 1);
		stream << tmp;
		return stream;
	}
	std::ostream& operator<<(std::ostream &stream, av_float16 x)
	{
		float tmp;
		refChangeTypeHost(AVOCADO_NULL_DESCRIPTOR, &tmp, AVOCADO_DTYPE_FLOAT32, &x, AVOCADO_DTYPE_FLOAT16, 1);
		stream << tmp;
		return stream;
	}

	template<typename T>
	void print(const std::vector<T> &vec)
	{
		for (size_t i = 0; i < vec.size(); i++)
			std::cout << vec[i] << ' ';
		std::cout << '\n';
	}

	int dtypeSize(avDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case AVOCADO_DTYPE_UNKNOWN:
				return 0;
			case AVOCADO_DTYPE_UINT8:
			case AVOCADO_DTYPE_INT8:
				return 1;
			case AVOCADO_DTYPE_INT16:
			case AVOCADO_DTYPE_FLOAT16:
			case AVOCADO_DTYPE_BFLOAT16:
				return 2;
			case AVOCADO_DTYPE_INT32:
			case AVOCADO_DTYPE_FLOAT32:
				return 4;
			case AVOCADO_DTYPE_INT64:
			case AVOCADO_DTYPE_FLOAT64:
			case AVOCADO_DTYPE_COMPLEX32:
				return 8;
			case AVOCADO_DTYPE_COMPLEX64:
				return 16;
		}
	}

	std::vector<int> winograd_matrices_shape(const std::vector<int> &inputShape, const std::vector<int> &filterShape, int transformSize) noexcept
	{
		int nb_tiles = inputShape[0]; // batch size
		for (size_t i = 1; i < inputShape.size() - 1; i++)
			nb_tiles *= ((inputShape[i] + transformSize - 1) / transformSize);
		int tile_size = filterShape[1] + transformSize - 1;

		return std::vector<int>( { tile_size * tile_size, nb_tiles, inputShape.back() });
	}
	template<typename T>
	T square(T x) noexcept
	{
		return x * x;
	}
}

namespace avocado
{
	namespace backend
	{

		DualTensorWrapper::DualTensorWrapper(std::vector<int> shape, avDataType_t dtype, avDeviceIndex_t device) :
		reference(shape, dtype, device),
				tested(shape, dtype, device)
		{
		}
		const TensorWrapper& DualTensorWrapper::getReference() const
		{
			return reference;
		}
		TensorWrapper& DualTensorWrapper::getReference()
		{
			return reference;
		}
		const TensorWrapper& DualTensorWrapper::getTested() const
		{
			return tested;
		}
		TensorWrapper& DualTensorWrapper::getTested()
		{
			return tested;
		}
		double DualTensorWrapper::getDifference() const
		{
			return diffForTest(reference, tested);
		}

//		void setMasterContext(avDeviceIndex_t deviceIndex, bool useDefault)
//		{
//			master_context = ContextWrapper(deviceIndex);
//		}
		static const ContextWrapper& getMasterContext()
		{
			static const ContextWrapper master_context(0, true, true);
			return master_context;
		}

		avContextDescriptor_t getContext()
		{
			return getMasterContext().getDescriptor();
		}
		avContextDescriptor_t getReferenceContext()
		{
			return getMasterContext().getRefDescriptor();
		}
		avDeviceIndex_t getDevice()
		{
			return getMasterContext().getDeviceIndex();
		}

		bool supportsType(avDataType_t dtype)
		{
			switch (dtype)
			{
				default:
				case AVOCADO_DTYPE_UNKNOWN:
					return false;
				case AVOCADO_DTYPE_UINT8:
				case AVOCADO_DTYPE_INT8:
				case AVOCADO_DTYPE_INT16:
				case AVOCADO_DTYPE_INT32:
				case AVOCADO_DTYPE_INT64:
					return true;
				case AVOCADO_DTYPE_FLOAT16:
				{
					bool result = false;
#if defined(CPU_BACKEND)
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, &result);
#elif defined(CUDA_BACKEND)
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, &result);
#endif
					return result;
				}
				case AVOCADO_DTYPE_BFLOAT16:
				{
					bool result = false;
#if defined(CPU_BACKEND)
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_BFLOAT16, &result);
#elif defined(CUDA_BACKEND)
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_BFLOAT16, &result);
#endif
					return result;
				}
				case AVOCADO_DTYPE_FLOAT32:
				case AVOCADO_DTYPE_COMPLEX32:
				{
					bool result = false;
#if defined(CPU_BACKEND)
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION, &result);
#elif defined(CUDA_BACKEND)
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION, &result);
#endif
					return result;
				}
				case AVOCADO_DTYPE_FLOAT64:
				case AVOCADO_DTYPE_COMPLEX64:
				{
					bool result = false;
#if defined(CPU_BACKEND)
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, &result);
#elif defined(CUDA_BACKEND)
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, &result);
#endif
					return result;
				}
			}
		}

		bool isDeviceAvailable(const std::string &str)
		{
#if defined(CPU_BACKEND)
			if (str == "CPU" || str == "cpu")
				return true;
#elif defined(CUDA_BACKEND)
			if (str.substr(0, 5) == "CUDA:" || str.substr(0, 5) == "cuda:")
			{
				int idx = std::atoi(str.data() + 5);
				return idx >= 0 and idx < cudaGetNumberOfDevices();
			}
#elif defined(OPENCL_BACKEND)
			if (str.substr(0, 7) == "OPENCL:" || str.substr(0, 7) == "opencl:")
			{
				int idx = std::atoi(str.data() + 7);
				return idx >= 0 and idx < openclGetNumberOfDevices();
			}
#endif
			return false;
		}
		avDataType_t dtypeFromString(const std::string &str)
		{
			if (str == "uint8")
				return AVOCADO_DTYPE_UINT8;
			if (str == "int8")
				return AVOCADO_DTYPE_INT8;
			if (str == "int16")
				return AVOCADO_DTYPE_INT16;
			if (str == "uint32")
				return AVOCADO_DTYPE_INT32;
			if (str == "uint64")
				return AVOCADO_DTYPE_INT64;
			if (str == "float16")
				return AVOCADO_DTYPE_FLOAT16;
			if (str == "bfloat16")
				return AVOCADO_DTYPE_BFLOAT16;
			if (str == "float32")
				return AVOCADO_DTYPE_FLOAT32;
			if (str == "float64")
				return AVOCADO_DTYPE_FLOAT64;
			if (str == "complex32")
				return AVOCADO_DTYPE_COMPLEX32;
			if (str == "complex64")
				return AVOCADO_DTYPE_COMPLEX64;
			return AVOCADO_DTYPE_UNKNOWN;
		}

		void initForTest(TensorWrapper &t, double offset, double minValue, double maxValue)
		{
			std::unique_ptr<char[]> tmp = std::make_unique<char[]>(t.sizeInBytes());
			switch (t.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					init_for_test<uint8_t>(tmp.get(), t.volume(), offset, 100 * minValue, 100 * maxValue);
					break;
				case AVOCADO_DTYPE_INT8:
					init_for_test<int8_t>(tmp.get(), t.volume(), offset, 100 * minValue, 100 * maxValue);
					break;
				case AVOCADO_DTYPE_INT16:
					init_for_test<int16_t>(tmp.get(), t.volume(), offset, 100 * minValue, 100 * maxValue);
					break;
				case AVOCADO_DTYPE_INT32:
					init_for_test<int32_t>(tmp.get(), t.volume(), offset, 100 * minValue, 100 * maxValue);
					break;
				case AVOCADO_DTYPE_INT64:
					init_for_test<int64_t>(tmp.get(), t.volume(), offset, 100 * minValue, 100 * maxValue);
					break;
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2 = std::make_unique<float[]>(t.volume());
					init_for_test<float>(tmp2.get(), t.volume(), offset, minValue, maxValue);
					refChangeTypeHost(0ll, tmp.get(), static_cast<avDataType_t>(t.dtype()), tmp2.get(), AVOCADO_DTYPE_FLOAT32, t.volume());
					break;
				}
				case AVOCADO_DTYPE_FLOAT32:
					init_for_test<float>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					init_for_test<double>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					init_for_test<float>(tmp.get(), 2 * t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					init_for_test<double>(tmp.get(), 2 * t.volume(), offset, minValue, maxValue);
					break;
				default:
					throw std::logic_error("initForTest() : unknown datatype '" + std::to_string(t.dtype()) + "'");
			}
			t.copyFromHost(tmp.get());
		}
		double diffForTest(const TensorWrapper &lhs, const TensorWrapper &rhs)
		{
			assert(lhs.volume() == rhs.volume());
			assert(lhs.dtype() == rhs.dtype());

			if (lhs.volume() == 0)
				return 0.0;

			std::unique_ptr<char[]> tmp_lhs = std::make_unique<char[]>(lhs.sizeInBytes());
			std::unique_ptr<char[]> tmp_rhs = std::make_unique<char[]>(rhs.sizeInBytes());
			lhs.copyToHost(tmp_lhs.get());
			rhs.copyToHost(tmp_rhs.get());
			switch (lhs.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					return diff_for_test<uint8_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT8:
					return diff_for_test<int8_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT16:
					return diff_for_test<int16_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT32:
					return diff_for_test<int32_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT64:
					return diff_for_test<int64_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2_lhs = std::make_unique<float[]>(lhs.volume());
					std::unique_ptr<float[]> tmp2_rhs = std::make_unique<float[]>(rhs.volume());
					refChangeTypeHost(0ll, tmp2_lhs.get(), AVOCADO_DTYPE_FLOAT32, tmp_lhs.get(), static_cast<avDataType_t>(lhs.dtype()),
							lhs.volume());
					refChangeTypeHost(0ll, tmp2_rhs.get(), AVOCADO_DTYPE_FLOAT32, tmp_rhs.get(), static_cast<avDataType_t>(rhs.dtype()),
							rhs.volume());
					return diff_for_test<float>(tmp2_lhs.get(), tmp2_rhs.get(), lhs.volume());
				}
				case AVOCADO_DTYPE_FLOAT32:
					return diff_for_test<float>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_FLOAT64:
					return diff_for_test<double>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_COMPLEX32:
					return diff_for_test<float>(tmp_lhs.get(), tmp_rhs.get(), 2 * lhs.volume());
				case AVOCADO_DTYPE_COMPLEX64:
					return diff_for_test<double>(tmp_lhs.get(), tmp_rhs.get(), 2 * lhs.volume());
				default:
					throw std::logic_error("diffForTest() : unknown datatype '" + std::to_string(lhs.dtype()) + "'");
			}
		}
		double normForTest(const TensorWrapper &tensor)
		{
			std::unique_ptr<char[]> tmp = std::make_unique<char[]>(tensor.sizeInBytes());
			tensor.copyToHost(tmp.get());
			switch (tensor.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					return norm_for_test<uint8_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT8:
					return norm_for_test<int8_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT16:
					return norm_for_test<int16_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT32:
					return norm_for_test<int32_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT64:
					return norm_for_test<int64_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2 = std::make_unique<float[]>(tensor.volume());
					refChangeTypeHost(0ll, tmp2.get(), AVOCADO_DTYPE_FLOAT32, tmp.get(), static_cast<avDataType_t>(tensor.dtype()), tensor.volume());
					return norm_for_test<float>(tmp2.get(), tensor.volume());
				}
				case AVOCADO_DTYPE_FLOAT32:
					return norm_for_test<float>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_FLOAT64:
					return norm_for_test<double>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_COMPLEX32:
					return norm_for_test<float>(tmp.get(), 2 * tensor.volume());
				case AVOCADO_DTYPE_COMPLEX64:
					return norm_for_test<double>(tmp.get(), 2 * tensor.volume());
				default:
					throw std::logic_error("normForTest() : unknown datatype '" + std::to_string(tensor.dtype()) + "'");
			}
		}
		void absForTest(TensorWrapper &tensor)
		{
			std::unique_ptr<char[]> tmp = std::make_unique<char[]>(tensor.sizeInBytes());
			tensor.copyToHost(tmp.get());
			switch (tensor.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					abs_for_test<uint8_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT8:
					abs_for_test<int8_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT16:
					abs_for_test<int16_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT32:
					abs_for_test<int32_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT64:
					abs_for_test<int64_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2 = std::make_unique<float[]>(tensor.volume());
					refChangeTypeHost(0ll, tmp2.get(), AVOCADO_DTYPE_FLOAT32, tmp.get(), static_cast<avDataType_t>(tensor.dtype()), tensor.volume());
					abs_for_test<float>(tmp2.get(), tensor.volume());
					break;
				}
				case AVOCADO_DTYPE_FLOAT32:
					abs_for_test<float>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_FLOAT64:
					abs_for_test<double>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					abs_for_test<float>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					abs_for_test<double>(tmp.get(), tensor.volume());
					break;
				default:
					throw std::logic_error("absForTest() : unknown datatype '" + std::to_string(tensor.dtype()) + "'");
			}
			tensor.copyFromHost(tmp.get());
		}
		template<typename T = float>
		void printForTest(const TensorWrapper &tensor)
		{
			if (tensor.numberOfDimensions() == 1)
			{
				std::cout << "------------------------------------------------------------\n";
				for (int i = 0; i < tensor.volume(); i++)
					std::cout << tensor.get<T>( { i }) << ' ';
				std::cout << '\n';
				std::cout << "------------------------------------------------------------\n";
			}
			if (tensor.numberOfDimensions() == 2)
			{
				std::cout << "------------------------------------------------------------\n";
				for (int i = 0; i < tensor.firstDim(); i++)
				{
					for (int j = 0; j < tensor.lastDim(); j++)
						std::cout << tensor.get<T>( { i, j }) << ' ';
					std::cout << '\n';
				}
				std::cout << "------------------------------------------------------------\n";
			}
			if (tensor.numberOfDimensions() == 4)
			{
				std::cout << "------------------------------------------------------------\n";
				for (int b = 0; b < tensor.firstDim(); b++)
				{
					std::cout << "--batch " << b << '\n';
					for (int f = 0; f < tensor.lastDim(); f++)
					{
						std::cout << "----channel " << f << '\n';
						for (int h = 0; h < tensor.dimension(1); h++)
						{
							for (int w = 0; w < tensor.dimension(2); w++)
								std::cout << tensor.get<T>( { b, h, w, f }) << ' ';
							std::cout << '\n';
						}
					}
				}
				std::cout << "------------------------------------------------------------\n";
			}
		}

		void initForTest(DualTensorWrapper &tensor, double offset, double minValue, double maxValue)
		{
			initForTest(tensor.getReference(), offset, minValue, maxValue);
			initForTest(tensor.getTested(), offset, minValue, maxValue);
		}
		double diffForTest(const DualTensorWrapper &tensor)
		{
			return diffForTest(tensor.getTested(), tensor.getReference());
		}

		double epsilonForTest(avDataType_t dtype)
		{
			switch (dtype)
			{
				default:
					return 0.0;
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
					return 1.0e-2;
				case AVOCADO_DTYPE_FLOAT32:
					return 1.0e-4;
				case AVOCADO_DTYPE_FLOAT64:
					return 1.0e-6;
			}
		}

	} /* namespace backend */
} /* namespace avocado */
