/*
 * wrappers.cpp
 *
 *  Created on: May 8, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/testing/wrappers.hpp>
#if defined(CPU_BACKEND)
#  include <Avocado/cpu_backend.h>
#elif defined(CUDA_BACKEND)
#  include <Avocado/cuda_backend.h>
#elif defined(OPENCL_BACKEND)
#  include <Avocado/opencl_backend.h>
#endif
#include <Avocado/reference_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include <algorithm>

namespace
{
	size_t shape_volume(const std::vector<int> &shape) noexcept
	{
		if (shape.size() == 0)
			return 0;
		size_t result = 1;
		for (size_t i = 0; i < shape.size(); i++)
			result *= shape[i];
		return result;
	}
}

namespace avocado
{
	namespace backend
	{
		ContextWrapper::ContextWrapper(avDeviceIndex_t device, bool isDefault, bool isSynchronized) :
				m_device_index(device),
				m_is_default(isDefault),
				m_is_synchronized(isDefault or isSynchronized)
		{
			if (m_is_default)
			{
#if defined(CPU_BACKEND)
				m_desc = cpuGetDefaultContext();
#elif defined(CUDA_BACKEND)
				m_desc = cudaGetDefaultContext(m_device_index);
#elif defined(OPENCL_BACKEND)
				m_desc = openclGetDefaultContext(m_device_index);
#endif
			}
			else
			{
#if defined(CPU_BACKEND)
				cpuCreateContextDescriptor(&m_desc);
#elif defined(CUDA_BACKEND)
				cudaCreateContextDescriptor(&m_desc, isSynchronized, device);
#elif defined(OPENCL_BACKEND)
				openclCreateContextDescriptor(&m_desc, isSynchronized, device);
#endif
			}
			refCreateContextDescriptor(&m_ref_desc);
		}
		ContextWrapper::ContextWrapper(ContextWrapper &&other) noexcept :
				m_desc(other.m_desc),
				m_ref_desc(other.m_ref_desc),
				m_device_index(other.m_device_index),
				m_is_default(other.m_is_default),
				m_is_synchronized(other.m_is_synchronized)
		{
			other.m_desc = AVOCADO_NULL_DESCRIPTOR;
			other.m_ref_desc = AVOCADO_NULL_DESCRIPTOR;
			other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
		}
		ContextWrapper& ContextWrapper::operator=(ContextWrapper &&other) noexcept
		{
			std::swap(this->m_desc, other.m_desc);
			std::swap(this->m_ref_desc, other.m_ref_desc);
			std::swap(this->m_device_index, other.m_device_index);
			std::swap(this->m_is_default, other.m_is_default);
			std::swap(this->m_is_synchronized, other.m_is_synchronized);
			return *this;
		}
		ContextWrapper::~ContextWrapper()
		{
			if (m_desc != AVOCADO_NULL_DESCRIPTOR and not m_is_default)
			{
#if defined(CPU_BACKEND)
				cpuDestroyContextDescriptor(m_desc);
#elif defined(CUDA_BACKEND)
				cudaDestroyContextDescriptor(m_desc);
#elif defined(OPENCL_BACKEND)
				openclDestroyContextDescriptor(m_desc);
#endif
			}
			if (m_ref_desc != AVOCADO_NULL_DESCRIPTOR)
				refDestroyContextDescriptor(m_ref_desc);
		}
		avContextDescriptor_t ContextWrapper::getDescriptor() const noexcept
		{
			return m_desc;
		}
		avContextDescriptor_t ContextWrapper::getRefDescriptor() const noexcept
		{
			return m_ref_desc;
		}
		avDeviceIndex_t ContextWrapper::getDeviceIndex() const noexcept
		{
			return m_device_index;
		}
		bool ContextWrapper::isSynchronized() const noexcept
		{
			return m_is_synchronized;
		}
		void ContextWrapper::synchronize() const
		{
#if defined(CPU_BACKEND)
			cpuSynchronizeWithContext(m_desc);
#elif defined(CUDA_BACKEND)
			cudaSynchronizeWithContext(m_desc);
#elif defined(OPENCL_BACKEND)
			openclSynchronizeWithContext(m_desc);
#endif
		}

		TensorWrapper::TensorWrapper(std::vector<int> shape, avDataType_t dtype, avDeviceIndex_t device) :
				m_device_index(device)
		{
#if defined(CPU_BACKEND)
			size_t size_in_bytes = shape_volume(shape) * cpu::dataTypeSize(dtype);
			cpuCreateTensorDescriptor(&m_tensor_descriptor);
			cpuSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			cpuCreateMemoryDescriptor(&m_memory_descriptor, size_in_bytes);
#elif defined(CUDA_BACKEND)
			size_t size_in_bytes = shape_volume(shape) * cuda::dataTypeSize(dtype);
			cudaCreateTensorDescriptor(&m_tensor_descriptor);
			cudaSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			cudaCreateMemoryDescriptor(&m_memory_descriptor, device, size_in_bytes);
#elif defined(OPENCL_BACKEND)
			size_t size_in_bytes = shape_volume(shape) * opencl::dataTypeSize(dtype);
			openclCreateTensorDescriptor(&m_tensor_descriptor);
			openclSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			openclCreateMemoryDescriptor(&m_memory_descriptor, device, size_in_bytes);
#else
			size_t size_in_bytes = shape_volume(shape) * reference::dataTypeSize(dtype);
#endif
			refCreateTensorDescriptor(&m_ref_tensor_descriptor);
			refSetTensorDescriptor(m_ref_tensor_descriptor, dtype, shape.size(), shape.data());
			refCreateMemoryDescriptor(&m_ref_memory_descriptor, size_in_bytes);

			zeroall();
		}
		TensorWrapper::TensorWrapper(TensorWrapper &&other) noexcept :
				m_device_index(other.m_device_index),
				m_tensor_descriptor(other.m_tensor_descriptor),
				m_memory_descriptor(other.m_memory_descriptor),
				m_ref_tensor_descriptor(other.m_ref_tensor_descriptor),
				m_ref_memory_descriptor(other.m_ref_memory_descriptor)
		{
			other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
			other.m_tensor_descriptor = AVOCADO_NULL_DESCRIPTOR;
			other.m_memory_descriptor = AVOCADO_NULL_DESCRIPTOR;
			other.m_ref_tensor_descriptor = AVOCADO_NULL_DESCRIPTOR;
			other.m_ref_memory_descriptor = AVOCADO_NULL_DESCRIPTOR;
		}
		TensorWrapper& TensorWrapper::operator=(TensorWrapper &&other) noexcept
		{
			std::swap(this->m_device_index, other.m_device_index);
			std::swap(this->m_tensor_descriptor, other.m_tensor_descriptor);
			std::swap(this->m_memory_descriptor, other.m_memory_descriptor);
			std::swap(this->m_ref_tensor_descriptor, other.m_ref_tensor_descriptor);
			std::swap(this->m_ref_memory_descriptor, other.m_ref_memory_descriptor);
			return *this;
		}
		TensorWrapper::~TensorWrapper() noexcept
		{
#if defined(CPU_BACKEND)
			if (m_tensor_descriptor != AVOCADO_NULL_DESCRIPTOR)
				cpuDestroyTensorDescriptor(m_tensor_descriptor);
			if (m_memory_descriptor != AVOCADO_NULL_DESCRIPTOR)
				cpuDestroyMemoryDescriptor(m_memory_descriptor);
#elif defined(CUDA_BACKEND)
			if (m_tensor_descriptor != AVOCADO_NULL_DESCRIPTOR)
				cudaDestroyTensorDescriptor(m_tensor_descriptor);
			if (m_memory_descriptor != AVOCADO_NULL_DESCRIPTOR)
				cudaDestroyMemoryDescriptor(m_memory_descriptor);
#elif defined(OPENCL_BACKEND)
			if (m_tensor_descriptor != AVOCADO_NULL_DESCRIPTOR)
			openclDestroyTensorDescriptor(m_tensor_descriptor);
			if (m_memory_descriptor != AVOCADO_NULL_DESCRIPTOR)
			openclDestroyMemoryDescriptor(m_memory_descriptor);
#endif
			if (m_ref_tensor_descriptor != AVOCADO_NULL_DESCRIPTOR)
				refDestroyTensorDescriptor(m_ref_tensor_descriptor);
			if (m_ref_memory_descriptor != AVOCADO_NULL_DESCRIPTOR)
				refDestroyMemoryDescriptor(m_ref_memory_descriptor);
		}

		avDataType_t TensorWrapper::dtype() const noexcept
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).dtype();
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).dtype();
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).dtype();
#else
			return reference::getTensor(m_tensor_descriptor).dtype();
#endif
		}
		size_t TensorWrapper::sizeInBytes() const noexcept
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).sizeInBytes();
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).sizeInBytes();
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).sizeInBytes();
#else
			return reference::getTensor(m_tensor_descriptor).sizeInBytes();
#endif
		}

		int TensorWrapper::numberOfDimensions() const noexcept
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).nbDims();
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).nbDims();
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).nbDims();
#else
			return reference::getTensor(m_tensor_descriptor).nbDims();
#endif
		}
		int TensorWrapper::dimension(int idx) const noexcept
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).dimension(idx);
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).dimension(idx);
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).dimension(idx);
#else
			return reference::getTensor(m_tensor_descriptor).dimension(idx);
#endif
		}
		int TensorWrapper::firstDim() const noexcept
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).firstDim();
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).firstDim();
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).firstDim();
#else
			return reference::getTensor(m_tensor_descriptor).firstDim();
#endif
		}
		int TensorWrapper::lastDim() const noexcept
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).lastDim();
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).lastDim();
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).lastDim();
#else
			return reference::getTensor(m_tensor_descriptor).lastDim();
#endif
		}
		int TensorWrapper::volume() const noexcept
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).volume();
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).volume();
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).volume();
#else
			return reference::getTensor(m_tensor_descriptor).volume();
#endif
		}

		void TensorWrapper::synchronize() const
		{
			switch (m_sync)
			{
				case -1: // reference is more recent, copy to device
				{
#if defined(CPU_BACKEND)
					std::memcpy(cpuGetMemoryPointer(m_memory_descriptor), refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
#elif defined(CUDA_BACKEND)
					cudaCopyMemoryFromHost(cudaGetDefaultContext(m_device_index), m_memory_descriptor, 0, refGetMemoryPointer(m_ref_memory_descriptor),
							sizeInBytes());
#elif defined(OPENCL_BACKEND)
					openclCopyMemoryFromHost(openclGetDefaultContext(m_device_index), m_memory_descriptor, 0, refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
#endif
					break;
				}
				case 0: // in sync
					break;
				case 1: // device is more recent, copy to reference
				{
#if defined(CPU_BACKEND)
					std::memcpy(refGetMemoryPointer(m_ref_memory_descriptor), cpuGetMemoryPointer(m_memory_descriptor), sizeInBytes());
#elif defined(CUDA_BACKEND)
					cudaCopyMemoryToHost(cudaGetDefaultContext(m_device_index), refGetMemoryPointer(m_ref_memory_descriptor), m_memory_descriptor, 0,
							sizeInBytes());
#elif defined(OPENCL_BACKEND)
					openclCopyMemoryToHost(openclGetDefaultContext(m_device_index), refGetMemoryPointer(m_ref_memory_descriptor),m_memory_descriptor, 0, sizeInBytes());
#endif
					break;
				}
			}
			m_sync = 0;
		}
		void TensorWrapper::zeroall()
		{
			set_pattern(nullptr, 0);
		}
		void TensorWrapper::copyToHost(void *dst) const
		{
			copy_data_to_cpu(dst, 0, sizeInBytes());
		}
		void TensorWrapper::copyFromHost(const void *src)
		{
			copy_data_from_cpu(0, src, sizeInBytes());
		}
		avTensorDescriptor_t TensorWrapper::getDescriptor() const noexcept
		{
			return m_tensor_descriptor;
		}
		avMemoryDescriptor_t TensorWrapper::getMemory() const noexcept
		{
			m_sync = 1;
			return m_memory_descriptor;
		}
		avTensorDescriptor_t TensorWrapper::getRefDescriptor() const noexcept
		{
			return m_ref_tensor_descriptor;
		}
		avMemoryDescriptor_t TensorWrapper::getRefMemory() const noexcept
		{
			m_sync = -1;
			return m_ref_memory_descriptor;
		}
		/*
		 * private
		 */
		size_t TensorWrapper::get_index(std::initializer_list<int> idx) const
		{
#if defined(CPU_BACKEND)
			return cpu::getTensor(m_tensor_descriptor).getIndex(idx);
#elif defined(CUDA_BACKEND)
			return cuda::getTensor(m_tensor_descriptor).getIndex(idx);
#elif defined(OPENCL_BACKEND)
			return opencl::getTensor(m_tensor_descriptor).getIndex(idx);
#else
			return reference::getTensor(m_tensor_descriptor).getIndex(idx);
#endif
		}
		void TensorWrapper::copy_data_to_cpu(void *dst, size_t src_offset, size_t count) const
		{
			synchronize();
			if (refGetMemoryPointer(m_ref_memory_descriptor) == nullptr)
				return;
			std::memcpy(dst, reinterpret_cast<const int8_t*>(refGetMemoryPointer(m_ref_memory_descriptor)) + src_offset, count);
		}
		void TensorWrapper::copy_data_from_cpu(size_t dst_offset, const void *src, size_t count)
		{
			if (refGetMemoryPointer(m_ref_memory_descriptor) == nullptr)
				return;
			std::memcpy(reinterpret_cast<int8_t*>(refGetMemoryPointer(m_ref_memory_descriptor)) + dst_offset, src, count);
#if defined(CPU_BACKEND)
			std::memcpy(reinterpret_cast<int8_t*>(cpuGetMemoryPointer(m_memory_descriptor)) + dst_offset, src, count);
#elif defined(CUDA_BACKEND)
			cudaCopyMemoryFromHost(cudaGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, src, count);
#elif defined(OPENCL_BACKEND)
			openclCopyMemoryFromHost(openclGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, src, count);
#endif
		}
		void TensorWrapper::set_pattern(const void *pattern, size_t patternSize)
		{
			if (refGetMemoryPointer(m_ref_memory_descriptor) == nullptr)
				return;
			refSetMemory(0, m_ref_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#if defined(CPU_BACKEND)
			cpuSetMemory(cpuGetDefaultContext(), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#elif defined(CUDA_BACKEND)
			cudaSetMemory(cudaGetDefaultContext(m_device_index), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#elif defined(OPENCL_BACKEND)
			openclSetMemory(openclGetDefaultContext(m_device_index), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#endif
		}

		OptimizerWrapper::OptimizerWrapper(avDeviceIndex_t device)
		{
#if defined(CPU_BACKEND)
			cpuCreateOptimizerDescriptor(&m_desc);
#elif defined(CUDA_BACKEND)
			cudaCreateOptimizerDescriptor(&m_desc);
#elif defined(OPENCL_BACKEND)
			openclCreateOptimizerDescriptor(&m_desc);
#endif
			refCreateOptimizerDescriptor(&m_ref_desc);
		}
		OptimizerWrapper::OptimizerWrapper(OptimizerWrapper &&other) noexcept :
				m_desc(other.m_desc),
				m_ref_desc(other.m_ref_desc)
		{
			other.m_desc = AVOCADO_NULL_DESCRIPTOR;
			other.m_ref_desc = AVOCADO_NULL_DESCRIPTOR;
		}
		OptimizerWrapper& OptimizerWrapper::operator=(OptimizerWrapper &&other) noexcept
		{
			std::swap(this->m_desc, other.m_desc);
			std::swap(this->m_ref_desc, other.m_ref_desc);
			return *this;
		}
		OptimizerWrapper::~OptimizerWrapper()
		{
			if (m_desc != AVOCADO_NULL_DESCRIPTOR)
			{
#if defined(CPU_BACKEND)
				cpuDestroyOptimizerDescriptor(m_desc);
#elif defined(CUDA_BACKEND)
				cudaDestroyOptimizerDescriptor(m_desc);
#elif defined(OPENCL_BACKEND)
				openclDestroyOptimizerDescriptor(m_desc);
#endif
			}
			if (m_ref_desc != AVOCADO_NULL_DESCRIPTOR)
				refDestroyOptimizerDescriptor(m_ref_desc);
		}
		avOptimizerDescriptor_t OptimizerWrapper::getDescriptor() const noexcept
		{
			return m_desc;
		}
		avOptimizerDescriptor_t OptimizerWrapper::getRefDescriptor() const noexcept
		{
			return m_ref_desc;
		}
		void OptimizerWrapper::set(avOptimizerType_t type, int64_t steps, double learningRate, const std::array<double, 4> &coefficients,
				const std::array<bool, 4> &flags)
		{
#if defined(CPU_BACKEND)
			cpuSetOptimizerDescriptor(m_desc, type, steps, learningRate, coefficients.data(), flags.data());
#elif defined(CUDA_BACKEND)
			cudaSetOptimizerDescriptor(m_desc, type, steps, learningRate, coefficients.data(), flags.data());
#elif defined(OPENCL_BACKEND)
			openclSetOptimizerDescriptor(m_desc, type, steps, learningRate, coefficients.data(), flags.data());
#endif
			refSetOptimizerDescriptor(m_ref_desc, type, steps, learningRate, coefficients.data(), flags.data());
		}
		size_t OptimizerWrapper::getWorkspaceSize(const TensorWrapper &weights)
		{
			av_int64 result;
			refGetOptimizerWorkspaceSize(m_ref_desc, weights.getRefDescriptor(), &result);
			return result;
		}

		ConvolutionWrapper::ConvolutionWrapper(avDeviceIndex_t device, int nbDims) :
				nbDims(nbDims)
		{
#if defined(CPU_BACKEND)
			cpuCreateConvolutionDescriptor(&m_desc);
#elif defined(CUDA_BACKEND)
			cudaCreateConvolutionDescriptor(&m_desc);
#elif defined(OPENCL_BACKEND)
			openclCreateConvolutionDescriptor(&m_desc);
#endif
			refCreateConvolutionDescriptor(&m_ref_desc);
		}
		ConvolutionWrapper::ConvolutionWrapper(ConvolutionWrapper &&other) noexcept :
				nbDims(other.nbDims),
				m_desc(other.m_desc),
				m_ref_desc(other.m_ref_desc)
		{
			other.m_desc = AVOCADO_NULL_DESCRIPTOR;
			other.m_ref_desc = AVOCADO_NULL_DESCRIPTOR;
		}
		ConvolutionWrapper& ConvolutionWrapper::operator=(ConvolutionWrapper &&other) noexcept
		{
			std::swap(this->nbDims, other.nbDims);
			std::swap(this->m_desc, other.m_desc);
			std::swap(this->m_ref_desc, other.m_ref_desc);
			return *this;
		}
		ConvolutionWrapper::~ConvolutionWrapper()
		{
			if (m_desc != AVOCADO_NULL_DESCRIPTOR)
			{
#if defined(CPU_BACKEND)
				cpuDestroyConvolutionDescriptor(m_desc);
#elif defined(CUDA_BACKEND)
				cudaDestroyConvolutionDescriptor(m_desc);
#elif defined(OPENCL_BACKEND)
				openclDestroyConvolutionDescriptor(m_desc);
#endif
			}
			if (m_ref_desc != AVOCADO_NULL_DESCRIPTOR)
				refDestroyConvolutionDescriptor(m_ref_desc);
		}
		avConvolutionDescriptor_t ConvolutionWrapper::getDescriptor() const noexcept
		{
			return m_desc;
		}
		avConvolutionDescriptor_t ConvolutionWrapper::getRefDescriptor() const noexcept
		{
			return m_ref_desc;
		}
		void ConvolutionWrapper::set(avConvolutionMode_t mode, const std::array<int, 3> &padding, const std::array<int, 3> &strides,
				const std::array<int, 3> &dilation, int groups, const void *paddingValue)
		{
#if defined(CPU_BACKEND)
			cpuSetConvolutionDescriptor(m_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
#elif defined(CUDA_BACKEND)
			cudaSetConvolutionDescriptor(m_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
#elif defined(OPENCL_BACKEND)
			openclSetConvolutionDescriptor(m_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
#endif
			refSetConvolutionDescriptor(m_ref_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
		}
		std::vector<int> ConvolutionWrapper::getOutputShape(const TensorWrapper &input, const TensorWrapper &weights)
		{
#if defined(CPU_BACKEND)
			cpu::TensorDescriptor tmp = cpu::getConvolution(m_desc).getOutputShape(cpu::getTensor(input.getDescriptor()),
					cpu::getTensor(weights.getDescriptor()));
#elif defined(CUDA_BACKEND)
			cuda::TensorDescriptor tmp = cuda::getConvolution(m_desc).getOutputShape(cuda::getTensor(input.getDescriptor()),
					cuda::getTensor(weights.getDescriptor()));
#elif defined(OPENCL_BACKEND)
			opencl::TensorDescriptor tmp = opencl::getConvolution(m_desc).getOutputShape(opencl::getTensor(input.getDescriptor()),
					opencl::getTensor(weights.getDescriptor()));
#else
			reference::TensorDescriptor tmp = reference::getConvolution(m_desc).getOutputShape(reference::getTensor(input.getDescriptor()),
					reference::getTensor(weights.getDescriptor()));
#endif
			int size;
			tmp.get(nullptr, &size, nullptr);
			std::vector<int> result(size);
			tmp.get(nullptr, nullptr, result.data());
			return result;
		}
	} /* namespace backend */
} /* namespace avocado */

