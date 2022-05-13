/*
 * backend_utils.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_UTILS_HPP_
#define AVOCADO_BACKEND_UTILS_HPP_

#include "backend_defs.h"

#include <cassert>
#include <stdexcept>
#include <string>

#if defined(CPU_BACKEND)
#  define BACKEND_NAMESPACE cpu
#elif defined(CUDA_BACKEND)
#  define BACKEND_NAMESPACE cuda
#  include <cuda_runtime_api.h>
#  include <cublas_v2.h>
#elif defined(OPENCL_BACKEND)
#  define BACKEND_NAMESPACE opencl
#else
#  define BACKEND_NAMESPACE reference
#endif

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{
			std::string descriptorTypeToString(av_int64 desc);
			std::string statusToString(avStatus_t status);
			std::string dtypeToString(avDataType_t dtype);
			std::string deviceTypeToString(avDeviceType_t type);

			int getNumberOfDevices() noexcept;
			avDeviceType_t getDeviceType(av_int64 descriptor) noexcept;
			int getDescriptorType(av_int64 descriptor) noexcept;
			avDeviceIndex_t getDeviceIndex(av_int64 descriptor) noexcept;
			int getDescriptorIndex(av_int64 descriptor) noexcept;

			avDeviceType_t getCurrentDeviceType() noexcept;
			av_int64 getCurrentDeviceIndex() noexcept;

			av_int64 createDescriptor(int index, av_int64 type) noexcept;

			int dataTypeSize(avDataType_t dtype) noexcept;

			template<typename T>
			void setScalarValue(void *scalar, T x) noexcept
			{
				assert(scalar != nullptr);
				reinterpret_cast<T*>(scalar)[0] = x;
			}

			template<typename T>
			T getScalarValue(const void *scalar) noexcept
			{
				assert(scalar != nullptr);
				return reinterpret_cast<const T*>(scalar)[0];
			}

			template<typename T = float>
			T getAlphaValue(const void *alpha) noexcept
			{
				if (alpha == nullptr)
					return static_cast<T>(1);
				else
					return reinterpret_cast<const T*>(alpha)[0];
			}
			template<typename T = float>
			T getBetaValue(const void *beta) noexcept
			{
				if (beta == nullptr)
					return static_cast<T>(0);
				else
					return reinterpret_cast<const T*>(beta)[0];
			}

			template<>
			av_complex32 getAlphaValue<av_complex32>(const void *alpha) noexcept;
			template<>
			av_complex32 getBetaValue<av_complex32>(const void *beta) noexcept;
			template<>
			av_complex64 getAlphaValue<av_complex64>(const void *alpha) noexcept;
			template<>
			av_complex64 getBetaValue<av_complex64>(const void *beta) noexcept;

#ifdef CUDA_BACKEND
			template<>
			cuComplex getAlphaValue<cuComplex>(const void *alpha) noexcept;
			template<>
			cuComplex getBetaValue<cuComplex>(const void *beta) noexcept;
			template<>
			cuDoubleComplex getAlphaValue<cuDoubleComplex>(const void *alpha) noexcept;
			template<>
			cuDoubleComplex getBetaValue<cuDoubleComplex>(const void *beta) noexcept;
#endif

			struct BroadcastedDimensions
			{
					int first;
					int last;
			};

			bool is_transpose(avGemmOperation_t op) noexcept;
			bool is_logical(avBinaryOp_t op) noexcept;
			bool is_logical(avUnaryOp_t op) noexcept;
			bool is_logical(avReduceOp_t op) noexcept;

			template<typename T, typename U>
			bool same_device_type(T lhs, U rhs)
			{
				return getDeviceType(lhs) == getDeviceType(rhs);
			}
			template<typename T, typename U, typename ... ARGS>
			bool same_device_type(T lhs, U rhs, ARGS ... args)
			{
				if (getDeviceType(lhs) == getDeviceType(rhs))
					return same_device_type(lhs, args...);
				else
					return false;
			}

			class InvalidDescriptorError: public std::logic_error
			{
				public:
					InvalidDescriptorError(const std::string &msg) :
							std::logic_error(msg)
					{
					}
			};
			class DeallocationError: public std::runtime_error
			{
				public:
					DeallocationError(const std::string &msg) :
							std::runtime_error(msg)
					{
					}
			};

			struct ErrorDescription
			{
					avStatus_t status = AVOCADO_STATUS_SUCCESS;
					std::string method_name;
					std::string message;

					std::string toString() const;
			};

			/*
			 * \brief Returns last error description and resets internal storage for errors.
			 */
			ErrorDescription getLastError();
			/*
			 * \brief Returns last error description without resetting the internal storage for errors.
			 */
			ErrorDescription peekLastError();
			/*
			 * \brief Registers an error with its description.
			 */
			avStatus_t reportError(avStatus_t status, const char *method, const std::string &msg);

#ifdef __GNUC__
#  define REPORT_ERROR(status, message) reportError((status), __PRETTY_FUNCTION__, (message))
#else
#  define REPORT_ERROR(status, message) reportError((status), __FUNCTION__, (message))
#endif

#ifdef CUDA_BACKEND
			std::runtime_error cuda_runtime_error_creator(const char* methodName, cudaError_t status);
			std::runtime_error cublas_runtime_error_creator(const char* methodName, cublasStatus_t status);
#  ifdef __GNUC__
#    define CHECK_CUDA_ERROR(status) if (status != cudaSuccess) throw cuda_runtime_error_creator(__PRETTY_FUNCTION__, status)
#    define CHECK_CUBLAS_STATUS(status) if (status != cudaSuccess) throw cublas_runtime_error_creator(__PRETTY_FUNCTION__, status)
#  else
#    define CHECK_CUDA_ERROR(status) if (status != cudaSuccess) throw cuda_runtime_error_creator(__FUNCTION__, status)
#    define CHECK_CUBLAS_STATUS(status) if (status != cudaSuccess) throw cublas_runtime_error_creator(__FUNCTION__, status)
#  endif
#endif

			template<typename T, typename U, typename ... Args>
			avStatus_t create_descriptor(U *result, Args &&... args)
			{
				if (result == nullptr)
					return REPORT_ERROR(AVOCADO_STATUS_BAD_PARAM, "result is null");
				try
				{
					result[0] = T::create(std::forward<Args>(args)...);
					return AVOCADO_STATUS_SUCCESS;
				} catch (std::exception &e)
				{
					return REPORT_ERROR(AVOCADO_STATUS_INTERNAL_ERROR, e.what());
				}
			}
			template<typename T, typename U>
			avStatus_t destroy_descriptor(U desc)
			{
				try
				{
					T::destroy(desc);
					return AVOCADO_STATUS_SUCCESS;
				} catch (DeallocationError &e)
				{
					return REPORT_ERROR(AVOCADO_STATUS_FREE_FAILED, e.what());
				} catch (InvalidDescriptorError &e)
				{
					return REPORT_ERROR(AVOCADO_STATUS_BAD_PARAM, e.what());
				} catch (std::exception &e)
				{
					return REPORT_ERROR(AVOCADO_STATUS_INTERNAL_ERROR, e.what());
				}
			}
		}
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_UTILS_HPP_ */
