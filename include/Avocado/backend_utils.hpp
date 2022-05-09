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
#include <string>

#if defined(CPU_BACKEND)
#  define BACKEND_NAMESPACE cpu
#elif defined(CUDA_BACKEND)
#  define BACKEND_NAMESPACE cuda
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
				return get_device_type(lhs) == get_device_type(rhs);
			}
			template<typename T, typename U, typename ... ARGS>
			bool same_device_type(T lhs, U rhs, ARGS ... args)
			{
				if (get_device_type(lhs) == get_device_type(rhs))
					return same_device_type(lhs, args...);
				else
					return false;
			}

			struct ErrorDescription
			{
					avStatus_t status = AVOCADO_STATUS_SUCCESS;
					std::string method_name;
					std::string message;

					std::string toString() const;
			};

			ErrorDescription getLastError();
			avStatus_t reportError(avStatus_t status, const char *method, const std::string &msg);
#ifdef __GNUC__
#  define REPORT_ERROR(status, message) reportError((status), __PRETTY_FUNCTION__, (message))
#else
#  define REPORT_ERROR(status, message) reportError((status), __FUNCTION__, (message))
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
			avStatus_t destroy_descriptor(T desc)
			{
				if (T::isValid(desc) == false)
					return REPORT_ERROR(AVOCADO_STATUS_BAD_PARAM, "descriptor is invalid");
				try
				{
					T::destroy(desc);
					return AVOCADO_STATUS_SUCCESS;
				} catch (std::exception &e)
				{
					return REPORT_ERROR(AVOCADO_STATUS_FREE_FAILED, e.what());
				}
			}
		}
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_UTILS_HPP_ */
