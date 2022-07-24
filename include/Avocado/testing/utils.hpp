/*
 * utils.hpp
 *
 *  Created on: May 8, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_TESTING_UTILS_HPP_
#define AVOCADO_TESTING_UTILS_HPP_

#include <Avocado/backend_defs.h>
#include <Avocado/testing/wrappers.hpp>

#include <chrono>
#include <iostream>

namespace avocado
{
	namespace backend
	{
		class DualTensorWrapper
		{
				TensorWrapper reference;
				TensorWrapper tested;
			public:
				DualTensorWrapper(std::vector<int> shape, avDataType_t dtype, avDeviceIndex_t device);
				const TensorWrapper& getReference() const;
				TensorWrapper& getReference();
				const TensorWrapper& getTested() const;
				TensorWrapper& getTested();
				double getDifference() const;
		};

		class Timer
		{
				double m_timer_start = 0.0;
				double m_timer_stop = 0.0;
				uint64_t m_total_count = 0;
			public:
				Timer() :
						m_timer_start(getTime())
				{
				}
				static double getTime() noexcept
				{
					return std::chrono::steady_clock::now().time_since_epoch().count() * 1.0e-9;
				}
				double get() const noexcept
				{
					return (getTime() - m_timer_start) / m_total_count;
				}
				Timer& operator ++(int i) noexcept
				{
					m_total_count++;
					return *this;
				}
				bool canContinue(double maxTime) const noexcept
				{
					return (getTime() - m_timer_start) < maxTime;
				}
		};

//		void setMasterContext(avDeviceIndex_t deviceIndex, bool useDefault);
//		const ContextWrapper& getMasterContext();
		avContextDescriptor_t getContext();
		avContextDescriptor_t getReferenceContext();
		avDeviceIndex_t getDevice();

		bool supportsType(avDataType_t dtype);
		bool isDeviceAvailable(const std::string &str);
		avDataType_t dtypeFromString(const std::string &str);

		void initForTest(TensorWrapper &t, double offset, double minValue = -1.0, double maxValue = 1.0);
		double diffForTest(const TensorWrapper &lhs, const TensorWrapper &rhs);
		double normForTest(const TensorWrapper &tensor);
		void absForTest(TensorWrapper &tensor);

		void initForTest(DualTensorWrapper &t, double offset, double minValue = -1.0, double maxValue = 1.0);
		double diffForTest(const DualTensorWrapper &tensor);

		double epsilonForTest(avDataType_t dtype);

	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_TESTING_UTILS_HPP_ */
