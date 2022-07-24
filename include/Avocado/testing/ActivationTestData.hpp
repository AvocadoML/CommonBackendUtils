/*
 * ActivationTestData.hpp
 *
 *  Created on: May 13, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_TESTING_ACTIVATIONTESTDATA_HPP_
#define AVOCADO_TESTING_ACTIVATIONTESTDATA_HPP_

#include <Avocado/testing/utils.hpp>

namespace avocado
{
	namespace backend
	{
		class ActivationTestData
		{
			private:
				avActivationType_t act;
				TensorWrapper input;
				TensorWrapper gradientOut;
				DualTensorWrapper output;
				DualTensorWrapper gradientIn;
			public:
				ActivationTestData(avActivationType_t activation, std::initializer_list<int> shape, avDataType_t dtype);
				double runBenchmarkForward(double maxTime, const void *alpha, const void *beta) noexcept;
				double runBenchmarkBackward(double maxTime, const void *alpha, const void *beta) noexcept;

				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta) noexcept;
		};
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_TESTING_ACTIVATIONTESTDATA_HPP_ */
