/*
 * OptimizerDescriptor.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_DESCRIPTORS_OPTIMIZERDESCRIPTOR_HPP_
#define AVOCADO_DESCRIPTORS_OPTIMIZERDESCRIPTOR_HPP_

#include <Avocado/backend_defs.h>
#include <Avocado/backend_utils.hpp>
#include <Avocado/descriptors/TensorDescriptor.hpp>

#include <string>
#include <array>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{
			class OptimizerDescriptor
			{
				public:
					static constexpr av_int64 descriptor_type = 6;
					static constexpr bool must_check_device_index = false;

					avOptimizerType_t type = AVOCADO_OPTIMIZER_SGD;
					int64_t steps = 0;
					double learning_rate = 0.0;
					std::array<double, 4> coef;
					std::array<bool, 4> flags;

					OptimizerDescriptor() noexcept;

					static std::string className();
					static avOptimizerDescriptor_t create();
					static void destroy(avOptimizerDescriptor_t desc);
					static OptimizerDescriptor& getObject(avOptimizerDescriptor_t desc);
					static bool isValid(avOptimizerDescriptor_t desc);

					void set(avOptimizerType_t optimizerType, av_int64 steps, double learningRate, const double coefficients[], const bool flags[]);
					void get(avOptimizerType_t *optimizerType, av_int64 *steps, double *learningRate, double coefficients[], bool flags[]);
					void get_workspace_size(av_int64 *result, const TensorDescriptor &wDesc) const;
			};
		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_DESCRIPTORS_OPTIMIZERDESCRIPTOR_HPP_ */
