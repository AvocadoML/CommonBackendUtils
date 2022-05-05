/*
 * OptimizerDescriptor.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/descriptors/OptimizerDescriptor.hpp>
#include "DescriptorPool.hpp"

#include <cstring>
#include <algorithm>

namespace avocado
{
	namespace backend
	{
		namespace BACKEND_NAMESPACE
		{

			static DescriptorPool<OptimizerDescriptor> optimizer_descriptor_pool;

			OptimizerDescriptor::OptimizerDescriptor() noexcept :
					type(AVOCADO_OPTIMIZER_SGD),
					learning_rate(0.0)
			{
				coef.fill(0);
				flags.fill(false);
			}
			std::string OptimizerDescriptor::className()
			{
				return "OptimizerDescriptor";
			}
			avOptimizerDescriptor_t OptimizerDescriptor::create()
			{
				return optimizer_descriptor_pool.create();
			}
			void OptimizerDescriptor::destroy(avOptimizerDescriptor_t desc)
			{
				optimizer_descriptor_pool.destroy(desc);
			}
			OptimizerDescriptor& OptimizerDescriptor::getObject(avOptimizerDescriptor_t desc)
			{
				return optimizer_descriptor_pool.get(desc);
			}
			void OptimizerDescriptor::set(avOptimizerType_t optimizerType, av_int64 steps, double learningRate, const double coefficients[],
					const bool flags[])
			{
				if (coefficients == nullptr)
					throw std::invalid_argument("");
				if (flags == nullptr)
					throw std::invalid_argument("");

				this->type = optimizerType;
				this->steps = steps;
				this->learning_rate = learningRate;
				std::memcpy(this->coef.data(), coefficients, sizeof(this->coef));
				std::memcpy(this->flags.data(), flags, sizeof(this->flags));
			}
			void OptimizerDescriptor::get(avOptimizerType_t *optimizerType, av_int64 *steps, double *learningRate, double coefficients[],
					bool flags[])
			{
				if (optimizerType != nullptr)
					optimizerType[0] = this->type;
				if (steps != nullptr)
					steps[0] = this->steps;
				if (learningRate != nullptr)
					learningRate[0] = this->learning_rate;
				if (coefficients != nullptr)
					std::memcpy(coefficients, this->coef.data(), sizeof(this->coef));
				if (flags != nullptr)
					std::memcpy(flags, this->flags.data(), sizeof(this->flags));
			}
			void OptimizerDescriptor::get_workspace_size(av_int64 *result, const TensorDescriptor &wDesc) const
			{
				if (result == nullptr)
					throw std::invalid_argument("");
				switch (type)
				{
					case AVOCADO_OPTIMIZER_SGD:
						if (flags[0] == true)
							result[0] = wDesc.volume() * dataTypeSize(wDesc.dtype());
						else
							result[0] = 0;
						break;
					case AVOCADO_OPTIMIZER_ADAM:
						result[0] = 2 * wDesc.volume() * dataTypeSize(wDesc.dtype());
						break;
					default:
						result[0] = 0;
				}
			}

		} /* BACKEND_NAMESPACE */
	} /* namespace backend */
} /* namespace avocado */

