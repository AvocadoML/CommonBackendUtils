/*
 * ActivationTestData.cpp
 *
 *  Created on: May 13, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/testing/ActivationTestData.hpp>

#include <Avocado/backend_defs.h>
#include <Avocado/backend_descriptors.hpp>
#if defined(CPU_BACKEND)
#  include <Avocado/cpu_backend.h>
#  define RUN_ACTIVATION_FORWARD cpuActivationForward
#  define RUN_ACTIVATION_BACKWARD cpuActivationForward
#elif defined(CUDA_BACKEND)
#  include <Avocado/cuda_backend.h>
#  define RUN_ACTIVATION_FORWARD cudaActivationForward
#  define RUN_ACTIVATION_BACKWARD cudaActivationForward
#elif defined(OPENCL_BACKEND)
#  include <Avocado/opencl_backend.h>
#  define RUN_ACTIVATION_FORWARD openclActivationForward
#  define RUN_ACTIVATION_BACKWARD openclActivationBackward
#else
#  define RUN_ACTIVATION_FORWARD refActivationForward
#  define RUN_ACTIVATION_BACKWARD refActivationBackward
#endif
#include <Avocado/reference_backend.h>

namespace avocado
{
	namespace backend
	{
		ActivationTestData::ActivationTestData(avActivationType_t activation, std::initializer_list<int> shape, avDataType_t dtype) :
				act(activation),
				input(shape, dtype, getDevice()),
				gradientOut(shape, dtype, getDevice()),
				output(shape, dtype, getDevice()),
				gradientIn(shape, dtype, getDevice())
		{
			initForTest(input, 0.0);
			initForTest(gradientOut, 1.0);
			initForTest(output, 0.1);
			initForTest(gradientIn, 0.2);
		}
		double ActivationTestData::runBenchmarkForward(double maxTime, const void *alpha, const void *beta) noexcept
		{
			Timer timer;
			for (; timer.canContinue(maxTime); timer++)
			{
				RUN_ACTIVATION_FORWARD(getContext(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output.getTested().getDescriptor(),
						output.getTested().getMemory());
			}
			return timer.get();
		}
		double ActivationTestData::runBenchmarkBackward(double maxTime, const void *alpha, const void *beta) noexcept
		{
			Timer timer;
			for (; timer.canContinue(maxTime); timer++)
			{
				RUN_ACTIVATION_BACKWARD(getContext(), act, alpha, output.getTested().getDescriptor(), output.getTested().getMemory(),
						gradientOut.getDescriptor(), gradientOut.getMemory(), beta, gradientIn.getTested().getDescriptor(),
						gradientIn.getTested().getMemory());
			}
			return timer.get();
		}
		double ActivationTestData::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			refActivationForward(getReferenceContext(), act, alpha, input.getRefDescriptor(), input.getRefMemory(), beta,
					output.getReference().getRefDescriptor(), output.getReference().getRefMemory());

			RUN_ACTIVATION_FORWARD(getContext(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output.getTested().getDescriptor(),
					output.getTested().getMemory());

			return output.getDifference();
		}
		double ActivationTestData::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			getDifferenceForward(alpha, beta);

			refActivationBackward(getReferenceContext(), act, alpha, output.getReference().getRefDescriptor(), output.getReference().getRefMemory(),
					gradientOut.getRefDescriptor(), gradientOut.getRefMemory(), beta, gradientIn.getReference().getRefDescriptor(),
					gradientIn.getReference().getRefMemory());

			RUN_ACTIVATION_BACKWARD(getContext(), act, alpha, output.getTested().getDescriptor(), output.getTested().getMemory(),
					gradientOut.getDescriptor(), gradientOut.getMemory(), beta, gradientIn.getTested().getDescriptor(),
					gradientIn.getTested().getMemory());

			return gradientIn.getDifference();
		}

	} /* namespace backend */
} /* namespace avocado */

