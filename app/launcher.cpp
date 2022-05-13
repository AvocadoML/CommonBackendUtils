/*
 * launcher.cpp
 *
 *  Created on: May 5, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/backend_utils.hpp>
#include <Avocado/descriptors/TensorDescriptor.hpp>

#include <iostream>

using namespace avocado::backend;
using namespace avocado::backend::reference;

int main()
{
	avTensorDescriptor_t desc = TensorDescriptor::create( { 10, 10 }, AVOCADO_DTYPE_FLOAT32);
	std::cout << getLastError().toString() << std::endl;
	TensorDescriptor::destroy(desc);
	std::cout << getLastError().toString() << std::endl;
	return 0;
}

