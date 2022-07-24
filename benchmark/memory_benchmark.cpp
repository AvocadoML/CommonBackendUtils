/*
 * memory_benchmark.cpp
 *
 *  Created on: May 13, 2022
 *      Author: Maciej Kozarzewski
 */
#include <Avocado/testing/utils.hpp>
#include <iostream>
#include <memory>
#include <cstring>

using namespace avocado::backend;

int main(int argc, char *argv[])
{
	if (argc != 3)
		return -1;
	const double max_time = std::stod(argv[1]);
	const size_t size_in_bytes = std::stoll(argv[2]);

	std::unique_ptr<uint8_t[]> src = std::make_unique<uint8_t[]>(size_in_bytes);
	std::unique_ptr<uint8_t[]> dst = std::make_unique<uint8_t[]>(size_in_bytes);

	Timer timer;
	for (; timer.canContinue(max_time); timer++)
		std::memcpy(dst.get(), src.get(), size_in_bytes);
	std::cout << timer.get() << std::endl;
	return 0;
}

