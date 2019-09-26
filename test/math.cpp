#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>

#include "math.hpp"

using namespace TensorScript;

int main(int argc, char *argv[]) {
	printf("%G \n",0.f);

	Tensor<float> a({3, 4}, 1);
	Tensor<float> b({3, 4}, 2);
	Tensor<float> c({3, 4});

	//float *p=new((std::aligned_storage_t)64) float[512];

	auto p = (float *) aligned_alloc(64, 512);
	free(p);
	//add(a, b, c);

	for (int i = 0; i < 12; i++) printf("%f ", c.data()[i]);
	printf("\n");
	return 0;
}