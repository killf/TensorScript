#include <iostream>
#include "../include/tensor.hpp"

using namespace TensorScript;

int main() {
	Tensor<float> a({3, 4});
	a.fill(10.f);

	auto *ptr = a.data();
	for (int i = 0; i < 12; i++) printf("%f ", ptr[i]);
	printf("\n");

	for (int i = 0; i < 12; i++) {
		ptr[i] = (float) i * 2;
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			a({i, j}) = i * 4 + j;
		}
	}

	for (int i = 0; i < 12; i++) printf( "%f ", ptr[i]);
	printf("\n");
	return 0;
}