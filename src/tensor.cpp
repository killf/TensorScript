#include <iostream>
#include "../include/tensor.hpp"

using namespace TensorScript;

int main() {
	Tensor<float> a({3, 4});
//	a.fill(10.f);
	a._data.get()[2] = 1;
	float b = a.data()[0];
	fprintf(stdout, "%f \n", b);
	return 0;
//
////
////	printf("%f \n", *a.data());
//
//	auto *ptr = a.data();
//	printf("%f ", ptr[0]);
//	for (int i = 0; i < 12; i++) std::cout << ptr[i] << " ";
//	printf("\n");
//
//	for (int i = 0; i < 12; i++) {
//		ptr[i] = (float) i * 2;
//	}
//
////	for (int i = 0; i < 3; i++) {
////		for (int j = 0; j < 4; j++) {
////			a({i, j}) = i * 4 + j;
////		}
////	}
//
//	a.fill(0);
//
//	ptr = a.data();
//	for (int i = 0; i < 12; i++) fprintf(stdout, "%f ", ptr[i]);
//	fprintf(stdout, "\n");
//	return 0;
}