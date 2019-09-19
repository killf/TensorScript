#include "../include/tensor.hpp"

using namespace TensorScript;

int main() {
	Tensor a({3, 4}, DataType::f32);
	auto *ptr = a.data<float>();
	for (int i = 0; i < 12; i++) fprintf(stdout, "%f ", ptr[i]);
	fprintf(stdout, "\n");

	for (int i = 0; i < 12; i++) {
		ptr[i] = (float) i * 2;
	}

//	for (int i = 0; i < 3; i++) {
//		for (int j = 0; j < 4; j++) {
//			fprintf(stdout, "%f ", a.at({i, j}));
//		}
//	}
	return 0;
}