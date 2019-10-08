#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>

#include "math.hpp"

using namespace TensorScript;

int main(int argc, char *argv[]) {
	Tensor a({3, 4}, 1);
	Tensor b({3, 4}, 2);
	Tensor c({3, 4});

	add(a, b, c);

	for (int i = 0; i < 12; i++) printf("%f ", c.data()[i]);
	printf("\n");
	return 0;
}