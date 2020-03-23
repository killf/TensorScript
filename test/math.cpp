#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>

#include <tensor.hpp>
#include <math.hpp>

using namespace TensorScript;

void test_add() {
    Tensor<float> a({3, 4});
    Tensor<float> b({3, 4});
    Tensor<float> c({3, 4});

    a.fill(3);
    b.fill(4);

    add(a, b, c);

    for (int i = 0; i < 12; i++) printf("%f ", c.data()[i]);
    printf("\n");
}


int main(int argc, char *argv[]) {
    test_add();
    return 0;
}