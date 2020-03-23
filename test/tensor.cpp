#include <iostream>
#include <stdlib.h>
#include "../include/tensor.hpp"

using namespace TensorScript;

int main(int argc, char *argv[]) {
    Tensor a({3, 4});

    auto *ptr = a.data<float>();
    for (int i = 0; i < 12; i++) printf("%f ", ptr[i]);
    printf("\n");

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            a[{i, j}] = i * 4 + j;
        }
    }

    for (int i = 0; i < 12; i++) printf("%f ", ptr[i]);
    printf("\n");
    return 0;
}