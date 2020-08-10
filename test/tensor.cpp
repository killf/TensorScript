#include <iostream>
#include <stdlib.h>
#include "tensor.hpp"

using namespace TensorScript;

int main(int argc, char *argv[]) {
  Tensor<int> a({3, 4}, 0);

  auto *ptr = a.data();
  for (int i = 0; i < 12; i++) printf("%d ", ptr[i]);
  printf("\n");

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      a[{i, j}] = i * 4 + j;
    }
  }

  for (int i = 0; i < 12; i++) printf("%d ", ptr[i]);
  printf("\n");

  cout << a << endl;
  return 0;
}