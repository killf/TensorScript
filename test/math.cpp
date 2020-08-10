#include <iostream>
#include <cstdlib>
#include <cmath>

#include "math.hpp"

using namespace TensorScript;

int main(int argc, char *argv[]) {
  Tensor<float> a({3, 4}, 1);
  Tensor<int> b({3, 4}, 2);
  Tensor<int> c({3, 4});

  add(a, b, c);

  for (int i = 0; i < 12; i++) printf("%d ", c.data()[i]);
  printf("\n");
  return 0;
}